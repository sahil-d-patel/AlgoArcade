"""E1, E2 and E4 against the shared baseline arm (VIZ-04, #42): one table and four figures per
experiment, every number from algoarcade.training.stats. Contract: docs/technical-plan.md,
Experiments and Visualization.

    uv run python -m algoarcade.viz.comparison [--root data/runs]

A run's score at a checkpoint is the mean milestone_count over that checkpoint's evaluation
episodes, the metric experiments/e1_pqn.py fixed before any run. Its final score is its last
checkpoint's. Only finished runs count; the table states how many were left out.
"""

import warnings
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import plotly.graph_objects as go
import tyro
from numpy.typing import ArrayLike, NDArray
from plotly.colors import DEFAULT_PLOTLY_COLORS

from algoarcade.training import stats
from algoarcade.viz.site import RUNS, connect

# The experiment id of the shared baseline arm (PPO, symbolic, count, archive off, from S1).
# OPS-06 (#34) queues it and has not named it yet: a guess, to check against `runqueue status`.
BASELINE = "baseline"
EXPERIMENTS = ("E1", "E2", "E4")
METRIC = "milestone_count"

# A run's current row is its end row when it has one (docs/harness-spec.md, Per-run table).
CURRENT = """
    SELECT * FROM runs
    QUALIFY row_number() OVER (PARTITION BY run_id ORDER BY status = 'running') = 1
"""


@dataclass(frozen=True)
class Arm:
    seeds: list[int]
    machines: list[str]
    steps: NDArray[np.int64]
    """The checkpoints every seed reached."""
    hours: NDArray[np.float64]
    """(seeds, checkpoints): when each checkpoint's evaluation finished."""
    scores: NDArray[np.float64]
    """(seeds, checkpoints)."""


def load(con: duckdb.DuckDBPyConnection, experiment: str) -> dict[str, Arm]:
    """The experiment's finished runs by arm, the baseline first as "baseline"."""
    rows = con.sql(
        f"""
        WITH finished AS (
            SELECT run_id, seed, machine,
                CASE WHEN experiment_id = $baseline THEN 'baseline' ELSE arm END AS arm
            FROM ({CURRENT})
            WHERE status = 'finished' AND experiment_id IN ($baseline, $experiment)
        )
        SELECT arm, finished.seed, machine, checkpoint_step, avg({METRIC}), max(wall_clock_s) / 3600
        FROM episodes JOIN finished USING (run_id)
        WHERE is_eval
        GROUP BY 1, 2, 3, 4
        ORDER BY arm != 'baseline', 1, 2, 4
        """,
        params={"baseline": BASELINE, "experiment": experiment},
    ).fetchall()
    runs: dict[str, dict[tuple[int, str], dict[int, tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for arm, seed, machine, step, score, hours in rows:
        runs[arm][seed, machine][step] = (score, hours)
    arms = {}
    for name, seeds in runs.items():
        steps = sorted(set.intersection(*(set(points) for points in seeds.values())))
        points = np.array([[run[step] for step in steps] for run in seeds.values()])
        arms[name] = Arm(
            seeds=[seed for seed, _ in seeds],
            machines=sorted({machine for _, machine in seeds}),
            steps=np.array(steps),
            hours=points[:, :, 1],
            scores=points[:, :, 0],
        )
    return arms


def _label(name: str, seeds: int) -> str:
    exploratory = ", exploratory" if seeds < stats.HEADLINE_SEEDS else ""
    return f"{name} (n={seeds}{exploratory})"


def improvement(arms: Mapping[str, Arm]) -> dict[str, float]:
    """P(a run of the arm scores above a baseline run), on final scores, for each other arm."""
    baseline = arms["baseline"].scores[:, -1]
    return {
        name: stats.probability_of_improvement(arm.scores[:, -1], baseline)
        for name, arm in arms.items()
        if name != "baseline"
    }


def iqm_curve(
    curves: Mapping[str, tuple[ArrayLike, ArrayLike]], title: str, x_title: str, y_title: str
) -> go.Figure:
    """One line per name: at each x, the IQM over seeds and its 95% bootstrap band. A curve is
    (x, scores) with scores shaped (seeds, len(x)). The legend gives each seed count and labels
    a curve under five seeds exploratory."""
    figure = go.Figure(layout={"title": title, "xaxis_title": x_title, "yaxis_title": y_title})
    for i, (name, (x, scores)) in enumerate(curves.items()):
        columns = np.asarray(scores, dtype=np.float64).T
        bands = np.array([stats.interval(column) for column in columns])
        color = DEFAULT_PLOTLY_COLORS[i % len(DEFAULT_PLOTLY_COLORS)]
        label = _label(name, columns.shape[1])
        band = {"x": x, "mode": "lines", "line_width": 0, "legendgroup": label}
        figure.add_scatter(y=bands[:, 1], showlegend=False, hoverinfo="skip", **band)
        fill = color.replace("rgb", "rgba").replace(")", ", 0.2)")
        figure.add_scatter(y=bands[:, 0], fill="tonexty", fillcolor=fill, name=label, **band)
        iqm = [stats.iqm(column) for column in columns]
        figure.add_scatter(x=x, y=iqm, name=label, line_color=color, legendgroup=label)
    return figure


def figures(arms: Mapping[str, Arm]) -> list[go.Figure]:
    """The IQM curve against steps and against hours, the performance profile, and, when the
    baseline ran, the probability of improvement over it."""
    y_title = f"eval {METRIC}, IQM"
    steps = iqm_curve(
        {name: (arm.steps, arm.scores) for name, arm in arms.items()},
        "IQM learning curve, 95% bootstrap band",
        "environment steps",
        y_title,
    )
    on = {name: ", ".join(arm.machines) for name, arm in arms.items()}
    machines = set(on.values())
    where = machines.pop() if len(machines) == 1 else "; ".join(f"{k}: {v}" for k, v in on.items())
    hours = iqm_curve(
        {name: (np.median(arm.hours, axis=0), arm.scores) for name, arm in arms.items()},
        "IQM against wall-clock time, each checkpoint at its median hour over seeds",
        f"wall-clock hours on {where}",
        y_title,
    )

    finals = np.concatenate([arm.scores[:, -1] for arm in arms.values()])
    thresholds = np.linspace(finals.min(), finals.max(), 101)
    profile = go.Figure(
        layout={
            "title": "Performance profile of final scores",
            "xaxis_title": f"threshold, final eval {METRIC}",
            "yaxis_title": "fraction of runs above the threshold",
        }
    )
    for name, arm in arms.items():
        fraction = stats.performance_profile(arm.scores[:, -1], thresholds)
        profile.add_scatter(
            x=thresholds, y=fraction, line_shape="hv", name=_label(name, len(arm.seeds))
        )

    result = [steps, hours, profile]
    if "baseline" in arms:
        n = len(arms["baseline"].seeds)
        names = [_label(name, len(arms[name].seeds)) for name in arms if name != "baseline"]
        p = list(improvement(arms).values())
        bars = go.Figure(
            layout={
                "title": f"Probability of improvement over {_label('baseline', n)}",
                "xaxis": {
                    "title": "P(a run of the arm scores above a baseline run)",
                    "range": [0, 1],
                },
            }
        )
        bars.add_bar(x=p, y=names, orientation="h", text=[f"{v:.2f}" for v in p])
        bars.add_vline(x=0.5, line_dash="dash")
        result.append(bars)
    return result


def table(con: duckdb.DuckDBPyConnection, experiment: str) -> str:
    """Markdown: per arm, the seed count, IQM, 95% interval, probability of improvement over the
    baseline and every seed's final score."""
    [(finished, unfinished)] = con.sql(
        f"""
        SELECT count(*) FILTER (status = 'finished'), count(*) FILTER (status != 'finished')
        FROM ({CURRENT}) WHERE experiment_id IN ($baseline, $experiment)
        """,
        params={"baseline": BASELINE, "experiment": experiment},
    ).fetchall()
    lines = [
        f"## {experiment}",
        "",
        f"Score: mean eval {METRIC} at each run's last checkpoint. "
        f"Finished runs: {finished}. Unfinished, left out: {unfinished}.",
        "",
    ]
    arms = load(con, experiment)
    if not arms:
        return "\n".join([*lines, "No finished runs.", ""])
    p = improvement(arms) if "baseline" in arms else {}
    lines += [
        "| Arm | n | IQM | 95% interval | P(arm > baseline) | Final score by seed |",
        "|---|---|---|---|---|---|",
    ]
    for name, arm in arms.items():
        final = arm.scores[:, -1]
        low, high = stats.interval(final)
        n = str(len(final)) + (", exploratory" if len(final) < stats.HEADLINE_SEEDS else "")
        improves = f"{p[name]:.2f}" if name in p else ""
        by_seed = ", ".join(f"{s}: {v:.2f}" for s, v in zip(arm.seeds, final, strict=True))
        lines.append(
            f"| {name} | {n} | {stats.iqm(final):.2f} | [{low:.2f}, {high:.2f}] | {improves} "
            f"| {by_seed} |"
        )
    if len({tuple(arm.seeds) for arm in arms.values()}) > 1:
        lines += ["", "Seed sets differ between arms: compare with care."]
    return "\n".join([*lines, ""])


def main(root: Path = RUNS) -> None:
    con = connect(root)
    with warnings.catch_warnings():
        # The table labels every arm under five seeds; the per-call warning repeats it.
        warnings.simplefilter("ignore", UserWarning)
        print("\n".join(table(con, experiment) for experiment in EXPERIMENTS))  # noqa: T201


if __name__ == "__main__":
    tyro.cli(main)
