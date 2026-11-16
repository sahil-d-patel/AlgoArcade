"""First sampled progress at or beyond each FireRed stage, one row per seed."""

from html import escape
from pathlib import Path

import plotly.graph_objects as go
import pyarrow as pa
import pyarrow.parquet as pq
import tyro

from algoarcade.games.firered.milestones import MILESTONES
from algoarcade.logs.schema import STEPS, validate

COLUMNS = (
    "run_id",
    "seed",
    "env_idx",
    "episode_id",
    "episode_step",
    "env_steps",
    "milestone_count",
)
STAGES = [name.replace("_", " ").capitalize() for name in MILESTONES]


def first_observed(steps: pa.Table, run: dict) -> dict[int, tuple[int, bool]]:
    fields = pa.schema([STEPS.field(name) for name in COLUMNS])
    selected = steps.select(COLUMNS)
    if not selected.schema.equals(fields) or any(col.null_count for col in selected.columns):
        raise ValueError("Timeline step columns must match the frozen schema and contain no nulls")
    first_sample: dict[tuple[int, int], int] = {}
    earliest: dict[int, tuple[int, int, int, int]] = {}
    for batch in selected.to_batches(max_chunksize=65_536):
        for row in batch.to_pylist():
            if row["run_id"] != run["run_id"] or row["seed"] != run["seed"]:
                raise ValueError("Timeline rows belong to another run or seed")
            step, local = row["env_steps"], row["episode_step"]
            env, episode, stage = row["env_idx"], row["episode_id"], row["milestone_count"]
            if not (0 < step <= run["env_steps"] and 0 < local <= step):
                raise ValueError("Timeline step lies outside the completed run")
            if not (0 <= env < run["num_envs"] and episode >= 0 and 0 <= stage <= len(STAGES)):
                raise ValueError("Invalid environment, episode or progress stage")
            key = (env, episode)
            first_sample[key] = min(first_sample.get(key, local), local)
            candidate = (step, env, episode, local)
            if stage and (stage not in earliest or candidate < earliest[stage]):
                earliest[stage] = candidate
    result = {}
    for stage in range(1, len(STAGES) + 1):
        candidates = [value for count, value in earliest.items() if count >= stage]
        if candidates:
            step, env, episode, local = min(candidates)
            result[stage] = (step, local == first_sample[(env, episode)])
    return result


def timeline(records: list[tuple[dict, dict[int, tuple[int, bool]]]]) -> go.Figure:
    if not records:
        raise ValueError("Select at least one run")
    if len({run["seed"] for run, _ in records}) != len(records):
        raise ValueError("Select only one run per seed; do not pool separate runs")
    context = (
        "experiment_id",
        "arm",
        "env_id",
        "algorithm",
        "trainer",
        "start_state",
        "obs_mode",
        "reward_config_hash",
    )
    if len({tuple(run[field] for field in context) for run, _ in records}) != 1:
        raise ValueError(
            "Timeline runs must share experiment, arm, game, trainer, start state and reward config"
        )
    if any(run["env_id"] not in ("firered", "fake") for run, _ in records):
        raise ValueError("The stage timeline requires FireRed or synthetic FireRed logs")
    fig = go.Figure()
    records = sorted(records, key=lambda record: record[0]["seed"])
    points: dict[bool, list[tuple[int, str, str]]] = {True: [], False: []}
    for run, observed in records:
        label = f"Seed {run['seed']}"
        fig.add_trace(
            go.Scatter(
                x=[0, run["env_steps"]],
                y=[label, label],
                mode="lines",
                line={"color": "#d5d9df", "width": 2},
                hoverinfo="skip",
                showlegend=False,
            )
        )
        grouped: dict[tuple[int, bool], list[int]] = {}
        for stage, location in observed.items():
            grouped.setdefault(location, []).append(stage)
        for (step, initial), stages in grouped.items():
            names = "<br>".join(f"{stage}. {STAGES[stage - 1]}" for stage in stages)
            detail = (
                f"Run: {escape(run['run_id'])}<br>Start state: {escape(run['start_state'])}<br>"
                f"First recorded at/above:<br>{names}<br>"
                f"Logging interval: every {run['step_log_every']} episode steps"
            )
            points[initial].append((step, label, detail))
        missing = ", ".join(STAGES[i - 1] for i in range(1, len(STAGES) + 1) if i not in observed)
        fig.add_trace(
            go.Scatter(
                x=[run["env_steps"]],
                y=[label],
                mode="markers",
                marker={"symbol": "line-ns", "size": 16, "line": {"width": 2, "color": "#717b88"}},
                text=[missing or "None"],
                hovertemplate="Run end: %{x:,} steps<br>Stages not observed: %{text}<extra></extra>",
                showlegend=False,
            )
        )
    for initial, name, symbol, color in (
        (True, "Present at first episode sample", "diamond-open", "#c27719"),
        (False, "First observed later in episode", "circle", "#087f8c"),
    ):
        entries = points[initial]
        fig.add_trace(
            go.Scatter(
                x=[p[0] for p in entries],
                y=[p[1] for p in entries],
                text=[p[2] for p in entries],
                mode="markers",
                name=name,
                marker={"symbol": symbol, "color": color, "size": 13, "line": {"width": 2}},
                hovertemplate="%{y} · logged step %{x:,}<br>%{text}<extra>%{fullData.name}</extra>",
            )
        )
    synthetic = records[0][0]["env_id"] == "fake"
    fig.update_layout(
        title={
            "text": (
                "Synthetic progress · demonstration only"
                if synthetic
                else "First recorded progress by seed"
            )
            + "<br><sup>At or beyond each stage; not exact event-completion times</sup>",
            "x": 0.03,
        },
        height=max(460, 230 + 65 * len(records)),
        margin={"l": 85, "r": 40, "t": 95, "b": 150},
        xaxis={
            "title": "Training steps across all environments",
            "range": [
                -0.03 * max(1, max(r["env_steps"] for r, _ in records)),
                1.03 * max(1, max(r["env_steps"] for r, _ in records)),
            ],
            "tickformat": ",",
        },
        yaxis={
            "categoryorder": "array",
            "categoryarray": [f"Seed {r['seed']}" for r, _ in records],
            "autorange": "reversed",
        },
        legend={"orientation": "h", "y": -0.25, "x": 0},
        paper_bgcolor="#faf9f5",
        plot_bgcolor="#faf9f5",
        font={"family": "Arial", "size": 13},
    )
    fig.add_annotation(
        x=0,
        y=-0.45,
        xref="paper",
        yref="paper",
        xanchor="left",
        showarrow=False,
        text="First-sample progress may predate a reset or come from a save state.<br>"
        "No marker means not observed in sampled logs. Earlier individual events are not proven.",
    )
    return fig


def main(runs: list[Path], out: Path) -> None:
    records = []
    for directory in runs:
        files = sorted((directory / "runs").glob("*.parquet"))
        if not files:
            raise ValueError(f"No run metadata in {directory}")
        metadata = pa.concat_tables([pq.ParquetFile(path).read() for path in files])
        validate(metadata, "runs")
        rows = metadata.to_pylist()
        completed = [row for row in rows if row["status"] == "finished"]
        if len({row["run_id"] for row in rows}) != 1 or len(completed) != 1:
            raise ValueError("Each path must hold exactly one finished run")
        files = sorted((directory / "steps").glob("*.parquet"))
        steps = (
            pa.concat_tables([pq.ParquetFile(path).read(columns=COLUMNS) for path in files])
            if files
            else pa.Table.from_pylist([], schema=pa.schema([STEPS.field(name) for name in COLUMNS]))
        )
        records.append((completed[0], first_observed(steps, completed[0])))
    figure = timeline(records)
    figure.write_html(out, include_plotlyjs=True, auto_play=False, config={"displaylogo": False})
    print(  # noqa: T201
        f"Saved recorded progress for {len(records)} seeds to {out} ({out.stat().st_size:,} bytes)"
    )


if __name__ == "__main__":
    tyro.cli(main)
