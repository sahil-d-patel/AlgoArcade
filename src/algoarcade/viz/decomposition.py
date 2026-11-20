"""What each reward component paid per training episode, per checkpoint window (VIZ-05).

A component's share of an episode is its weight times the raw total the episodes table logs, so
the shares of an episode sum to its `episode_return`. Bars are raw values stacked from zero:
penalties go below the axis, so the stack never normalises mixed signs. Contract:
docs/harness-spec.md, Logging schema (Reward columns).

Check the sum on one run, printing the largest gap over its windows:

    uv run python -m algoarcade.viz.decomposition --run <run_id>
"""

import argparse
import json
from pathlib import Path

import duckdb
import plotly.graph_objects as go
import pyarrow as pa
from plotly.colors import qualitative
from plotly.subplots import make_subplots

from algoarcade.logs.schema import REWARD_COMPONENTS
from algoarcade.viz.site import RUNS, connect

# The checkpoint and visit-table window (docs/harness-spec.md, Visit-count table).
WINDOW_STEPS = 500_000
COLORS = dict(zip(REWARD_COMPONENTS, qualitative.Plotly, strict=True))


def runs_of(con: duckdb.DuckDBPyConnection, experiment_id: str, arm: str) -> list[str]:
    rows = con.execute(
        "SELECT DISTINCT run_id FROM runs WHERE experiment_id = ? AND arm = ? ORDER BY run_id",
        [experiment_id, arm],
    ).fetchall()
    return [run_id for (run_id,) in rows]


def decompose(
    con: duckdb.DuckDBPyConnection, run_ids: list[str]
) -> tuple[list[int], list[float], dict[str, list[float]]]:
    """Window starts, the mean episode return in each window, and each weighted component's mean
    per episode in each window, over the training episodes of `run_ids` pooled."""
    hyperparameters = con.execute(
        "SELECT DISTINCT run_id, hyperparameters FROM runs WHERE list_contains(?, run_id)",
        [run_ids],
    ).fetchall()
    weights = pa.Table.from_pylist(
        [
            {"run_id": run_id, "name": f"reward_{name}", "weight": weight}
            for run_id, text in hyperparameters
            for name, weight in json.loads(text)["reward_weights"].items()
        ],
        schema=pa.schema(
            [("run_id", pa.string()), ("name", pa.string()), ("weight", pa.float64())]
        ),
    )
    con.register("weights", weights)
    columns = ", ".join(f"reward_{name}" for name in REWARD_COMPONENTS)
    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW windowed AS
        SELECT *, (env_steps - 1) // {WINDOW_STEPS} * {WINDOW_STEPS} AS window_start
        FROM episodes WHERE NOT is_eval AND run_id IN (SELECT run_id FROM weights)
        """
    )
    totals = con.sql(
        "SELECT window_start, count(*), avg(episode_return) FROM windowed GROUP BY 1 ORDER BY 1"
    ).fetchall()
    starts = [start for start, _, _ in totals]
    episodes = {start: count for start, count, _ in totals}
    sums = con.sql(
        f"""
        SELECT window_start, name, sum(weight * raw)
        FROM (UNPIVOT windowed ON {columns} INTO NAME name VALUE raw)
        JOIN weights USING (run_id, name)
        GROUP BY ALL
        """
    ).fetchall()
    shares = {(start, name.removeprefix("reward_")): total for start, name, total in sums}
    names = [name for name in REWARD_COMPONENTS if any(key[1] == name for key in shares)]
    parts = {
        name: [shares.get((start, name), 0.0) / episodes[start] for start in starts]
        for name in names
    }
    return starts, [mean for _, _, mean in totals], parts


def largest_gap(con: duckdb.DuckDBPyConnection, run_id: str) -> float:
    """The largest difference, over one run's windows, between its stacked components and its
    mean episode return."""
    _, returns, parts = decompose(con, [run_id])
    return max(
        abs(sum(values[i] for values in parts.values()) - mean) for i, mean in enumerate(returns)
    )


def figure(con: duckdb.DuckDBPyConnection, arms: dict[str, tuple[str, str]]) -> go.Figure:
    """One panel per arm, `arms` mapping its label to (experiment_id, arm), each pooling the
    arm's seeds. The markers are the mean episode return the bars must stack to."""
    run_ids = {label: runs_of(con, *key) for label, key in arms.items()}
    fig = make_subplots(
        cols=len(arms),
        shared_yaxes=True,
        subplot_titles=[f"{label} (n={len(ids)})" for label, ids in run_ids.items()],
    )
    shown = set()
    for col, ids in enumerate(run_ids.values(), start=1):
        if not ids:
            continue
        starts, returns, parts = decompose(con, ids)
        for name, values in parts.items():
            fig.add_bar(
                x=starts,
                y=values,
                name=name,
                legendgroup=name,
                showlegend=name not in shown,
                marker_color=COLORS[name],
                col=col,
                row=1,
            )
            shown.add(name)
        fig.add_scatter(
            x=starts,
            y=returns,
            mode="markers",
            name="episode return",
            legendgroup="return",
            showlegend="return" not in shown,
            marker={"color": "black", "symbol": "line-ew-open", "size": 14},
            col=col,
            row=1,
        )
        shown.add("return")
    fig.update_layout(
        barmode="relative",
        title="Mean reward per training episode by component, per checkpoint window",
        yaxis_title="weighted reward per episode",
        height=520,
        margin={"t": 110},
    )
    fig.update_xaxes(title="window start, steps")
    return fig


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m algoarcade.viz.decomposition")
    parser.add_argument("--run", required=True)
    parser.add_argument("--root", type=Path, default=RUNS)
    args = parser.parse_args(argv)
    con = connect(args.root)
    starts, _, _ = decompose(con, [args.run])
    gap = largest_gap(con, args.run)
    print(f"{args.run}: {len(starts)} windows, largest |components - return| {gap:.3g}")  # noqa: T201


if __name__ == "__main__":
    main()
