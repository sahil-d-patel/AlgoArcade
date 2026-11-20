"""What each success cost, RL against the LLM: one row per (agent, segment), every cell from the
Parquet logs. Contract: docs/technical-plan.md, LLM arm and Visualization.

An RL agent is an experiment's arm, over its seeds. A seed's first success is the earliest
checkpoint at which an evaluation episode reached the start state's goal: its training steps are
that checkpoint's, and its hours run from the run's start to the end of that checkpoint's last
evaluation episode. The row has the median over the seeds that succeeded, and the median goal step
of the successful evaluation episodes at those checkpoints. An LLM agent is a model and tool level
in E7, summarised by agent.grid.cell_rows.

    uv run python -m algoarcade.viz.cost_table
"""

import statistics
from typing import Any

import duckdb

from algoarcade.agent.grid import cell_rows
from algoarcade.viz.site import connect

EVALS = """
SELECT r.experiment_id, r.arm, e.run_id, e.start_state, e.checkpoint_step, e.wall_clock_s,
    e.goal_reached, e.goal_step
FROM episodes e JOIN (SELECT DISTINCT run_id, experiment_id, arm FROM runs) r USING (run_id)
WHERE e.is_eval
"""
ATTEMPTS = """
SELECT * FROM episodes
WHERE run_id IN (SELECT run_id FROM runs WHERE experiment_id = 'E7' AND status = 'finished')
"""


def rl_rows(con: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    seeds: dict[tuple[str, str, str], dict[str, list[tuple[Any, ...]]]] = {}
    for experiment, arm, run, segment, *episode in con.sql(EVALS).fetchall():
        seeds.setdefault((experiment, arm, segment), {}).setdefault(run, []).append(tuple(episode))
    rows = []
    for (experiment, arm, segment), runs in seeds.items():
        steps, hours, presses = [], [], []
        for episodes in runs.values():
            successes = [step for step, _, reached, _ in episodes if reached]
            if not successes:
                continue
            first = min(successes)
            at_first = [episode for episode in episodes if episode[0] == first]
            steps.append(first)
            hours.append(max(clock for _, clock, _, _ in at_first) / 3600)
            presses += [goal for _, _, reached, goal in at_first if reached]
        rows.append(
            {
                "agent": f"{experiment} {arm}",
                "segment": segment,
                "tried": len(runs),
                "unit": "seeds",
                "successes": len(steps),
                "steps_tried": max(
                    episode[0] for episodes in runs.values() for episode in episodes
                ),
                "steps": statistics.median(steps) if steps else None,
                "hours": statistics.median(hours) if hours else None,
                "presses": statistics.median(presses) if presses else None,
            }
        )
    return rows


def llm_rows(con: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    attempts = con.sql(ATTEMPTS)
    episodes = [dict(zip(attempts.columns, row, strict=True)) for row in attempts.fetchall()]
    return [
        {
            "agent": f"{cell['model']} L{cell['level']}",
            "segment": cell["segment"],
            "tried": cell["attempts"],
            "unit": "attempts",
            "successes": cell["successes"],
            "interval": cell["interval"],
            "presses": cell["presses_to_goal"],
            "per_success": cell["per_success"],
        }
        for cell in cell_rows(episodes)
    ]


def rows(con: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    return sorted(rl_rows(con) + llm_rows(con), key=lambda row: (row["segment"], row["agent"]))


def markdown(table: list[dict[str, Any]]) -> str:
    """RL rows have no interval or dollars and LLM rows no training: those cells read n/a. A
    cost an agent never paid off reads none."""
    lines = [
        "| Agent | Segment | Tried | Succeeded | Success rate [95% CI] "
        "| Training steps to first success | GPU-hours to first success "
        "| Median presses to goal | Dollars per success |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in table:
        rl = row["unit"] == "seeds"
        if rl:
            rate = per_success = "n/a"
            done = row["steps"] is not None
            steps = f"{row['steps']:,.0f}" if done else f"none in {row['steps_tried']:,} steps"
            hours = f"{row['hours']:.2f}" if done else "none"
        else:
            low, high = row["interval"]
            rate = f"{row['successes'] / row['tried']:.2f} [{low:.2f}, {high:.2f}]"
            per = row["per_success"]
            per_success = "none" if per is None else f"${per:.2f}"
            steps = hours = "n/a"
        presses = "none" if row["presses"] is None else f"{row['presses']:g}"
        unit = row["unit"] if row["tried"] != 1 else row["unit"].removesuffix("s")
        lines.append(
            f"| {row['agent']} | {row['segment']} | {row['tried']} {unit} "
            f"| {row['successes']} | {rate} | {steps} | {hours} | {presses} | {per_success} |"
        )
    return "\n".join(lines)


def caption(con: duckdb.DuckDBPyConnection) -> str:
    machines = con.sql(
        f"SELECT DISTINCT machine FROM runs WHERE run_id IN (SELECT run_id FROM ({EVALS})) "
        "ORDER BY machine"
    ).fetchall()
    names = ", ".join(name for (name,) in machines) or "none yet"
    return (
        "First success is the earliest checkpoint at which an evaluation episode from the "
        "segment's start state reached its goal; runs evaluate every eval_every steps, so it is "
        "exact to that interval. Training steps are the environment steps at "
        "that checkpoint. GPU-hours are wall-clock hours from the run's start to the end of that "
        "checkpoint's evaluation, on the machine each run logged "
        f"({names}); the training box is the Windows desktop with an RTX-class GPU. RL cells "
        "are medians over the seeds that succeeded. LLM rows are the E7 attempts: an exact "
        "binomial 95% interval, and dollars per success divides every attempt's spend, failures "
        "included, by the successes."
    )


if __name__ == "__main__":
    con = connect()
    print(f"{markdown(rows(con))}\n\n{caption(con)}")  # noqa: T201
