"""E8's result from the Parquet logs: for each arm, the goal rate of the final checkpoint's
evaluation from the fixed seg_05 state, as an IQM over seeds, and the first checkpoint whose
evaluation reached the goal in each seed. Every line gives the seed count, and an arm with fewer
than five seeds is labelled exploratory.

    uv run python scripts/e8_summary.py [--runs data/runs]
"""

from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import tyro

from algoarcade.training.stats import HEADLINE_SEEDS, describe, describe_improvement


def score(evaluations: list[dict[str, Any]]) -> tuple[float, int | None]:
    """The goal rate at the last checkpoint, and the first checkpoint with any goal (None if
    no evaluation reached it)."""
    by_checkpoint: dict[int, list[bool]] = defaultdict(list)
    for row in evaluations:
        by_checkpoint[row["checkpoint_step"]].append(row["goal_reached"])
    rates = {step: sum(goals) / len(goals) for step, goals in sorted(by_checkpoint.items())}
    first = next((step for step, rate in rates.items() if rate > 0), None)
    return rates[max(rates)], first


def scores(runs: Path, experiment: str = "E8") -> dict[str, list[tuple[float, int | None]]]:
    """Each finished run of `experiment`, scored, by arm."""
    arms: dict[str, list[tuple[float, int | None]]] = defaultdict(list)
    for logs in sorted(runs.glob(f"run_id={experiment}-*")):
        run = pq.read_table(logs / "runs").to_pylist()[-1]
        if run["experiment_id"] != experiment or run["status"] != "finished":
            continue
        episodes = pq.read_table(logs / "episodes").to_pylist()
        arms[run["arm"]].append(score([row for row in episodes if row["is_eval"]]))
    return dict(arms)


def summary(arms: dict[str, list[tuple[float, int | None]]]) -> str:
    lines = []
    for arm, runs in sorted(arms.items()):
        n = len(runs)
        label = "" if n >= HEADLINE_SEEDS else ", exploratory"
        firsts = ", ".join("never" if first is None else f"{first:,}" for _, first in runs)
        lines.append(
            f"{arm}: final eval goal rate {describe([rate for rate, _ in runs])}; "
            f"first goal at {firsts} steps; {n} seeds{label}"
        )
    if {"curriculum", "none"} <= arms.keys():
        curriculum = [rate for rate, _ in arms["curriculum"]]
        none = [rate for rate, _ in arms["none"]]
        lines.append(f"curriculum over none: {describe_improvement(curriculum, none)}")
    return "\n".join(lines)


def main(runs: Path = Path("data/runs"), experiment: str = "E8") -> None:
    print(summary(scores(runs, experiment)))


if __name__ == "__main__":
    tyro.cli(main)
