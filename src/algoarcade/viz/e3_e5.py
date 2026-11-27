"""The E3 and E5 tables (VIZ-05), with the statistics from algoarcade.training.stats.

    uv run python -m algoarcade.viz.e3_e5

A run's score is the mean `milestone_count` over the evaluation episodes at its last checkpoint.
The arms pay different rewards (E3's none has no explore term, RND's bonus is not in the logs,
E5 adds penalties), so the return cannot compare them; milestones are the task's progress. Only
finished runs count. Contract: docs/technical-plan.md, Experiments and Statistics.
"""

import argparse
from pathlib import Path

import duckdb

from algoarcade.training import stats
from algoarcade.viz.site import RUNS, connect

# E3's count arm is the shared baseline arm, queued by experiments/baseline.py.
COUNT = ("baseline", "v1")
E3 = {"none": ("E3", "none"), "count": COUNT, "rnd": ("E3", "rnd")}
E5 = {"positive-only": ("E5", "positive-only"), "plus-penalties": ("E5", "plus-penalties")}

type Scores = list[tuple[int, float]]


def scores(con: duckdb.DuckDBPyConnection, experiment_id: str, arm: str) -> Scores:
    """(seed, score) for each finished run of the arm, by seed."""
    return con.execute(
        """
        WITH finished AS (
            SELECT run_id, seed FROM runs
            WHERE status = 'finished' AND experiment_id = ? AND arm = ?
        ),
        evals AS (
            SELECT run_id, checkpoint_step, milestone_count FROM episodes
            WHERE is_eval AND run_id IN (SELECT run_id FROM finished)
        )
        SELECT seed, avg(milestone_count) FROM evals JOIN finished USING (run_id)
        WHERE (run_id, checkpoint_step) IN (
            SELECT (run_id, max(checkpoint_step)) FROM evals GROUP BY run_id
        )
        GROUP BY run_id, seed ORDER BY seed, run_id
        """,
        [experiment_id, arm],
    ).fetchall()


def table(title: str, arms: dict[str, Scores], comparisons: list[tuple[str, str]]) -> str:
    """A Markdown table of each arm's seeds, IQM and 95% interval, then P(X > Y) for each
    (X, Y) in `comparisons`. Below five seeds in any arm the title says exploratory; below
    four the IQM trims nothing, so the column says Mean."""
    fewest = min(len(runs) for runs in arms.values())
    label = ", exploratory: under five seeds" if fewest < stats.HEADLINE_SEEDS else ""
    centre = "Mean" if fewest < stats.TRIMMED else "IQM"
    lines = [
        f"## {title}{label}",
        "",
        f"| Arm | n | {centre} | 95% interval | Score by seed |",
        "|---|---|---|---|---|",
    ]
    for arm, runs in arms.items():
        if not runs:
            lines.append(f"| {arm} | 0 | | | |")
            continue
        values = [score for _, score in runs]
        low, high = stats.interval(values)
        by_seed = ", ".join(f"{seed}: {score:.2f}" for seed, score in runs)
        lines.append(
            f"| {arm} | {len(runs)} | {stats.iqm(values):.2f} | [{low:.2f}, {high:.2f}] "
            f"| {by_seed} |"
        )
    lines += ["", "| Comparison | Probability of improvement |", "|---|---|"]
    for x, y in comparisons:
        empty = [arm for arm in (x, y) if not arms[arm]]
        if empty:
            result = f"no finished runs in {' and '.join(empty)}"
        else:
            result = stats.describe_improvement(
                [score for _, score in arms[x]], [score for _, score in arms[y]]
            )
            if sorted(seed for seed, _ in arms[x]) != sorted(seed for seed, _ in arms[y]):
                result += ", exploratory: seed sets differ"
        lines.append(f"| {x} over {y} | {result} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m algoarcade.viz.e3_e5", description=__doc__)
    parser.add_argument("--root", type=Path, default=RUNS)
    args = parser.parse_args(argv)
    con = connect(args.root)
    e3 = {arm: scores(con, *key) for arm, key in E3.items()}
    e5 = {arm: scores(con, *key) for arm, key in E5.items()}
    print(table("E3: does learned novelty beat counts", e3, [("count", "none"), ("rnd", "count")]))  # noqa: T201
    print(table("E5: do penalties help or hurt", e5, [("plus-penalties", "positive-only")]))  # noqa: T201


if __name__ == "__main__":
    main()
