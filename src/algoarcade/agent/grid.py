"""E7: attempts of the LLM loop on every (model, tool level) cell and segment, and their summary.

Contract: docs/technical-plan.md, LLM arm. Each attempt is `loop.run` with its own server process
and run id `<experiment>-<model>-L<level>-<segment>-<index>`, logged under data/llm/<experiment>/
and data/runs/. The grid goes round by round, one more attempt for each cell per round, so a
budget stop leaves the cells even. It resumes from what is on disk: an attempt whose runs end row
says finished counts, any other is skipped as crashed, and the next index runs. Before each
attempt the grid adds one attempt's dollar cap to the spend so far and stops if that would pass
the budget. A crashed attempt counts its attempt.json total if it wrote one; otherwise the larger
of its logged requests and the cap it ran under, from its runs row, since the request in flight
when it died was never logged. Every attempt of a cell must start from the start state hash of
the cell's first finished attempt; the loop refuses another before its first request.
"""

import asyncio
import json
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import tyro
from anthropic import AsyncAnthropic
from scipy.stats import binomtest

from algoarcade.agent.loop import Create, Prices, Run, run
from algoarcade.logs.writer import run_dir

# Only cards with a source. Add a model's card, with the date read, before running its cells.
PRICES = {
    "claude-haiku-5-5": Prices(),
    # Anthropic's model table on 2026-11-14: $4 input, $20 output, $0.20 cache read per million
    # tokens, with no long-prompt card. It lists no cache-write rate; this assumes 1.25x input,
    # as on Haiku's card.
    "claude-opus-5-5": Prices(
        input=4.0,
        output=20.0,
        cache_write=5.0,
        cache_read=0.20,
        long_input=4.0,
        long_output=20.0,
        long_cache_write=5.0,
        long_cache_read=0.20,
    ),
}
CELLS = (
    ("claude-haiku-5-5", 0),
    ("claude-haiku-5-5", 1),
    ("claude-haiku-5-5", 2),
    ("claude-opus-5-5", 2),
)
SEGMENTS = ("seg_01_house", "seg_02_pallet", "seg_03_rival")


@dataclass(frozen=True, kw_only=True)
class Grid:
    """Run the grid. `max_dollars` caps each attempt and `budget` the whole grid, resumes
    included."""

    max_dollars: float
    budget: float
    attempts: int = 10
    max_actions: int = 300
    cells: tuple[tuple[str, int], ...] = CELLS
    segments: tuple[str, ...] = SEGMENTS
    experiment: str = "E7"
    out: Path = Path("data/llm")
    runs: Path = Path("data/runs")


def run_row(runs: Path, name: str) -> dict[str, Any] | None:
    """The run's current row in the runs table, or None before its first row closed."""
    table = run_dir(name, runs) / "runs"
    rows = pq.read_table(table).to_pylist() if table.is_dir() else []
    return rows[-1] if rows else None


def finished(runs: Path, name: str) -> bool:
    row = run_row(runs, name)
    return row is not None and row["status"] == "finished"


def spent(folder: Path, runs: Path) -> tuple[float, float]:
    """What an attempt logged, and what it counts against the budget. One with attempt.json,
    finished or not, made all its requests and counts its total."""
    summary = folder / "attempt.json"
    if summary.is_file():
        dollars = json.loads(summary.read_text())["dollars"]
        return dollars, dollars
    steps = folder / "steps.jsonl"
    lines = steps.read_text().splitlines() if steps.is_file() else []
    logged = sum(json.loads(line)["dollars"] for line in lines)
    row = run_row(runs, folder.name)
    # The loop writes the runs row before its first request, so without one nothing was sent.
    cap = json.loads(row["hyperparameters"])["max_dollars"] if row else 0.0
    return logged, max(logged, cap)


async def run_grid(grid: Grid, create: Create) -> float:
    """Run attempts until every cell has `attempts` finished or the budget would be passed.
    Returns the spend, each crashed attempt counted at its cap or more."""
    missing = sorted({model for model, _ in grid.cells} - PRICES.keys())
    if missing:
        raise ValueError(f"no prices for {', '.join(missing)}: add each card to PRICES in grid.py")
    folder = grid.out / grid.experiment
    folder.mkdir(parents=True, exist_ok=True)
    total = sum(spent(path, grid.runs)[1] for path in folder.iterdir() if path.is_dir())
    for done in range(grid.attempts):
        for segment in grid.segments:
            for model, level in grid.cells:
                arm = f"{model}-L{level}"
                prefix = f"{grid.experiment}-{arm}-{segment}-"
                tried = sorted(folder.glob(f"{prefix}*"))
                done_here = [path for path in tried if finished(grid.runs, path.name)]
                if len(done_here) > done:
                    continue
                if total + grid.max_dollars > grid.budget:
                    print(f"stopped: ${total:.4f} spent, one more attempt could pass the budget")  # noqa: T201
                    return total
                settings = Run(
                    name=f"{prefix}{len(tried):02d}",
                    segment=segment,
                    experiment_id=grid.experiment,
                    arm=arm,
                    level=level,
                    model=model,
                    prices=PRICES[model],
                    max_actions=grid.max_actions,
                    max_dollars=grid.max_dollars,
                    out=folder,
                    runs=grid.runs,
                    start_sha256=start_hash(done_here[0]) if done_here else "",
                )
                attempt = await run(settings, create)
                total += attempt.dollars
                print(  # noqa: T201
                    f"{settings.name}: {attempt.outcome}, {attempt.actions} presses, "
                    f"${attempt.dollars:.4f}; grid ${total:.4f}"
                )
    return total


def start_hash(folder: Path) -> str:
    return json.loads((folder / "attempt.json").read_text())["start_sha256"]


def results(runs: Path, experiment: str) -> list[dict[str, Any]]:
    """The episode row of every finished attempt of `experiment`, with its arm."""
    rows = []
    for logs in sorted(runs.glob(f"run_id={experiment}-*")):
        run_row = pq.read_table(logs / "runs").to_pylist()[-1]
        if run_row["experiment_id"] != experiment or run_row["status"] != "finished":
            continue
        (episode,) = pq.read_table(logs / "episodes").to_pylist()
        rows.append(episode | {"arm": run_row["arm"]})
    return rows


def cell_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One summary per (model, level, segment). The interval is Clopper-Pearson at 95%. Dollars
    per success divides every attempt's spend, failed ones included, by the successes."""
    cells: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["llm_model"], row["llm_tool_level"], row["start_state"])
        cells.setdefault(key, []).append(row)
    summaries = []
    for (model, level, segment), attempts in sorted(cells.items()):
        n = len(attempts)
        steps = [row["goal_step"] for row in attempts if row["goal_reached"]]
        dollars = sum(row["llm_cost_usd"] for row in attempts)
        interval = binomtest(len(steps), n).proportion_ci(method="exact")
        summaries.append(
            {
                "model": model,
                "level": level,
                "segment": segment,
                "attempts": n,
                "successes": len(steps),
                "interval": (interval.low, interval.high),
                "presses_to_goal": statistics.median(steps) if steps else None,
                "tokens_in": sum(row["llm_tokens_in"] for row in attempts) / n,
                "tokens_out": sum(row["llm_tokens_out"] for row in attempts) / n,
                "dollars": dollars,
                "per_success": dollars / len(steps) if steps else None,
            }
        )
    return summaries


def table(cells: list[dict[str, Any]]) -> str:
    lines = [
        "| Model | Level | Segment | Attempts | Successes | Rate [95% CI] | Median presses to goal "
        "| Mean tokens in | Mean tokens out | Dollars | Dollars per success |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for cell in cells:
        low, high = cell["interval"]
        rate = cell["successes"] / cell["attempts"]
        presses = cell["presses_to_goal"]
        per_success = cell["per_success"]
        lines.append(
            f"| {cell['model']} | {cell['level']} | {cell['segment']} | {cell['attempts']} "
            f"| {cell['successes']} | {rate:.2f} [{low:.2f}, {high:.2f}] "
            f"| {'none' if presses is None else f'{presses:g}'} | {cell['tokens_in']:.0f} "
            f"| {cell['tokens_out']:.0f} | ${cell['dollars']:.2f} "
            f"| {'none' if per_success is None else f'${per_success:.2f}'} |"
        )
    return "\n".join(lines)


@dataclass(frozen=True, kw_only=True)
class Summary:
    """Print the summary per cell from the Parquet logs, the transcripts of those attempts, and
    the crashed attempts with what they logged and count against the budget."""

    experiment: str = "E7"
    out: Path = Path("data/llm")
    runs: Path = Path("data/runs")

    def render(self) -> str:
        rows = results(self.runs, self.experiment)
        folder = self.out / self.experiment
        names = {row["run_id"] for row in rows}
        transcripts = sum((folder / name / "transcript.json").is_file() for name in names)
        dollars = sum(row["llm_cost_usd"] for row in rows)
        crashed = [
            spent(path, self.runs)
            for path in (folder.iterdir() if folder.is_dir() else [])
            if path.is_dir() and path.name not in names
        ]
        logged = sum(pair[0] for pair in crashed)
        budget = sum(pair[1] for pair in crashed)
        footer = (
            f"{len(rows)} attempts, {transcripts} transcripts, ${dollars:.4f} spent; "
            f"{len(crashed)} crashed, ${logged:.4f} logged, ${budget:.4f} counted against the budget"
        )
        return f"{table(cell_rows(rows))}\n\n{footer}"


def main() -> None:
    command = tyro.extras.subcommand_cli_from_dict({"run": Grid, "summary": Summary})
    if isinstance(command, Grid):
        # Without a key the first attempt would crash after making its folders, and a crashed
        # attempt counts against the budget at its cap.
        if "ANTHROPIC_API_KEY" not in os.environ:
            raise SystemExit("set ANTHROPIC_API_KEY first")
        asyncio.run(run_grid(command, AsyncAnthropic().messages.create))
    else:
        print(command.render())  # noqa: T201


if __name__ == "__main__":
    main()
