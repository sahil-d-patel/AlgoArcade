import asyncio
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest
from anthropic.types import Message, ToolUseBlock, Usage

from algoarcade.agent import loop
from algoarcade.agent.grid import Grid, Summary, cell_rows, results, run_grid, table

# $0.001 a request at Haiku's short card.
USAGE = Usage(input_tokens=10_000, output_tokens=0)


class Presser:
    """A model that presses the same buttons on every request, and raises on the request after
    `crash_after`."""

    def __init__(self, buttons: list[str], crash_after: int | None = None) -> None:
        self.buttons = buttons
        self.crash_after = crash_after
        self.requests = 0

    async def __call__(self, **_: Any) -> Message:
        if self.requests == self.crash_after:
            raise RuntimeError("the API went away")
        self.requests += 1
        block = ToolUseBlock(
            type="tool_use", id="toolu_1", name="press", input={"buttons": self.buttons}
        )
        return Message(
            id="msg_1",
            type="message",
            role="assistant",
            model="stub",
            content=[block],
            stop_reason="tool_use",
            stop_sequence=None,
            usage=USAGE,
        )


def grid(tmp_path: Path, **settings: Any) -> Grid:
    defaults: dict[str, Any] = {
        "max_dollars": 0.05,
        "budget": 1.0,
        "attempts": 2,
        "max_actions": 5,
        "cells": (("claude-haiku-5-5", 0), ("claude-haiku-5-5", 2)),
        "segments": ("fake",),
        "out": tmp_path / "llm",
        "runs": tmp_path / "runs",
    }
    return Grid(**(defaults | settings))


def attempts(settings: Grid) -> dict[str, dict[str, Any]]:
    folder = settings.out / settings.experiment
    return {
        path.name: json.loads((path / "attempt.json").read_text())
        for path in sorted(folder.iterdir())
        if (path / "attempt.json").is_file()
    }


def test_two_attempts_in_a_row_get_their_own_server_and_the_same_start(tmp_path: Path) -> None:
    settings = grid(tmp_path, cells=(("claude-haiku-5-5", 1),))
    asyncio.run(run_grid(settings, Presser(["RIGHT"])))
    first, second = attempts(settings).values()
    assert first["server_pid"] != second["server_pid"]
    assert first["start_sha256"] == second["start_sha256"]


def test_the_action_cap_stops_an_attempt_at_5_and_grades_it_failed(tmp_path: Path) -> None:
    # RIGHT never opens the chest: that takes A.
    settings = grid(tmp_path, attempts=1, cells=(("claude-haiku-5-5", 1),))
    asyncio.run(run_grid(settings, Presser(["RIGHT", "RIGHT"])))
    (summary,) = attempts(settings).values()
    assert (summary["outcome"], summary["actions"]) == ("action_cap", 5)
    (row,) = results(settings.runs, settings.experiment)
    assert (row["goal_reached"], row["end_reason"], row["episode_length"]) == (
        False,
        "action_cap",
        5,
    )


def test_the_grid_runs_every_cell_and_logs_a_transcript_per_row(tmp_path: Path) -> None:
    settings = grid(tmp_path)
    # RIGHT, RIGHT, A opens the chest from (1, 1) on the third press.
    asyncio.run(run_grid(settings, Presser(["RIGHT", "RIGHT", "A"])))
    rows = results(settings.runs, settings.experiment)
    assert len(rows) == 4
    transcripts = list((settings.out / settings.experiment).glob("*/transcript.json"))
    assert len(transcripts) == len(rows)
    assert {(row["arm"], row["llm_tool_level"]) for row in rows} == {
        ("claude-haiku-5-5-L0", 0),
        ("claude-haiku-5-5-L2", 2),
    }
    assert all(row["goal_step"] == 3 for row in rows)
    runs_row = pq.read_table(settings.runs / f"run_id={rows[0]['run_id']}" / "runs").to_pylist()
    assert runs_row[-1]["experiment_id"] == "E7"


def test_a_transcript_keeps_the_messages_and_drops_the_image_data(tmp_path: Path) -> None:
    settings = grid(tmp_path, attempts=1, cells=(("claude-haiku-5-5", 0),), max_actions=1)
    asyncio.run(run_grid(settings, Presser(["RIGHT"])))
    (path,) = (settings.out / settings.experiment).glob("*/transcript.json")
    task, call, result = json.loads(path.read_text())
    assert task["content"].startswith("Open the chest")
    assert call["content"][0]["input"] == {"buttons": ["RIGHT"]}
    (image,) = result["content"][0]["content"]
    assert image["source"]["data"].endswith("base64 characters>")


def test_the_grid_resumes_where_it_stopped(tmp_path: Path) -> None:
    model = Presser(["RIGHT"])
    asyncio.run(run_grid(grid(tmp_path, attempts=1), model))
    assert model.requests == 2 * 5
    asyncio.run(run_grid(grid(tmp_path, attempts=2), model))
    assert model.requests == 4 * 5
    assert len(attempts(grid(tmp_path))) == 4


ONE_CELL = (("claude-haiku-5-5", 1),)


def crash(tmp_path: Path, max_dollars: float) -> None:
    """One attempt that made two $0.001 requests and died on the third."""
    settings = grid(tmp_path, attempts=1, cells=ONE_CELL, max_dollars=max_dollars)
    with pytest.raises(ExceptionGroup):
        asyncio.run(run_grid(settings, Presser(["RIGHT"], crash_after=2)))


def test_a_crashed_attempt_is_skipped_and_counted_at_its_cap(tmp_path: Path) -> None:
    crash(tmp_path, max_dollars=0.05)
    # The crash counts $0.05, so another $0.05 attempt would pass $0.06.
    model = Presser(["RIGHT"])
    spent = asyncio.run(run_grid(grid(tmp_path, attempts=1, cells=ONE_CELL, budget=0.06), model))
    assert (model.requests, spent) == (0, pytest.approx(0.05))
    settings = grid(tmp_path, attempts=1, cells=ONE_CELL)
    spent = asyncio.run(run_grid(settings, model))
    assert model.requests == 5
    assert spent == pytest.approx(0.05 + 0.005)
    assert list(attempts(settings)) == ["E7-claude-haiku-5-5-L1-fake-01"]


def test_a_crash_counts_the_cap_it_ran_under_after_a_resume_with_a_lower_cap(
    tmp_path: Path,
) -> None:
    crash(tmp_path, max_dollars=0.05)
    # At the new $0.01 cap the crash would count $0.01 and leave room for an attempt.
    model = Presser(["RIGHT"])
    settings = grid(tmp_path, attempts=1, cells=ONE_CELL, max_dollars=0.01, budget=0.055)
    spent = asyncio.run(run_grid(settings, model))
    assert (model.requests, spent) == (0, pytest.approx(0.05))


def test_the_summary_reports_crashed_attempts_and_their_spend(tmp_path: Path) -> None:
    crash(tmp_path, max_dollars=0.05)
    settings = grid(tmp_path, attempts=1, cells=ONE_CELL)
    asyncio.run(run_grid(settings, Presser(["RIGHT"])))
    footer = Summary(out=settings.out, runs=settings.runs).render().splitlines()[-1]
    assert footer == (
        "1 attempts, 1 transcripts, $0.0050 spent; 1 crashed, $0.0020 logged, "
        "$0.0500 counted against the budget"
    )


def test_an_attempt_killed_before_its_parquet_episode_counts_its_exact_spend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def killed(*_: Any) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(loop, "write_attempt", killed)
    settings = grid(tmp_path, attempts=1, cells=ONE_CELL)
    with pytest.raises(KeyboardInterrupt):
        asyncio.run(run_grid(settings, Presser(["RIGHT"])))
    monkeypatch.undo()
    first = "E7-claude-haiku-5-5-L1-fake-00"
    (summary,) = attempts(settings).values()
    status = pq.read_table(settings.runs / f"run_id={first}" / "runs").to_pylist()[-1]["status"]
    assert (summary["dollars"], status) == (pytest.approx(0.005), "crashed")
    # Counted at its $0.005, not the $0.05 cap, and run again under the next index.
    model = Presser(["RIGHT"])
    spent = asyncio.run(run_grid(settings, model))
    assert (model.requests, spent) == (5, pytest.approx(0.01))
    assert [row["run_id"] for row in results(settings.runs, "E7")] == [
        "E7-claude-haiku-5-5-L1-fake-01"
    ]


def test_an_attempt_from_another_start_fails_before_its_first_request(tmp_path: Path) -> None:
    settings = grid(tmp_path, attempts=1, cells=ONE_CELL)
    asyncio.run(run_grid(settings, Presser(["RIGHT"])))
    path = settings.out / "E7" / "E7-claude-haiku-5-5-L1-fake-00" / "attempt.json"
    path.write_text(json.dumps(json.loads(path.read_text()) | {"start_sha256": "0" * 64}))
    model = Presser(["RIGHT"])
    with pytest.raises(ExceptionGroup) as raised:
        asyncio.run(run_grid(grid(tmp_path, attempts=2, cells=ONE_CELL), model))
    assert raised.group_contains(ValueError, match="not 0000")
    assert model.requests == 0


def test_the_grid_stops_before_an_attempt_that_could_pass_the_budget(tmp_path: Path) -> None:
    # Each attempt is capped at $0.05: three fit in $0.16, a fourth could pass it.
    settings = grid(tmp_path, budget=0.16, max_actions=300)
    model = Presser(["RIGHT"])
    spent = asyncio.run(run_grid(settings, model))
    assert len(attempts(settings)) == 3
    assert spent <= settings.budget
    assert spent == pytest.approx(sum(row["dollars"] for row in attempts(settings).values()))


def test_a_model_with_no_prices_is_refused_before_any_request(tmp_path: Path) -> None:
    model = Presser(["RIGHT"])
    settings = grid(tmp_path, cells=(("claude-haiku-5-5", 2), ("claude-new-model", 2)))
    with pytest.raises(ValueError, match="claude-new-model"):
        asyncio.run(run_grid(settings, model))
    assert model.requests == 0


def row(success: bool, dollars: float, goal_step: int | None = None) -> dict[str, Any]:
    return {
        "llm_model": "claude-haiku-5-5",
        "llm_tool_level": 1,
        "start_state": "seg_01_house",
        "goal_reached": success,
        "goal_step": goal_step,
        "llm_tokens_in": 1_000,
        "llm_tokens_out": 100,
        "llm_cost_usd": dollars,
    }


def test_no_success_in_10_is_0_to_0_31_and_dollars_per_success_none() -> None:
    (cell,) = cell_rows([row(False, 0.5)] * 10)
    assert (cell["attempts"], cell["successes"]) == (10, 0)
    assert cell["interval"] == pytest.approx((0.0, 0.3085), abs=1e-4)
    assert cell["per_success"] is None
    assert cell["presses_to_goal"] is None
    line = table([cell]).splitlines()[-1]
    assert "| 0.00 [0.00, 0.31] | none |" in line
    assert line.endswith("| $5.00 | none |")


def test_dollars_per_success_counts_the_failed_attempts() -> None:
    rows = [row(True, 1.0, goal_step=40), row(True, 1.0, goal_step=60)] + [row(False, 2.0)] * 8
    (cell,) = cell_rows(rows)
    assert cell["dollars"] == pytest.approx(18.0)
    assert cell["per_success"] == pytest.approx(9.0)
    assert cell["presses_to_goal"] == 50
    low, high = cell["interval"]
    assert (round(low, 4), round(high, 4)) == (0.0252, 0.5561)


def test_the_summary_reads_the_grid_from_the_parquet_logs(tmp_path: Path) -> None:
    settings = grid(tmp_path)
    asyncio.run(run_grid(settings, Presser(["RIGHT", "RIGHT", "A"])))
    text = Summary(out=settings.out, runs=settings.runs).render()
    lines = text.splitlines()
    assert lines[0].startswith("| Model | Level | Segment |")
    assert "| claude-haiku-5-5 | 0 | fake | 2 | 2 | 1.00 [0.16, 1.00] | 3 |" in text
    assert lines[-1] == (
        "4 attempts, 4 transcripts, $0.0040 spent; 0 crashed, $0.0000 logged, "
        "$0.0000 counted against the budget"
    )
