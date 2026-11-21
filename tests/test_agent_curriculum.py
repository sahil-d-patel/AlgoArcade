"""Save states along an LLM attempt, kept for E8's curriculum only when it reaches its goal."""

import asyncio
import dataclasses
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from anthropic.types import Message, TextBlock, ToolUseBlock, Usage

from algoarcade.agent.curriculum import keep
from algoarcade.agent.loop import Run, run
from algoarcade.agent.tools import Game
from algoarcade.backends.base import Button, GameBackend
from algoarcade.backends.fake import FakeBackend
from algoarcade.games.base import StateNotReady
from algoarcade.games.fake import FakeField, FakeReader, FakeState

USAGE = Usage(input_tokens=100, output_tokens=0)


def reply(content: list[Any], stop_reason: str) -> Message:
    return Message(
        id="msg_1",
        type="message",
        role="assistant",
        model="stub",
        content=content,
        stop_reason=stop_reason,  # pyrefly: ignore[bad-argument-type]
        stop_sequence=None,
        usage=USAGE,
    )


def press(buttons: list[str]) -> Message:
    block = ToolUseBlock(type="tool_use", id="toolu_1", name="press", input={"buttons": buttons})
    return reply([block], "tool_use")


END = reply([TextBlock(type="text", text="I give up.")], "end_turn")


class Scripted:
    def __init__(self, replies: Iterable[Message]) -> None:
        self.replies = iter(replies)

    async def __call__(self, **_: Any) -> Message:
        return next(self.replies)


def attempt(tmp_path: Path, replies: list[Message]) -> Run:
    settings = Run(
        name="walk",
        out=tmp_path / "llm",
        runs=tmp_path / "runs",
        curriculum=tmp_path / "curriculum",
        capture_every=2,
        max_actions=20,
        max_dollars=1.0,
    )
    asyncio.run(run(settings, Scripted(replies)))
    return settings


def test_a_successful_attempt_keeps_the_states_before_the_goal_with_a_manifest(
    tmp_path: Path,
) -> None:
    # The chest is at (3, 1); A on it at press 5 is the goal. Saves land on presses 2 and 4.
    attempt(tmp_path, [press(["DOWN", "RIGHT", "UP", "RIGHT", "A"])])
    folder = tmp_path / "curriculum" / "walk"
    manifest = json.loads((folder / "manifest.json").read_text())
    assert (manifest["start_state"], manifest["goal_step"], manifest["capture_every"]) == (
        "fake",
        5,
        2,
    )
    states = manifest["states"]
    assert [(s["action"], s["location"]) for s in states] == [(2, [0, 2, 2]), (4, [0, 3, 1])]
    assert [sorted(map(tuple, s["visited"])) for s in states] == [
        [(0, 1, 1), (0, 1, 2), (0, 2, 2)],
        [(0, 1, 1), (0, 1, 2), (0, 2, 1), (0, 2, 2), (0, 3, 1)],
    ]
    # The fake's counts never change: no flags, level sum 5, no badges, no milestones.
    assert states[1]["bests"] == {"events": 0, "level": 5, "badge": 0, "milestone": 0}
    assert states[1]["milestone_count"] == 0
    for entry in states:
        backend = FakeBackend()
        backend.load_state((folder / entry["file"]).read_bytes())
        assert list(FakeReader().read(backend).location) == entry["location"]
    assert not (tmp_path / "llm" / "walk" / "states").exists()


def test_a_failed_attempt_keeps_no_states(tmp_path: Path) -> None:
    attempt(tmp_path, [press(["DOWN", "RIGHT", "UP", "RIGHT"]), END])
    assert not (tmp_path / "curriculum").exists()
    assert not (tmp_path / "llm" / "walk" / "states").exists()


class ReadyOnOddFrames:
    """Not ready on every other press: the fake advances 24 frames a press."""

    def read(self, backend: GameBackend) -> FakeState:
        state = FakeReader().read(backend)
        if state.frame % 48:
            raise StateNotReady("odd press")
        return state


def test_a_save_due_on_a_press_that_is_not_ready_moves_to_the_next_ready_one(
    tmp_path: Path,
) -> None:
    game = Game(FakeBackend(), ReadyOnOddFrames(), FakeField(), capture=tmp_path, capture_every=3)
    with (tmp_path / "presses.jsonl").open("w") as game.trace:
        game.record(None)
        for _ in range(8):
            game.press(Button.DOWN)
    lines = [json.loads(line) for line in (tmp_path / "presses.jsonl").read_text().splitlines()]
    # Presses 1, 3, 5 and 7 are not ready. Press 3 is due and not ready, so 4 saves; 7 is due
    # next and not ready, so 8 saves.
    assert [i for i, line in enumerate(lines) if "saved" in line] == [4, 8]
    assert sorted(p.name for p in tmp_path.glob("*.state")) == ["00004.state", "00008.state"]


def test_a_torn_read_on_a_saved_press_is_not_its_best(tmp_path: Path) -> None:
    start = dataclasses.asdict(FakeReader().read(FakeBackend()))
    torn = start | {"x": 2, "badge_count": 5, "milestone_count": 9}
    presses = [
        {"state": start, "start_sha256": ""},
        {"state": start | {"x": 2}},
        {"state": torn, "saved": "00002.state"},
        {"state": start | {"x": 2}},
    ]
    (tmp_path / "captured").mkdir()
    (tmp_path / "captured" / "00002.state").write_bytes(b"")
    keep(presses, tmp_path / "captured", tmp_path / "out", 3, lambda s: FakeState(**s), "fake", 2)
    (state,) = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))[
        "states"
    ]
    assert state["bests"]["badge"] == 0
    assert state["bests"]["milestone"] == 0
