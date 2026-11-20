"""Play seg_05_forest through the LLM loop with a scripted stand-in for the model, and keep its
curriculum states. Nothing calls the API; every request is answered here and priced at $0.

    uv run python scripts/scripted_seg05.py --name scripted-seg05

At L2 it calls navigate_to through the corners of scripts/capture_states.py's
seg_05_forest_goal path and presses A through every battle and text box. The attempt is logged
as model "scripted" under data/llm/<name>/ and data/runs/, and its states go to
data/states/curriculum/<name>/. They are not an LLM's states: E8 compares against those, so use
these only for smoke runs and tests.
"""

import asyncio
import json
from pathlib import Path
from typing import Any

import tyro
from anthropic.types import Message, ToolUseBlock, Usage

from algoarcade.agent.loop import Attempt, Prices, Run, report, run

# Tile corners of the hand-recorded path, from (29, 62) at the forest's south end.
WAYPOINTS = (
    (29, 59),
    (31, 58),
    (31, 55),
    (39, 55),
    (39, 19),
    (32, 19),
    (32, 23),
    (23, 23),
    (23, 10),
    (15, 10),
    (15, 27),
    (7, 27),
    (7, 23),
    (6, 23),
    (6, 13),
    (5, 13),
    (5, 9),
)
FREE = Prices(0, 0, 0, 0, 0, 0, 0, 0)


def results(messages: list[Any]) -> list[Any]:
    """The text blocks of the last tool result, each parsed as JSON where it is JSON."""
    content = messages[-1]["content"]
    if isinstance(content, str):
        return []
    parsed = []
    for block in content[-1]["content"]:
        if block["type"] == "text":
            try:
                parsed.append(json.loads(block["text"]))
            except json.JSONDecodeError:
                parsed.append(block["text"])
    return parsed


def call(name: str, **arguments: Any) -> Message:
    return Message(
        id="msg_scripted",
        type="message",
        role="assistant",
        model="scripted",
        content=[ToolUseBlock(type="tool_use", id="toolu_scripted", name=name, input=arguments)],
        stop_reason="tool_use",
        stop_sequence=None,
        usage=Usage(input_tokens=0, output_tokens=0),
    )


class Walker:
    """Answers each request from the state in the last tool result."""

    def __init__(self) -> None:
        self.next = 0

    async def __call__(self, messages: list[Any], **_: Any) -> Message:
        parsed = results(messages)
        if not parsed:
            return call("get_state")
        state = parsed[-1]
        blocked = isinstance(parsed[0], dict) and parsed[0].get("reason") == "no_path"
        if blocked or not isinstance(state, dict):
            # A walker in the way, or a read that was not ready. B changes nothing in the
            # forest; it waits one decision, and the cap on presses still ends the attempt.
            return call("press", buttons=["B"])
        if state["in_battle"] or state["text_box_open"]:
            return call("press", buttons=["A"])
        here = (state["x"], state["y"])
        if here in WAYPOINTS:
            self.next = max(self.next, WAYPOINTS.index(here) + 1)
        if self.next == len(WAYPOINTS):
            # The gate's door is north of the last corner.
            return call("press", buttons=["UP"])
        x, y = WAYPOINTS[self.next]
        return call("navigate_to", x=x, y=y)


def play(settings: Run) -> Attempt:
    return asyncio.run(run(settings, Walker()))


def main(
    name: str = "scripted-seg05",
    capture_every: int = 10,
    out: Path = Path("data/llm"),
    runs: Path = Path("data/runs"),
    curriculum: Path = Path("data/states/curriculum"),
) -> None:
    settings = Run(
        name=name,
        segment="seg_05_forest",
        level=2,
        model="scripted",
        prices=FREE,
        max_actions=2_000,
        max_dollars=0.0,
        capture_every=capture_every,
        out=out,
        runs=runs,
        curriculum=curriculum,
    )
    print(report(play(settings), FREE))


if __name__ == "__main__":
    tyro.cli(main)
