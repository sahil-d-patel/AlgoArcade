import asyncio
import copy
import itertools
import sys
from collections.abc import Iterable
from typing import Any

import pytest
from anthropic.types import Message, TextBlock, ToolUseBlock, Usage
from mcp import Client, StdioServerParameters

from algoarcade.agent.loop import SEGMENTS, Attempt, AttemptConfig, Prices, run_attempt

PARAMS = StdioServerParameters(command=sys.executable, args=["-m", "algoarcade.agent.server"])
USAGE = Usage(
    input_tokens=1_000_000,
    output_tokens=0,
    cache_creation_input_tokens=0,
    cache_read_input_tokens=0,
)


def press(buttons: list[str], **arguments: Any) -> Message:
    block = ToolUseBlock(
        type="tool_use", id="toolu_1", name="press", input={"buttons": buttons, **arguments}
    )
    return reply([block], "tool_use")


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


class StubModel:
    def __init__(self, replies: Iterable[Message]) -> None:
        self.replies = iter(replies)
        self.requests: list[list[Any]] = []

    async def __call__(self, **request: Any) -> Message:
        self.requests.append(copy.deepcopy(request["messages"]))
        return next(self.replies)


def attempt(model: StubModel, max_actions: int = 100, max_dollars: float = 100.0) -> Attempt:
    config = AttemptConfig(max_actions=max_actions, max_dollars=max_dollars)

    async def go() -> Attempt:
        async with Client(PARAMS, raise_exceptions=True) as client:
            return await run_attempt(model, client, config, SEGMENTS["fake"])

    return asyncio.run(go())


def test_stops_at_the_action_cap_and_presses_nothing_past_it() -> None:
    model = StubModel(itertools.repeat(press(["RIGHT", "RIGHT"])))
    result = attempt(model, max_actions=3)
    assert result.outcome == "action_cap"
    assert result.actions == 3
    assert [step.buttons for step in result.steps] == [["RIGHT", "RIGHT"], ["RIGHT"]]
    assert result.steps[-1].state is not None
    assert result.steps[-1].state["x"] == 4


def test_stops_at_the_dollar_cap() -> None:
    model = StubModel(itertools.repeat(press(["DOWN"])))
    result = attempt(model, max_dollars=0.25)  # $0.10 per request at the default prices
    assert result.outcome == "dollar_cap"
    assert result.actions == 3
    assert result.dollars == pytest.approx(0.30)


def test_stops_at_the_goal() -> None:
    model = StubModel([press(["RIGHT", "RIGHT", "A"]), press(["LEFT"])])
    result = attempt(model)
    assert result.outcome == "goal"
    assert result.actions == 3
    assert len(result.steps) == 1


def test_stops_when_the_model_stops_pressing() -> None:
    model = StubModel([reply([TextBlock(type="text", text="I give up.")], "end_turn")])
    result = attempt(model)
    assert result.outcome == "end_turn"
    assert result.actions == 0
    assert result.steps[0].text == "I give up."


def test_never_changes_an_earlier_message() -> None:
    model = StubModel(itertools.repeat(press(["DOWN"], screenshot=True)))
    attempt(model, max_actions=4)
    assert len(model.requests) == 4
    assert model.requests[0] == [{"role": "user", "content": SEGMENTS["fake"].task}]
    for earlier, later in itertools.pairwise(model.requests):
        assert later[: len(earlier)] == earlier
        assert len(later) == len(earlier) + 2
    tool_result = model.requests[1][2]["content"][0]
    assert [block["type"] for block in tool_result["content"]] == ["text", "image"]


def test_dollars_follow_the_four_prices() -> None:
    usage = Usage(
        input_tokens=1_000_000,
        output_tokens=2_000_000,
        cache_creation_input_tokens=4_000_000,
        cache_read_input_tokens=8_000_000,
    )
    assert Prices(input=1, output=2, cache_write=3, cache_read=4).dollars(usage) == 1 + 4 + 12 + 32
