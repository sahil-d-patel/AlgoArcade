import asyncio
import base64
import copy
import itertools
import json
import sys
from collections.abc import Callable, Iterable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from anthropic.types import Message, TextBlock, ToolUseBlock, Usage
from mcp import Client, StdioServerParameters
from mcp.server.mcpserver import MCPServer

from algoarcade.agent.loop import (
    SEGMENTS,
    Attempt,
    AttemptConfig,
    Prices,
    Run,
    Step,
    json_bytes,
    project,
    report,
    run,
    run_attempt,
)
from algoarcade.agent.server import Game, build, png
from algoarcade.backends.base import GameBackend
from algoarcade.backends.fake import FakeBackend
from algoarcade.games.base import StateNotReady
from algoarcade.games.fake import FakeState

PARAMS = StdioServerParameters(command=sys.executable, args=["-m", "algoarcade.agent.server"])
USAGE = Usage(
    input_tokens=10_000,
    output_tokens=0,
    cache_creation_input_tokens=0,
    cache_read_input_tokens=0,
)


def press(buttons: object, usage: Usage = USAGE, **arguments: Any) -> Message:
    block = ToolUseBlock(
        type="tool_use", id="toolu_1", name="press", input={"buttons": buttons, **arguments}
    )
    return reply([block], "tool_use", usage)


def reply(content: list[Any], stop_reason: str, usage: Usage = USAGE) -> Message:
    return Message(
        id="msg_1",
        type="message",
        role="assistant",
        model="stub",
        content=content,
        stop_reason=stop_reason,  # pyrefly: ignore[bad-argument-type]
        stop_sequence=None,
        usage=usage,
    )


END = reply([TextBlock(type="text", text="I give up.")], "end_turn")


class StubModel:
    def __init__(self, replies: Iterable[Message]) -> None:
        self.replies = iter(replies)
        self.requests: list[list[Any]] = []

    async def __call__(self, **request: Any) -> Message:
        self.requests.append(copy.deepcopy(request["messages"]))
        return next(self.replies)


def attempt(
    model: StubModel,
    record: Callable[[Step], None] = lambda _: None,
    server: StdioServerParameters | MCPServer[Game] = PARAMS,
    **settings: Any,
) -> Attempt:
    """`settings` are AttemptConfig fields; the caps default to 100 actions and $100."""
    caps: dict[str, Any] = {"max_actions": 100, "max_dollars": 100.0}
    config = AttemptConfig(**(caps | settings))

    async def go() -> Attempt:
        async with Client(server, raise_exceptions=True) as client:
            return await run_attempt(model, client, config, SEGMENTS["fake"], record)

    return asyncio.run(go())


def test_stops_at_the_action_cap_and_presses_nothing_past_it() -> None:
    model = StubModel(itertools.repeat(press(["RIGHT", "RIGHT"])))
    result = attempt(model, max_actions=3)
    assert result.outcome == "action_cap"
    assert result.actions == 3
    assert [step.buttons for step in result.steps] == [["RIGHT", "RIGHT"], ["RIGHT"]]
    assert result.steps[-1].state is not None
    assert result.steps[-1].state["x"] == 4


@pytest.mark.parametrize("rejected", [["X"], "RIGHT"], ids=["unknown button", "string"])
def test_a_rejected_press_is_not_counted_or_logged(rejected: object) -> None:
    model = StubModel([press(rejected), press(["RIGHT"]), END])
    logged: list[Step] = []
    result = attempt(model, max_actions=3, record=logged.append)
    assert result.outcome == "end_turn"
    assert result.actions == 1
    assert [step.buttons for step in logged] == [[], ["RIGHT"], []]
    assert model.requests[1][-1]["content"][0]["is_error"]
    # One step right from x=1: the rejected call moved nothing.
    assert logged[1].state is not None
    assert logged[1].state["x"] == 2


class NeverReady:
    def read(self, backend: GameBackend) -> FakeState:
        raise StateNotReady("not yet")


def test_presses_into_a_state_that_is_not_ready_still_count() -> None:
    model = StubModel([press(["RIGHT", "DOWN"]), END])
    logged: list[Step] = []
    server = build(lambda: Game(FakeBackend(), NeverReady()))
    result = attempt(model, record=logged.append, server=server)
    assert result.actions == 2
    assert [step.buttons for step in logged] == [["RIGHT", "DOWN"], []]
    assert logged[0].state is None


def test_stops_before_a_request_that_could_pass_the_dollar_cap() -> None:
    model = StubModel(itertools.repeat(press(["DOWN"])))
    # Each request costs $0.001 for 10K input tokens. The next could cost about $0.0014 for its
    # prompt at the cache-write price plus $0.002 for a full reply: $0.002 + $0.0034 passes $0.005.
    result = attempt(model, max_dollars=0.005)
    assert result.outcome == "dollar_cap"
    assert result.dollars <= 0.005
    assert len(model.requests) == 2
    assert result.actions == 2
    assert result.dollars == pytest.approx(0.002)


def test_stops_before_a_request_whose_bound_crosses_100k_prompt_tokens() -> None:
    usage = Usage(
        input_tokens=99_990,
        output_tokens=0,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )
    # The first request costs $0.009999. The second prompt adds a turn, so its bound is past
    # 100K tokens: about $0.073 on the long card, against $0.015 on the short one.
    long_card = attempt(StubModel(itertools.repeat(press(["DOWN"], usage))), max_dollars=0.05)
    assert long_card.outcome == "dollar_cap"
    assert len(long_card.steps) == 1
    assert long_card.dollars <= 0.05
    no_long_card = Prices(long_input=0.10, long_output=0.50, long_cache_write=0.125)
    short_only = attempt(
        StubModel(itertools.repeat(press(["DOWN"], usage))), max_dollars=0.05, prices=no_long_card
    )
    assert len(short_only.steps) > 1


def test_a_request_above_100k_prompt_tokens_is_priced_on_the_long_card() -> None:
    def usage(cache_read: int) -> Usage:
        return Usage(
            input_tokens=1,
            output_tokens=10,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=cache_read,
        )

    model = StubModel([press(["DOWN"], usage(99_999)), press(["DOWN"], usage(100_000)), END])
    result = attempt(model)
    # 100,000 prompt tokens is the short card's last size; 100,001 is the long card's first.
    assert result.steps[0].dollars == pytest.approx((1 * 0.10 + 10 * 0.50 + 99_999 * 0.01) / 1e6)
    assert result.steps[1].dollars == pytest.approx((1 * 0.50 + 10 * 2.50 + 100_000 * 0.05) / 1e6)


def test_sends_nothing_when_the_first_request_could_pass_the_dollar_cap() -> None:
    model = StubModel(itertools.repeat(press(["DOWN"])))
    result = attempt(model, max_dollars=0.002)  # a full 4,096-token reply alone is $0.002048
    assert result.outcome == "dollar_cap"
    assert model.requests == []
    assert result.dollars == 0


def test_stops_at_the_goal() -> None:
    model = StubModel([press(["RIGHT", "RIGHT", "A"]), press(["LEFT"])])
    result = attempt(model)
    assert result.outcome == "goal"
    assert result.actions == 3
    assert len(result.steps) == 1


def test_stops_when_the_model_stops_pressing() -> None:
    model = StubModel([END])
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
        input_tokens=1_000,
        output_tokens=2_000,
        cache_creation_input_tokens=4_000,
        cache_read_input_tokens=8_000,
    )
    prices = Prices(input=1, output=2, cache_write=3, cache_read=4)
    assert prices.dollars(usage) == pytest.approx((1_000 + 4_000 + 12_000 + 32_000) / 1e6)


def test_the_bound_prices_the_prompt_at_the_dearer_of_input_and_cache_write() -> None:
    prices = Prices(
        input=1, output=2, cache_write=3, cache_read=0.5, long_input=5, long_cache_write=4
    )
    assert prices.most(prompt_tokens=100_000, max_tokens=1_000_000) == pytest.approx(0.3 + 2)
    assert prices.most(prompt_tokens=100_001, max_tokens=0) == pytest.approx(100_001 * 5 / 1e6)


def test_a_blank_screenshot_counts_for_more_than_its_image_tokens() -> None:
    # A blank screen compresses best, so its base64 PNG is the shortest a 480x320 one can be.
    blank = base64.b64encode(png(np.zeros((320, 480, 3), np.uint8))).decode()
    assert json_bytes(blank) > 480 * 320 / 750


def test_a_run_logs_every_request_and_reports_per_action(tmp_path: Path) -> None:
    usage = Usage(
        input_tokens=300,
        output_tokens=60,
        cache_creation_input_tokens=900,
        cache_read_input_tokens=1500,
    )
    model = StubModel([press(["RIGHT"], usage), press(["RIGHT", "A"], usage)])
    settings = Run(name="pilot", out=tmp_path, max_actions=20, max_dollars=1.0)
    result = asyncio.run(run(settings, model))

    lines = (tmp_path / "pilot" / "steps.jsonl").read_text().splitlines()
    assert [json.loads(line)["buttons"] for line in lines] == [["RIGHT"], ["RIGHT", "A"]]
    summary = json.loads((tmp_path / "pilot" / "attempt.json").read_text())
    assert (summary["segment"], summary["outcome"], summary["actions"]) == ("fake", "goal", 3)
    # 2 requests of $0.0001875 each over 3 presses.
    assert summary["per_action"] == pytest.approx(
        {
            "input_tokens": 200,
            "output_tokens": 40,
            "cache_write_tokens": 600,
            "cache_read_tokens": 1000,
            "dollars": 0.000125,
        }
    )
    # The prompt is 2,700 tokens on both requests, so all 300 cost $0.0001875.
    assert summary["projection_300_requests"] == pytest.approx(
        {"start": 2_700, "growth": 0, "dollars": 300 * 0.0001875}
    )
    assert report(result, settings.prices).splitlines() == [
        "goal: 3 actions in 2 requests, $0.0004",
        "per action: input_tokens 200.0, output_tokens 40.0, cache_write_tokens 600.0, "
        "cache_read_tokens 1000.0, $0.000125",
        "projected 300 requests: $0.06, a prompt of 2700 tokens growing 0 a request, "
        "each priced on its card",
    ]


def test_seg_01_is_graded_by_the_manifest_goal() -> None:
    pokemon = {"level": 5, "hp": 20, "max_hp": 20, "status": 0}
    bedroom = {"map_group": 4, "map_num": 1, "x": 6, "y": 6, "party": [pokemon], "badge_flags": 0}
    pallet_town = bedroom | {"map_group": 3, "map_num": 0, "x": 6, "y": 8}
    goal = SEGMENTS["seg_01_house"].goal
    assert (goal(bedroom), goal(pallet_town)) == (False, True)


def growing(prompts: list[int]) -> list[Message]:
    """One press per request. The first request is uncached; each later one sends 10 tokens
    uncached, writes the rest of the new turn to the cache and reads everything before it."""
    replies = [press(["DOWN"], Usage(input_tokens=prompts[0], output_tokens=5))]
    for before, now in itertools.pairwise(prompts):
        usage = Usage(
            input_tokens=10,
            output_tokens=5,
            cache_creation_input_tokens=now - before - 10,
            cache_read_input_tokens=before,
        )
        replies.append(press(["DOWN"], usage))
    return replies


SIMPLE = Prices(input=1, output=4, cache_write=2, cache_read=0.5)


def test_the_projection_sums_requests_whose_prompt_grows_by_the_mean() -> None:
    result = attempt(StubModel(growing([1_000, 1_050, 1_200])), max_actions=3, prices=SIMPLE)
    projection = project(result.steps, SIMPLE, requests=4)
    assert projection is not None
    assert (projection.start, projection.growth) == (1_000, 100)
    # The first request as measured, then prompts of 1,100, 1,200 and 1,300 tokens: 10 uncached,
    # 90 written, the rest read, and 5 output tokens each.
    later = sum(10 * 1 + 90 * 2 + (size - 100) * 0.5 + 5 * 4 for size in (1_100, 1_200, 1_300))
    assert projection.dollars == pytest.approx((1_000 * 1 + 5 * 4 + later) / 1e6)


def test_the_projection_moves_to_the_long_card_past_100k_tokens() -> None:
    prices = replace(SIMPLE, long_input=10, long_output=4, long_cache_write=2, long_cache_read=5)
    result = attempt(StubModel(growing([99_800, 99_900])), max_actions=2, prices=prices)
    projection = project(result.steps, prices, requests=4)
    assert projection is not None
    # Prompts of 99,900 and 100,000 tokens stay on the short card; 100,100 is on the long one,
    # where only the input and cache-read prices differ here.
    short = sum(10 * 1 + 90 * 2 + (size - 100) * 0.5 + 5 * 4 for size in (99_900, 100_000))
    long = 10 * 10 + 90 * 2 + 100_000 * 5 + 5 * 4
    assert projection.dollars == pytest.approx((99_800 * 1 + 5 * 4 + short + long) / 1e6)


def test_the_projection_sends_everything_uncached_when_the_cache_was_never_read() -> None:
    replies = [press(["DOWN"], Usage(input_tokens=n, output_tokens=5)) for n in (1_000, 1_100)]
    result = attempt(StubModel(replies), max_actions=2, prices=SIMPLE)
    projection = project(result.steps, SIMPLE, requests=3)
    assert projection is not None
    assert projection.dollars == pytest.approx((1_000 + 1_100 + 1_200 + 3 * 5 * 4) / 1e6)


def test_no_projection_from_a_single_request() -> None:
    result = attempt(StubModel([press(["DOWN"]), END]))
    assert project(result.steps[:1], Prices()) is None
