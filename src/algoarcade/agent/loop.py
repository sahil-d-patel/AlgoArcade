"""Drives a model through the MCP server for one attempt. Contract: docs/technical-plan.md, LLM arm.

The history is append-only and the tool list is fetched once, so every request reuses the cached
prefix. Each request's usage is priced and logged, and the attempt stops at the goal, at the
action cap, before a request that could pass the dollar cap, or when the model stops pressing
buttons.
"""

import asyncio
import json
import os
import sys
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import tyro
from anthropic import AsyncAnthropic
from anthropic.types import Message, MessageParam, ToolParam, ToolResultBlockParam, Usage
from anthropic.types.tool_result_block_param import Content as ToolResultContent
from mcp import Client, StdioServerParameters
from mcp.client.stdio import get_default_environment
from mcp_types import CallToolResult
from pydantic_core import to_jsonable_python

Create = Callable[..., Awaitable[Message]]
Goal = Callable[[dict[str, Any]], bool]

SYSTEM = (
    "You are playing a game through two tools. get_state reports where you are and can attach "
    "a screenshot. press presses buttons one decision each and reports the state after. Every "
    "button press counts against a fixed budget, so ask for the state, decide, and press. Keep "
    "going until you are told to stop; do not stop pressing to ask questions."
)
# The API wraps the tool definitions in a system prompt of its own. Its size for this model is
# not published; 1,000 tokens is a guess above it.
TOOL_PROMPT_TOKENS = 1_000


@dataclass(frozen=True)
class Segment:
    """What the model is told to do, the server that plays it, and how the attempt is graded."""

    task: str
    goal: Goal
    server: tuple[str, ...]


SEGMENTS = {
    "fake": Segment(
        task="Open the chest. The chest is on the top row of the first room.",
        goal=lambda state: bool(state["chest_opened"]),
        server=("--game", "fake"),
    ),
    "seg_01_house": Segment(
        task="You start in your bedroom, upstairs in your house in Pallet Town. Leave the house.",
        # ENV-06's left_house: on MAP_PALLET_TOWN, group 3 number 0.
        # TODO(#17): grade with the manifest's predicate once it merges.
        goal=lambda state: (state["map_group"], state["map_num"]) == (3, 0),
        server=("--game", "firered", "--state", "seg_01_house"),
    ),
}


@dataclass(frozen=True)
class Prices:
    """Dollars per million tokens. Defaults: claude-haiku-5-5 for prompts up to 100K tokens,
    platform.claude.com/docs/en/about-claude/pricing on 2026-11-01."""

    input: float = 0.10
    output: float = 0.50
    cache_write: float = 0.125
    cache_read: float = 0.01

    def dollars(self, usage: Usage) -> float:
        return (
            usage.input_tokens * self.input
            + usage.output_tokens * self.output
            + (usage.cache_creation_input_tokens or 0) * self.cache_write
            + (usage.cache_read_input_tokens or 0) * self.cache_read
        ) / 1_000_000

    def most(self, prompt_tokens: int, max_tokens: int) -> float:
        """The most a request can cost: each prompt token at the dearer of input and cache
        write, and a reply of `max_tokens`."""
        prompt_price = max(self.input, self.cache_write)
        return (prompt_tokens * prompt_price + max_tokens * self.output) / 1_000_000


@dataclass(frozen=True, kw_only=True)
class AttemptConfig:
    max_actions: int
    max_dollars: float
    model: str = "claude-haiku-5-5"
    prices: Prices = field(default_factory=Prices)
    max_tokens: int = 4096
    system: str = SYSTEM


@dataclass(frozen=True)
class Step:
    """One request to the model and the presses it made."""

    index: int
    input_tokens: int
    output_tokens: int
    cache_write_tokens: int
    cache_read_tokens: int
    dollars: float
    buttons: list[str]
    text: str
    stop_reason: str | None
    state: dict[str, Any] | None


@dataclass(frozen=True)
class Attempt:
    outcome: str  # goal, action_cap, dollar_cap, or the model's stop_reason
    actions: int
    dollars: float
    steps: list[Step]

    def per_action(self) -> dict[str, float] | None:
        """Mean tokens of each kind and dollars per button press; None if nothing was pressed."""
        if not self.actions:
            return None
        totals = {
            kind: sum(getattr(step, kind) for step in self.steps)
            for kind in ("input_tokens", "output_tokens", "cache_write_tokens", "cache_read_tokens")
        }
        return {kind: total / self.actions for kind, total in totals.items()} | {
            "dollars": self.dollars / self.actions
        }


def json_bytes(value: object) -> int:
    """At least the token count of `value`. A token is at least one byte of text, and a base64
    PNG is longer than the image's token count (about width x height / 750)."""
    return len(json.dumps(to_jsonable_python(value)).encode())


def tool_result_content(result: CallToolResult) -> list[ToolResultContent]:
    blocks: list[ToolResultContent] = []
    for block in result.content:
        if block.type == "text":
            blocks.append({"type": "text", "text": block.text})
        elif block.type == "image" and block.mime_type == "image/png":
            blocks.append(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": block.data},
                }
            )
    return blocks


@dataclass(frozen=True)
class ToolRound:
    results: list[ToolResultBlockParam]
    buttons: list[str]
    state: dict[str, Any] | None


async def call_tools(client: Client, response: Message, budget: int) -> ToolRound:
    """Run the response's tool calls, pressing no more buttons than `budget` allows."""
    results: list[ToolResultBlockParam] = []
    buttons: list[str] = []
    state = None
    for block in response.content:
        if block.type != "tool_use":
            continue
        arguments = dict(block.input)
        requested = arguments.get("buttons")
        allowed: list[str] = []
        # Anything but a list goes through untouched so the server rejects it.
        if block.name == "press" and isinstance(requested, list):
            allowed = arguments["buttons"] = requested[: budget - len(buttons)]
        result = await client.call_tool(block.name, arguments)
        if not result.is_error:
            buttons += allowed
        results.append(
            {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": tool_result_content(result),
                "is_error": result.is_error,
            }
        )
        if result.structured_content is not None:
            state = result.structured_content
    return ToolRound(results, buttons, state)


async def run_attempt(
    create: Create,
    client: Client,
    config: AttemptConfig,
    segment: Segment,
    record: Callable[[Step], None] = lambda _: None,
) -> Attempt:
    listed = sorted((await client.list_tools()).tools, key=lambda tool: tool.name)
    tools: list[ToolParam] = [
        {
            "name": tool.name,
            "description": tool.description or "",
            "input_schema": tool.input_schema,
        }
        for tool in listed
    ]
    messages: list[MessageParam] = [{"role": "user", "content": segment.task}]
    steps: list[Step] = []
    actions = 0
    dollars = 0.0
    prompt_tokens = TOOL_PROMPT_TOKENS + json_bytes([config.system, tools])
    sent = 0
    while True:
        # The prompt only grows by what was appended since the last request, whose size the API
        # reported, so this bounds the next request before it is paid for.
        prompt_tokens += json_bytes(messages[sent:])
        if dollars + config.prices.most(prompt_tokens, config.max_tokens) > config.max_dollars:
            return Attempt(outcome="dollar_cap", actions=actions, dollars=dollars, steps=steps)
        sent = len(messages)
        response = await create(
            model=config.model,
            max_tokens=config.max_tokens,
            system=[
                {"type": "text", "text": config.system, "cache_control": {"type": "ephemeral"}}
            ],
            tools=tools,
            cache_control={"type": "ephemeral"},
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})
        tool_round = await call_tools(client, response, config.max_actions - actions)
        if tool_round.results:
            messages.append({"role": "user", "content": tool_round.results})
        actions += len(tool_round.buttons)
        usage = response.usage
        prompt_tokens = (
            usage.input_tokens
            + (usage.cache_creation_input_tokens or 0)
            + (usage.cache_read_input_tokens or 0)
        )
        step = Step(
            index=len(steps),
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_write_tokens=usage.cache_creation_input_tokens or 0,
            cache_read_tokens=usage.cache_read_input_tokens or 0,
            dollars=config.prices.dollars(usage),
            buttons=tool_round.buttons,
            text="".join(block.text for block in response.content if block.type == "text"),
            stop_reason=response.stop_reason,
            state=tool_round.state,
        )
        dollars += step.dollars
        steps.append(step)
        record(step)

        if tool_round.state is not None and segment.goal(tool_round.state):
            outcome = "goal"
        elif actions >= config.max_actions:
            outcome = "action_cap"
        elif response.stop_reason != "tool_use":
            outcome = response.stop_reason or "end_turn"
        else:
            continue
        return Attempt(outcome=outcome, actions=actions, dollars=dollars, steps=steps)


def server(args: tuple[str, ...]) -> StdioServerParameters:
    """The MCP client passes the server only a few environment variables. On macOS the mGBA
    binding also needs DYLD_FALLBACK_LIBRARY_PATH (docs/setup.md)."""
    env = get_default_environment()
    if "DYLD_FALLBACK_LIBRARY_PATH" in os.environ:
        env["DYLD_FALLBACK_LIBRARY_PATH"] = os.environ["DYLD_FALLBACK_LIBRARY_PATH"]
    return StdioServerParameters(
        command=sys.executable, args=["-m", "algoarcade.agent.server", *args], env=env
    )


@dataclass(frozen=True, kw_only=True)
class Run(AttemptConfig):
    """One attempt, logged under data/llm/<name>/ as steps.jsonl and attempt.json."""

    name: str
    segment: str = "fake"
    out: Path = Path("data/llm")


async def run(settings: Run, create: Create) -> Attempt:
    segment = SEGMENTS[settings.segment]
    folder = settings.out / settings.name
    folder.mkdir(parents=True)
    with (folder / "steps.jsonl").open("w") as log:

        def record(step: Step) -> None:
            log.write(json.dumps(asdict(step)) + "\n")
            log.flush()

        async with Client(server(segment.server), raise_exceptions=True) as client:
            attempt = await run_attempt(create, client, settings, segment, record)
    summary = asdict(settings) | asdict(attempt) | {"per_action": attempt.per_action()}
    del summary["steps"]
    (folder / "attempt.json").write_text(json.dumps(summary, indent=2, default=str))
    return attempt


def report(attempt: Attempt) -> str:
    lines = [
        f"{attempt.outcome}: {attempt.actions} actions in {len(attempt.steps)} requests, "
        f"${attempt.dollars:.4f}"
    ]
    per_action = attempt.per_action()
    if per_action is not None:
        dollars = per_action.pop("dollars")
        tokens = ", ".join(f"{kind} {value:.1f}" for kind, value in per_action.items())
        lines.append(f"per action: {tokens}, ${dollars:.6f}")
        lines.append(f"300 actions at this rate: ${300 * dollars:.2f}")
    return "\n".join(lines)


def main() -> None:
    attempt = asyncio.run(run(tyro.cli(Run), AsyncAnthropic().messages.create))
    print(report(attempt))  # noqa: T201


if __name__ == "__main__":
    main()
