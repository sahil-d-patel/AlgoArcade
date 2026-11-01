"""Drives a model through the MCP server for one attempt. Contract: docs/technical-plan.md, LLM arm.

The history is append-only and the tool list is fetched once, so every request reuses the cached
prefix. Each request's usage is priced and logged, and the attempt stops at the goal, at the
action cap, at the dollar cap, or when the model stops pressing buttons.
"""

import asyncio
import json
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
from mcp_types import CallToolResult

Create = Callable[..., Awaitable[Message]]
Goal = Callable[[dict[str, Any]], bool]

SYSTEM = (
    "You are playing a game through two tools. get_state reports where you are and can attach "
    "a screenshot. press presses buttons one decision each and reports the state after. Every "
    "button press counts against a fixed budget, so ask for the state, decide, and press. Keep "
    "going until you are told to stop; do not stop pressing to ask questions."
)


@dataclass(frozen=True)
class Segment:
    """What the model is told to do and how the attempt is graded. ENV-06's manifest adds FireRed."""

    task: str
    goal: Goal


SEGMENTS = {
    "fake": Segment(
        task="Open the chest. The chest is on the top row of the first room.",
        goal=lambda state: bool(state["chest_opened"]),
    )
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
        arguments: dict[str, Any] = dict(block.input) if isinstance(block.input, dict) else {}
        if block.name == "press":
            allowed: list[str] = list(arguments.get("buttons", []))[: budget - len(buttons)]
            arguments["buttons"] = allowed
            buttons += allowed
        result = await client.call_tool(block.name, arguments)
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
    while True:
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
        elif dollars >= config.max_dollars:
            outcome = "dollar_cap"
        elif response.stop_reason != "tool_use":
            outcome = response.stop_reason or "end_turn"
        else:
            continue
        return Attempt(outcome=outcome, actions=actions, dollars=dollars, steps=steps)


@dataclass(frozen=True, kw_only=True)
class Run(AttemptConfig):
    """One attempt, logged under data/llm/<name>/ as steps.jsonl and attempt.json."""

    name: str
    game: str = "fake"
    out: Path = Path("data/llm")


async def run(settings: Run) -> Attempt:
    folder = settings.out / settings.name
    folder.mkdir(parents=True)
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "algoarcade.agent.server", "--game", settings.game]
    )
    api = AsyncAnthropic()
    with (folder / "steps.jsonl").open("w") as log:

        def record(step: Step) -> None:
            log.write(json.dumps(asdict(step)) + "\n")
            log.flush()

        async with Client(params, raise_exceptions=True) as client:
            attempt = await run_attempt(
                api.messages.create, client, settings, SEGMENTS[settings.game], record
            )
    summary = asdict(settings) | asdict(attempt)
    del summary["steps"]
    (folder / "attempt.json").write_text(json.dumps(summary, indent=2, default=str))
    return attempt


def main() -> None:
    attempt = asyncio.run(run(tyro.cli(Run)))
    print(f"{attempt.outcome}: {attempt.actions} actions, ${attempt.dollars:.4f}")  # noqa: T201


if __name__ == "__main__":
    main()
