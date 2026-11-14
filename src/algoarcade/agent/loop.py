"""Drives a model through the MCP server for one attempt. Contract: docs/technical-plan.md, LLM arm.

The history is append-only and the tool list is fetched once, so every request reuses the cached
prefix. Each request's usage is priced and logged, and the attempt stops at the goal, at the
action cap, before a request that could pass the dollar cap, or when the model stops pressing
buttons. Actions and the goal come from the server's trace, which holds every press a tool made
and the state after it, at every tool level.
"""

import asyncio
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TextIO

import tyro
from anthropic import AsyncAnthropic
from anthropic.types import Message, MessageParam, ToolParam, ToolResultBlockParam, Usage
from anthropic.types.tool_result_block_param import Content as ToolResultContent
from mcp import Client, StdioServerParameters
from mcp.client.stdio import get_default_environment
from mcp_types import CallToolResult
from pydantic_core import to_jsonable_python

from algoarcade.agent.parquet import write_attempt, write_run_row
from algoarcade.agent.tools import DIALOGUE_PRESSES, NAVIGATE_PRESSES
from algoarcade.envs.env import HOLD, RELEASE
from algoarcade.games.fake import FakeState
from algoarcade.games.firered.manifest import MANIFEST, ROM_SHA1
from algoarcade.games.firered.reader import FireRedState, PartyPokemon
from algoarcade.logs.writer import run_dir

Create = Callable[..., Awaitable[Message]]
Goal = Callable[[dict[str, Any]], bool]

# The same at every tool level, so a rung differs from the next only in its tools.
SYSTEM = (
    "You are playing a game through the tools you have; each tool's description says what it "
    "does. Every button press counts against a fixed budget, including the presses a tool makes "
    "for you, so look, decide, and act. Keep going until you are told to stop; do not stop to "
    "ask questions."
)
# What the loop caps to the presses left, per tool: press's buttons, the others' max_presses
# with the tool's own default.
PRESS_CAPS = {"navigate_to": NAVIGATE_PRESSES, "advance_dialogue": DIALOGUE_PRESSES}
# The API wraps the tool definitions in a system prompt of its own. Its size for this model is
# not published; 1,000 tokens is a guess above it.
TOOL_PROMPT_TOKENS = 1_000


def firered_state(state: dict[str, Any]) -> FireRedState:
    """The FireRedState the server's state JSON came from."""
    party = tuple(PartyPokemon(**pokemon) for pokemon in state["party"])
    return FireRedState(**state | {"party": party})


def firered_goal(name: str) -> Goal:
    """The manifest's goal for this start state, applied to the server's state JSON."""
    (entry,) = (entry for entry in MANIFEST if entry.name == name)
    return lambda state: entry.goal(firered_state(state))


@dataclass(frozen=True)
class Segment:
    """What the model is told to do, the game and start state the server plays, and how the
    attempt is graded."""

    task: str
    goal: Goal
    game: str
    start_state: str | None = None

    @property
    def server(self) -> tuple[str, ...]:
        state = ("--state", self.start_state) if self.start_state else ()
        return ("--game", self.game, *state)


# The server's --game: the backend and ROM it runs, and its GameState from the state JSON.
GAMES: dict[str, tuple[str, str, Callable[[dict[str, Any]], Any]]] = {
    "fake": ("fake", "", lambda state: FakeState(**state)),
    # The server refuses a manifest state on any ROM but the one the states were captured on.
    "firered": ("mgba", ROM_SHA1, firered_state),
}

SEGMENTS = {
    "fake": Segment(
        task="Open the chest. The chest is on the top row of the first room.",
        goal=lambda state: bool(state["chest_opened"]),
        game="fake",
    ),
    "seg_01_house": Segment(
        task="You start in your bedroom, upstairs in your house in Pallet Town. Leave the house.",
        goal=firered_goal("seg_01_house"),
        game="firered",
        start_state="seg_01_house",
    ),
    "seg_02_pallet": Segment(
        task="You are in Pallet Town with your first Pokemon. Go north into Route 1.",
        goal=firered_goal("seg_02_pallet"),
        game="firered",
        start_state="seg_02_pallet",
    ),
    "seg_03_rival": Segment(
        task=(
            "You are in Professor Oak's lab and have just chosen your first Pokemon. Your rival "
            "will challenge you as you head for the door. Win the battle."
        ),
        goal=firered_goal("seg_03_rival"),
        game="firered",
        start_state="seg_03_rival",
    ),
}


@dataclass(frozen=True)
class Prices:
    """Dollars per million tokens, on the short card up to `long_above` prompt tokens and the long
    card beyond. Defaults: claude-haiku-5-5, Anthropic's model table on 2026-11-03. The table
    lists no long-card cache rates; these assume the short card's multipliers, cache write 1.25x
    input and cache read 0.1x input."""

    input: float = 0.10
    output: float = 0.50
    cache_write: float = 0.125
    cache_read: float = 0.01
    long_input: float = 0.50
    long_output: float = 2.50
    long_cache_write: float = 0.625
    long_cache_read: float = 0.05
    long_above: int = 100_000

    def card(self, prompt_tokens: int) -> tuple[float, float, float, float]:
        """Input, output, cache-write and cache-read prices for a prompt of this size."""
        if prompt_tokens > self.long_above:
            return self.long_input, self.long_output, self.long_cache_write, self.long_cache_read
        return self.input, self.output, self.cache_write, self.cache_read

    def dollars(self, usage: Usage) -> float:
        cache_write = usage.cache_creation_input_tokens or 0
        cache_read = usage.cache_read_input_tokens or 0
        prices = self.card(usage.input_tokens + cache_write + cache_read)
        tokens = (usage.input_tokens, usage.output_tokens, cache_write, cache_read)
        return sum(count * price for count, price in zip(tokens, prices, strict=True)) / 1_000_000

    def most(self, prompt_tokens: int, max_tokens: int) -> float:
        """The most a request with at most `prompt_tokens` can cost: each prompt token at the
        dearer of input and cache write, and a reply of `max_tokens`, all on the card that size
        of prompt would use. The long card is dearer, so a smaller real prompt costs less."""
        input_price, output_price, cache_write_price, _ = self.card(prompt_tokens)
        prompt_price = max(input_price, cache_write_price)
        return (prompt_tokens * prompt_price + max_tokens * output_price) / 1_000_000


@dataclass(frozen=True, kw_only=True)
class AttemptConfig:
    max_actions: int
    max_dollars: float
    # The tool ladder's rung: 0 press and screenshots, 1 adds the state JSON, 2 adds
    # navigate_to and advance_dialogue.
    level: int = 1
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

    @property
    def prompt_tokens(self) -> int:
        return self.input_tokens + self.cache_write_tokens + self.cache_read_tokens


@dataclass(frozen=True)
class Attempt:
    outcome: str  # goal, action_cap, dollar_cap, or the model's stop_reason
    actions: int
    dollars: float
    steps: list[Step]
    goal_step: int | None = None  # the press after which the goal first held

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
    presses: list[dict[str, Any]]  # the trace lines the round's tool calls wrote


def capped(name: str, arguments: dict[str, Any], budget: int) -> dict[str, Any]:
    """The call with no more presses than `budget`. A button list is cut to the budget; a press
    count is always sent as a whole number no larger than it, since the server converts "50" and
    50.0 to 50. Anything else goes through untouched so the server rejects it."""
    budget = max(budget, 0)
    if name == "press" and isinstance(arguments.get("buttons"), list):
        return arguments | {"buttons": arguments["buttons"][:budget]}
    if name in PRESS_CAPS:
        cap = arguments.get("max_presses")
        if not isinstance(cap, int) or isinstance(cap, bool):
            cap = PRESS_CAPS[name]
        return arguments | {"max_presses": min(cap, budget)}
    return arguments


async def call_tools(client: Client, response: Message, budget: int, trace: TextIO) -> ToolRound:
    """Run the response's tool calls, pressing no more buttons than `budget` allows."""
    results: list[ToolResultBlockParam] = []
    presses: list[dict[str, Any]] = []
    for block in response.content:
        if block.type != "tool_use":
            continue
        arguments = capped(block.name, dict(block.input), budget - len(presses))
        result = await client.call_tool(block.name, arguments)
        # The server writes and flushes a press's line before it replies.
        presses += [json.loads(line) for line in trace.readlines()]
        results.append(
            {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": tool_result_content(result),
                "is_error": result.is_error,
            }
        )
    return ToolRound(results, presses)


async def run_attempt(  # noqa: PLR0913 the attempt's model, server, config, task and logs
    create: Create,
    client: Client,
    config: AttemptConfig,
    segment: Segment,
    trace: Path,
    record: Callable[[Step], None] = lambda _: None,
) -> Attempt:
    """`trace` is the file the server behind `client` writes its presses to."""
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
    goal_step = None
    prompt_tokens = TOOL_PROMPT_TOKENS + json_bytes([config.system, tools])
    sent = 0
    with trace.open() as presses:
        presses.readline()  # the start state
        while True:
            # The prompt only grows by what was appended since the last request, whose size the
            # API reported, so this bounds the next request before it is paid for.
            prompt_tokens += json_bytes(messages[sent:])
            if dollars + config.prices.most(prompt_tokens, config.max_tokens) > config.max_dollars:
                return Attempt("dollar_cap", actions, dollars, steps)
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
            tool_round = await call_tools(client, response, config.max_actions - actions, presses)
            if tool_round.results:
                messages.append({"role": "user", "content": tool_round.results})
            for press in tool_round.presses:
                actions += 1
                if goal_step is None and press["state"] and segment.goal(press["state"]):
                    goal_step = actions
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
                buttons=[press["button"] for press in tool_round.presses],
                text="".join(block.text for block in response.content if block.type == "text"),
                stop_reason=response.stop_reason,
                state=tool_round.presses[-1]["state"] if tool_round.presses else None,
            )
            dollars += step.dollars
            steps.append(step)
            record(step)

            if goal_step is not None:
                outcome = "goal"
            elif actions >= config.max_actions:
                outcome = "action_cap"
            elif response.stop_reason != "tool_use":
                outcome = response.stop_reason or "end_turn"
            else:
                continue
            return Attempt(outcome, actions, dollars, steps, goal_step)


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
    """One attempt, logged under data/llm/<name>/ as steps.jsonl, presses.jsonl (the server's
    trace) and attempt.json, and in the logging schema as run <name> under data/runs/."""

    name: str
    segment: str = "fake"
    out: Path = Path("data/llm")
    runs: Path = Path("data/runs")


def git_commit() -> str:
    package = Path(__file__).resolve().parent
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=package, capture_output=True, text=True, check=True
    ).stdout.strip()


async def run(settings: Run, create: Create) -> Attempt:
    segment = SEGMENTS[settings.segment]
    folder = settings.out / settings.name
    folder.mkdir(parents=True)
    trace = (folder / "presses.jsonl").resolve()
    logs = run_dir(settings.name, settings.runs)
    # A run id already under data/runs, from an RL run or an earlier attempt, is refused.
    logs.mkdir(parents=True)
    backend, rom_sha1, parse = GAMES[segment.game]
    ids = {"run_id": settings.name, "seed": 0}
    fixed = ids | {
        "experiment_id": "",
        "arm": "",
        "algorithm": "llm",
        "trainer": "algoarcade.agent.loop",
        "env_id": segment.game,
        "backend": backend,
        "rom_sha1": rom_sha1,
        "start_state": settings.segment,
        "obs_mode": "screenshot" if settings.level == 0 else "state_json",
        "frames_per_decision": HOLD + RELEASE,
        "num_envs": 1,
        "reward_config_hash": hashlib.sha256(b"{}").hexdigest(),
        "git_commit": git_commit(),
        "hyperparameters": json.dumps(asdict(settings) | {"reward_weights": {}}, default=str),
        "step_log_every": 1,
        "machine": socket.gethostname(),
    }
    # Wall clock, as the schema means it, and the clock the server stamps presses with.
    started = time.time()
    write_run_row(logs, fixed | {"status": "running", "env_steps": 0, "wall_clock_s": 0.0})
    status = "crashed"
    try:
        with (folder / "steps.jsonl").open("w") as log:

            def record(step: Step) -> None:
                log.write(json.dumps(asdict(step)) + "\n")
                log.flush()

            arguments = (*segment.server, "--level", str(settings.level), "--trace", str(trace))
            async with Client(server(arguments), raise_exceptions=True) as client:
                attempt = await run_attempt(create, client, settings, segment, trace, record)
        presses = [json.loads(line) for line in trace.read_text().splitlines()]
        episode = {
            "start_state": settings.segment,
            "end_reason": attempt.outcome,
            "goal_reached": attempt.goal_step is not None,
            "goal_step": attempt.goal_step,
            "llm_model": settings.model,
            "llm_tool_level": settings.level,
            "llm_tokens_in": sum(step.prompt_tokens for step in attempt.steps),
            "llm_tokens_out": sum(step.output_tokens for step in attempt.steps),
            "llm_cost_usd": attempt.dollars,
        }
        write_attempt(logs, ids, presses, parse, started, episode)
        status = "finished"
    finally:
        actions = len(trace.read_text().splitlines()) - 1 if trace.is_file() else 0
        end = {
            "status": status,
            "env_steps": max(actions, 0),
            "wall_clock_s": time.time() - started,
        }
        write_run_row(logs, fixed | end)
    projection = project(attempt.steps, settings.prices)
    summary = asdict(settings) | asdict(attempt)
    summary |= {
        "per_action": attempt.per_action(),
        "projection_300_actions": asdict(projection) if projection else None,
    }
    del summary["steps"]
    (folder / "attempt.json").write_text(json.dumps(summary, indent=2, default=str))
    return attempt


@dataclass(frozen=True)
class Projection:
    requests: int
    presses_per_request: float
    start: int
    growth: float
    dollars: float


def project(steps: list[Step], prices: Prices, actions: int = 300) -> Projection | None:
    """The cost of an attempt of `actions` presses, from an attempt's log. It takes
    ceil(actions / mean presses per request) requests, or `actions` if nothing was pressed.
    The first is priced as measured. Request i has a prompt of the first's size plus i times
    the mean growth per request, sends the mean uncached input and cache write of the later
    requests and reads the rest from the cache (all of it is input if nothing was read from the
    cache), and is priced on the card its prompt size picks. None with fewer than two requests,
    as there is no growth to measure."""
    if len(steps) < 2:
        return None
    presses_per_request = sum(len(step.buttons) for step in steps) / len(steps)
    requests = math.ceil(actions / presses_per_request) if presses_per_request else actions
    first, later = steps[0], steps[1:]
    start = first.prompt_tokens
    growth = (steps[-1].prompt_tokens - start) / len(later)

    def mean(kind: str) -> int:
        return round(sum(getattr(step, kind) for step in later) / len(later))

    output = mean("output_tokens")
    uncached, written = mean("input_tokens"), mean("cache_write_tokens")
    cached = any(step.cache_read_tokens for step in later)
    dollars = first.dollars
    for index in range(1, requests):
        size = round(start + index * growth)
        if cached:
            usage = Usage(
                input_tokens=uncached,
                output_tokens=output,
                cache_creation_input_tokens=written,
                cache_read_input_tokens=size - uncached - written,
            )
        else:
            usage = Usage(input_tokens=size, output_tokens=output)
        dollars += prices.dollars(usage)
    return Projection(
        requests=requests,
        presses_per_request=presses_per_request,
        start=start,
        growth=growth,
        dollars=dollars,
    )


def report(attempt: Attempt, prices: Prices) -> str:
    lines = [
        f"{attempt.outcome}: {attempt.actions} actions in {len(attempt.steps)} requests, "
        f"${attempt.dollars:.4f}"
    ]
    per_action = attempt.per_action()
    if per_action is not None:
        dollars = per_action.pop("dollars")
        tokens = ", ".join(f"{kind} {value:.1f}" for kind, value in per_action.items())
        lines.append(f"per action: {tokens}, ${dollars:.6f}")
    projection = project(attempt.steps, prices)
    if projection is not None:
        lines.append(
            f"projected 300 actions ({projection.requests} requests at "
            f"{projection.presses_per_request:.1f} presses a request): ${projection.dollars:.2f}, "
            f"a prompt of {projection.start} tokens growing {projection.growth:.0f} a request, "
            "each priced on its card"
        )
    return "\n".join(lines)


def main() -> None:
    settings = tyro.cli(Run)
    attempt = asyncio.run(run(settings, AsyncAnthropic().messages.create))
    print(report(attempt, settings.prices))  # noqa: T201


if __name__ == "__main__":
    main()
