"""The MCP server a language model plays through. Contract: docs/harness-spec.md, MCP tool surface.

One process owns one game, named by `--game` and created when the server starts, optionally from
a save state in data/states/ named by `--state`. `--level` picks the tools. At L0, `get_state`
and `press` return only a 2x PNG of the screen. At L1 they return the state as compact JSON text
plus structured content, and the PNG when asked for one. L2 adds `navigate_to` and
`advance_dialogue`. `--trace FILE` writes the start state and then every press with the state
after it, one JSON line each, for the loop to count and grade.
"""

import argparse
import hashlib
import json
import logging
import struct
import zlib
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
from mcp.server.mcpserver import Context, Image, MCPServer
from mcp_types import CallToolResult, ImageContent, TextContent
from numpy.typing import NDArray
from pydantic_core import to_jsonable_python

from algoarcade.agent import tools
from algoarcade.agent.tools import DIALOGUE_PRESSES, NAVIGATE_PRESSES, Game
from algoarcade.backends.base import Button
from algoarcade.backends.fake import FakeBackend
from algoarcade.backends.mgba import MgbaBackend
from algoarcade.games.base import StateNotReady
from algoarcade.games.fake import FakeField, FakeReader
from algoarcade.games.firered.field import FireRedField
from algoarcade.games.firered.manifest import MANIFEST, ROM_SHA1
from algoarcade.games.firered.reader import FireRedReader
from algoarcade.rom import identify

# Names, not the Button bit positions an IntEnum would put in the schema a model reads.
ButtonName = Literal["A", "B", "SELECT", "START", "RIGHT", "LEFT", "UP", "DOWN", "R", "L"]
SCALE = 2
DATA = Path(__file__).resolve().parents[3] / "data"

logger = logging.getLogger(__name__)


def firered() -> Game:
    backend = MgbaBackend(DATA / "roms" / "firered.gba")
    return Game(backend, FireRedReader(backend.rom.sha1), FireRedField(backend.rom.sha1))


GAMES: dict[str, Callable[[], Game]] = {
    "fake": lambda: Game(FakeBackend(), FakeReader(), FakeField()),
    "firered": firered,
}


def start_state(name: str) -> bytes:
    """data/states/<name>.state. When the manifest lists it, the ROM must be the one the states
    were captured on and the file must match the manifest hash."""
    path = DATA / "states" / f"{name}.state"
    state = path.read_bytes()
    for entry in MANIFEST:
        if entry.name != name:
            continue
        rom_path = DATA / "roms" / "firered.gba"
        rom = identify(rom_path)
        if rom.sha1 != ROM_SHA1:
            raise ValueError(f"{rom_path} is {rom.name}; the start states were captured on rev 1")
        if hashlib.sha256(state).hexdigest() != entry.sha256:
            raise ValueError(f"{path} does not match the manifest hash")
    return state


def png(rgb: NDArray[np.uint8]) -> bytes:
    height, width, _ = rgb.shape
    # One filter byte (none) in front of each row.
    rows = np.hstack([np.zeros((height, 1), np.uint8), rgb.reshape(height, -1)]).tobytes()

    def chunk(kind: bytes, body: bytes) -> bytes:
        return (
            struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit RGB
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


def screen(game: Game) -> ImageContent:
    rgb = game.backend.screen().repeat(SCALE, axis=0).repeat(SCALE, axis=1)
    return Image(data=png(rgb), format="png").to_image_content()


def state_result(game: Game, screenshot: bool) -> CallToolResult:
    # Not an error: after a press the buttons were pressed, and a later read can succeed.
    try:
        state = to_jsonable_python(game.reader.read(game.backend))
        text = json.dumps(state, separators=(",", ":"))
    except StateNotReady as error:
        state = None
        text = f"State not ready: {error}"
    content: list[TextContent | ImageContent] = [TextContent(type="text", text=text)]
    if screenshot:
        content.append(screen(game))
    return CallToolResult(content=content, structured_content=state)


def outcome_result(game: Game, outcome: dict[str, Any], screenshot: bool) -> CallToolResult:
    """What a tool did, as JSON, then the state after it."""
    state = state_result(game, screenshot)
    head = TextContent(type="text", text=json.dumps(outcome, separators=(",", ":")))
    return CallToolResult(
        content=[head, *state.content],
        structured_content=outcome | {"state": state.structured_content},
    )


def build(  # noqa: C901 one nested function per tool
    make_game: Callable[[], Game], level: int = 1, trace: Path | None = None
) -> MCPServer[Game]:
    @asynccontextmanager
    async def one_game(_: MCPServer[Game]) -> AsyncGenerator[Game]:
        game = make_game()
        try:
            if trace is None:
                yield game
            else:
                with trace.open("w") as game.trace:
                    game.record(None)
                    yield game
        finally:
            game.backend.close()

    mcp = MCPServer("algoarcade", lifespan=one_game)

    def press_buttons(context: Context[Game, Any], buttons: list[ButtonName]) -> Game:
        game = context.request_context.lifespan_context
        # One press per button: the backend holds everything in one call down together.
        for name in buttons:
            game.press(Button[name])
        logger.info("press %s", buttons)
        return game

    if level == 0:

        @mcp.tool(name="get_state")
        def look(context: Context[Game, Any]) -> CallToolResult:
            """A screenshot of the screen."""
            return CallToolResult(content=[screen(context.request_context.lifespan_context)])

        @mcp.tool(name="press")
        def press_and_look(
            buttons: list[ButtonName], context: Context[Game, Any]
        ) -> CallToolResult:
            """Press these buttons in order, one decision each, then show the screen."""
            return CallToolResult(content=[screen(press_buttons(context, buttons))])

        return mcp

    @mcp.tool()
    def get_state(context: Context[Game, Any], screenshot: bool = False) -> CallToolResult:
        """Where the player is, as JSON, with a screenshot of the screen if you ask for one."""
        return state_result(context.request_context.lifespan_context, screenshot)

    @mcp.tool()
    def press(
        buttons: list[ButtonName], context: Context[Game, Any], screenshot: bool = False
    ) -> CallToolResult:
        """Press these buttons in order, one decision each, then report the state they left behind."""
        result = state_result(press_buttons(context, buttons), screenshot)
        logger.info("state %s", result.structured_content)
        return result

    if level == 1:
        return mcp

    @mcp.tool()
    def navigate_to(
        x: int,
        y: int,
        context: Context[Game, Any],
        max_presses: int = NAVIGATE_PRESSES,
        screenshot: bool = False,
    ) -> CallToolResult:
        """Walk to tile (x, y) of the current map, in the state's x and y, by the shortest path
        around walls and people. Stops on arrival, when a battle starts, when a text box opens,
        when the map changes, when there is no path, or after max_presses presses, and says
        which as reason (arrived, battle, dialogue, warp, no_path, press_cap), with the presses
        it made and the state after."""
        game = context.request_context.lifespan_context
        outcome = asdict(tools.navigate_to(game, x, y, max_presses))
        logger.info("navigate_to %s -> %s", (x, y), outcome)
        return outcome_result(game, outcome, screenshot)

    @mcp.tool()
    def advance_dialogue(
        context: Context[Game, Any], max_presses: int = DIALOGUE_PRESSES, screenshot: bool = False
    ) -> CallToolResult:
        """Press A until the text box closes or a choice appears, at most max_presses times.
        Returns the text shown, the reason it stopped (closed, choice, press_cap), the presses it
        made and the state after. It never answers a choice."""
        game = context.request_context.lifespan_context
        outcome = asdict(tools.advance_dialogue(game, max_presses))
        logger.info("advance_dialogue -> %s", outcome)
        return outcome_result(game, outcome, screenshot)

    return mcp


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m algoarcade.agent.server")
    parser.add_argument("--game", choices=sorted(GAMES), default="fake")
    parser.add_argument("--state", help="load data/states/<STATE>.state before serving")
    parser.add_argument("--level", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--trace", type=Path, help="write each press and the state after it here")
    args = parser.parse_args(argv)
    state = start_state(args.state) if args.state else None

    def start() -> Game:
        game = GAMES[args.game]()
        if state is not None:
            game.backend.load_state(state)
        return game

    build(start, args.level, args.trace).run()


if __name__ == "__main__":
    main()
