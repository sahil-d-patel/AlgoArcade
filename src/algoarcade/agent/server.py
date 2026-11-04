"""The MCP server a language model plays through. Contract: docs/harness-spec.md, MCP tool surface.

One process owns one game, named by `--game` and created when the server starts, optionally from
a save state in data/states/ named by `--state`. It serves `get_state` and `press`, each
returning the state as compact JSON text plus structured content, and a 2x PNG of the screen
when asked for one.
"""

import argparse
import hashlib
import json
import logging
import struct
import zlib
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from mcp.server.mcpserver import Context, Image, MCPServer
from mcp_types import CallToolResult, ImageContent, TextContent
from numpy.typing import NDArray
from pydantic_core import to_jsonable_python

from algoarcade.backends.base import Button, GameBackend
from algoarcade.backends.fake import FakeBackend
from algoarcade.backends.mgba import MgbaBackend
from algoarcade.games.base import StateNotReady, StateReader
from algoarcade.games.fake import FakeReader
from algoarcade.games.firered.manifest import MANIFEST, ROM_SHA1
from algoarcade.games.firered.reader import FireRedReader
from algoarcade.rom import identify

# Names, not the Button bit positions an IntEnum would put in the schema a model reads.
ButtonName = Literal["A", "B", "SELECT", "START", "RIGHT", "LEFT", "UP", "DOWN", "R", "L"]
SCALE = 2
DATA = Path(__file__).resolve().parents[3] / "data"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Game:
    backend: GameBackend
    reader: StateReader


def firered() -> Game:
    backend = MgbaBackend(DATA / "roms" / "firered.gba")
    return Game(backend, FireRedReader(backend.rom.sha1))


GAMES: dict[str, Callable[[], Game]] = {
    "fake": lambda: Game(FakeBackend(), FakeReader()),
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
        screen = game.backend.screen().repeat(SCALE, axis=0).repeat(SCALE, axis=1)
        content.append(Image(data=png(screen), format="png").to_image_content())
    return CallToolResult(content=content, structured_content=state)


def build(make_game: Callable[[], Game]) -> MCPServer[Game]:
    @asynccontextmanager
    async def one_game(_: MCPServer[Game]) -> AsyncGenerator[Game]:
        game = make_game()
        try:
            yield game
        finally:
            game.backend.close()

    mcp = MCPServer("algoarcade", lifespan=one_game)

    @mcp.tool()
    def get_state(context: Context[Game, Any], screenshot: bool = False) -> CallToolResult:
        """Where the player is, as JSON, with a screenshot of the screen if you ask for one."""
        return state_result(context.request_context.lifespan_context, screenshot)

    @mcp.tool()
    def press(
        buttons: list[ButtonName], context: Context[Game, Any], screenshot: bool = False
    ) -> CallToolResult:
        """Press these buttons in order, one decision each, then report the state they left behind."""
        game = context.request_context.lifespan_context
        # One press per button: the backend holds everything in one call down together.
        for name in buttons:
            game.backend.press([Button[name]])
        result = state_result(game, screenshot)
        logger.info("press %s -> %s", buttons, result.structured_content)
        return result

    return mcp


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m algoarcade.agent.server")
    parser.add_argument("--game", choices=sorted(GAMES), default="fake")
    parser.add_argument("--state", help="load data/states/<STATE>.state before serving")
    args = parser.parse_args(argv)
    state = start_state(args.state) if args.state else None

    def start() -> Game:
        game = GAMES[args.game]()
        if state is not None:
            game.backend.load_state(state)
        return game

    build(start).run()


if __name__ == "__main__":
    main()
