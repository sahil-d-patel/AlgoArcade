"""The MCP server a language model plays through. Contract: docs/harness-spec.md, MCP tool surface.

One process owns one game, named by `--game` and created when the server starts. It serves
`get_state` and `press`, each returning the state as compact JSON text plus structured content,
and a 2x PNG of the screen when asked for one.
"""

import argparse
import json
import logging
import struct
import zlib
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from mcp.server.mcpserver import Context, Image, MCPServer
from mcp_types import CallToolResult, ImageContent, TextContent
from numpy.typing import NDArray
from pydantic_core import to_jsonable_python

from algoarcade.backends.base import Button, GameBackend
from algoarcade.backends.fake import FakeBackend
from algoarcade.games.base import StateReader
from algoarcade.games.fake import FakeReader

# Names, not the Button bit positions an IntEnum would put in the schema a model reads.
ButtonName = Literal["A", "B", "SELECT", "START", "RIGHT", "LEFT", "UP", "DOWN", "R", "L"]
SCALE = 2

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Game:
    backend: GameBackend
    reader: StateReader


# FireRed joins when ENV-04 (MgbaBackend) and ENV-05 (FireRedReader) land.
GAMES: dict[str, Callable[[], Game]] = {"fake": lambda: Game(FakeBackend(), FakeReader())}


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
    state = to_jsonable_python(game.reader.read(game.backend))
    content: list[TextContent | ImageContent] = [
        TextContent(type="text", text=json.dumps(state, separators=(",", ":")))
    ]
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
    args = parser.parse_args(argv)
    build(GAMES[args.game]).run()


if __name__ == "__main__":
    main()
