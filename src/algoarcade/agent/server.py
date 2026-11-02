"""The MCP server a language model plays through. Contract: docs/harness-spec.md, MCP tool surface.

One process owns one backend, created when the server starts. This version owns the ROM-free
fake backend and serves `get_state` and `press`.
"""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, Literal

from mcp.server.mcpserver import Context, MCPServer

from algoarcade.backends.base import Button
from algoarcade.backends.fake import FakeBackend
from algoarcade.games.fake import FakeReader, FakeState

# Names, not the Button bit positions an IntEnum would put in the schema a model reads.
ButtonName = Literal["A", "B", "SELECT", "START", "RIGHT", "LEFT", "UP", "DOWN", "R", "L"]

logger = logging.getLogger(__name__)


@asynccontextmanager
async def one_fake_backend(_: MCPServer[FakeBackend]) -> AsyncGenerator[FakeBackend]:
    backend = FakeBackend()
    try:
        yield backend
    finally:
        backend.close()


mcp = MCPServer("algoarcade", lifespan=one_fake_backend)


@mcp.tool()
def get_state(context: Context[FakeBackend, Any]) -> FakeState:
    """Where the player is: the room, the tile, whether the chest is open, and the frame count."""
    return FakeReader().read(context.request_context.lifespan_context)


@mcp.tool()
def press(buttons: list[ButtonName], context: Context[FakeBackend, Any]) -> FakeState:
    """Press these buttons in order, one decision each, then report the state they left behind."""
    backend = context.request_context.lifespan_context
    # One press per button: the backend holds everything in one call down together.
    for name in buttons:
        backend.press([Button[name]])
    state = FakeReader().read(backend)
    logger.info("press %s -> %s", buttons, state.location)
    return state


if __name__ == "__main__":
    mcp.run()
