"""The MCP server a language model plays through. Contract: docs/harness-spec.md, MCP tool surface.

One process owns one backend, created when the server starts. This version owns the ROM-free
fake backend and serves `get_state`.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from mcp.server.mcpserver import Context, MCPServer

from algoarcade.backends.fake import FakeBackend
from algoarcade.games.fake import FakeReader, FakeState


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


if __name__ == "__main__":
    mcp.run()
