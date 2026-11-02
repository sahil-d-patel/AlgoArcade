import asyncio
import sys
from dataclasses import asdict
from typing import Any

from mcp import Client, StdioServerParameters

from algoarcade.backends.fake import FakeBackend
from algoarcade.games.fake import FakeReader


async def tools_and_state() -> tuple[list[str], dict[str, Any] | None]:
    params = StdioServerParameters(command=sys.executable, args=["-m", "algoarcade.agent.server"])
    async with Client(params, raise_exceptions=True) as client:
        listing = await client.list_tools()
        result = await client.call_tool("get_state")
    return [tool.name for tool in listing.tools], result.structured_content


def test_get_state_is_the_only_tool_and_matches_a_fresh_read() -> None:
    tools, state = asyncio.run(tools_and_state())
    assert tools == ["get_state"]
    assert state == asdict(FakeReader().read(FakeBackend()))
