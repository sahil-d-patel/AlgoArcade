import asyncio
import base64
import json
import struct
import subprocess
import sys
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any, get_args

import pytest
from mcp import Client, StdioServerParameters

from algoarcade.agent.server import SCALE, ButtonName
from algoarcade.backends.base import Button
from algoarcade.backends.fake import FakeBackend
from algoarcade.games.fake import FakeReader
from algoarcade.rom import identify

PARAMS = StdioServerParameters(command=sys.executable, args=["-m", "algoarcade.agent.server"])
MCP_JSON = Path(__file__).parents[1] / ".mcp.json"
ROM = Path(__file__).parents[1] / "data" / "roms" / "firered.gba"
# Two steps right lands on the chest, A opens it. Each button is its own decision.
TO_THE_CHEST = ["RIGHT", "RIGHT", "A"]
RAW_REQUESTS: list[dict[str, Any]] = [
    {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "stdout-test", "version": "0"},
        },
    },
    {"jsonrpc": "2.0", "method": "notifications/initialized"},
    {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {"name": "get_state", "arguments": {}},
    },
    {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {"name": "press", "arguments": {"buttons": ["A"]}},
    },
]


async def tools_press_and_read() -> tuple[list[str], dict[str, Any] | None, dict[str, Any] | None]:
    async with Client(PARAMS, raise_exceptions=True) as client:
        listing = await client.list_tools()
        pressed = await client.call_tool("press", {"buttons": TO_THE_CHEST})
        read_back = await client.call_tool("get_state")
    names = [tool.name for tool in listing.tools]
    return names, pressed.structured_content, read_back.structured_content


async def fresh_state() -> dict[str, Any] | None:
    async with Client(PARAMS, raise_exceptions=True) as client:
        return (await client.call_tool("get_state")).structured_content


async def state_and_png(params: StdioServerParameters) -> tuple[dict[str, Any] | None, bytes]:
    async with Client(params, raise_exceptions=True) as client:
        result = await client.call_tool("get_state", {"screenshot": True})
    text, image = result.content
    assert text.type == "text"
    assert image.type == "image"
    assert json.loads(text.text) == result.structured_content
    return result.structured_content, base64.b64decode(image.data)


def png_size(png: bytes) -> tuple[int, int]:
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", png[16:24])
    return width, height


async def ten_calls() -> list[dict[str, Any] | None]:
    async with Client(PARAMS, raise_exceptions=True) as client:
        states = []
        for index in range(10):
            if index % 2:
                result = await client.call_tool("press", {"buttons": ["DOWN"], "screenshot": True})
            else:
                result = await client.call_tool("get_state")
            assert not result.is_error
            states.append(result.structured_content)
    return states


async def listed_tools(params: StdioServerParameters) -> list[str]:
    async with Client(params, raise_exceptions=True) as client:
        return [tool.name for tool in (await client.list_tools()).tools]


async def moved_and_untouched() -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    async with (
        Client(PARAMS, raise_exceptions=True) as moving,
        Client(PARAMS, raise_exceptions=True) as idle,
    ):
        await moving.call_tool("press", {"buttons": ["RIGHT"]})
        moved = await moving.call_tool("get_state")
        untouched = await idle.call_tool("get_state")
    return moved.structured_content, untouched.structured_content


def test_two_tools_and_press_matches_a_direct_backend_run() -> None:
    names, pressed, read_back = asyncio.run(tools_press_and_read())
    backend = FakeBackend()
    for name in TO_THE_CHEST:
        backend.press([Button[name]])
    assert sorted(names) == ["get_state", "press"]
    assert pressed == read_back == asdict(FakeReader().read(backend))


def test_get_state_matches_a_fresh_read() -> None:
    assert asyncio.run(fresh_state()) == asdict(FakeReader().read(FakeBackend()))


def test_screenshot_is_the_screen_upscaled_twice() -> None:
    height, width, _ = FakeBackend().screen().shape
    state, png = asyncio.run(state_and_png(PARAMS))
    assert state == asdict(FakeReader().read(FakeBackend()))
    assert png_size(png) == (width * SCALE, height * SCALE)


def test_ten_tool_calls_over_stdio() -> None:
    states = asyncio.run(ten_calls())
    assert len(states) == 10
    assert [state["y"] for state in states if state] == [1, 2, 2, 3, 3, 4, 4, 4, 4, 4]


@pytest.mark.rom
def test_firered_get_state_matches_the_reader_and_the_png_is_480x320() -> None:
    """Waits on ENV-04 (MgbaBackend), ENV-05 (FireRedReader) and a `firered` entry in GAMES."""
    if not ROM.exists():
        pytest.skip(f"no ROM at {ROM}")
    mgba = pytest.importorskip("algoarcade.backends.mgba")
    firered = pytest.importorskip("algoarcade.games.firered.reader")
    backend = mgba.MgbaBackend(ROM)
    try:
        expected = firered.FireRedReader(identify(ROM).sha1).read(backend)
    finally:
        backend.close()
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "algoarcade.agent.server", "--game", "firered"]
    )
    state, png = asyncio.run(state_and_png(params))
    assert state is not None
    assert (state["map_group"], state["map_num"], state["x"], state["y"]) == expected.location
    assert png_size(png) == (480, 320)


def test_button_names_are_the_backend_enum() -> None:
    assert set(get_args(ButtonName)) == {button.name for button in Button}


def test_only_protocol_frames_reach_stdout_and_the_log_reaches_stderr() -> None:
    process = subprocess.Popen(
        [sys.executable, "-m", "algoarcade.agent.server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdin
    assert process.stdout
    # readline blocks with no deadline of its own, so a server that stops replying would hang
    # the job. Killing it closes the pipe, readline returns "" and the assertion below fails.
    watchdog = threading.Timer(30, process.kill)
    watchdog.start()
    replies = []
    try:
        for request in RAW_REQUESTS:
            process.stdin.write(json.dumps(request) + "\n")
            process.stdin.flush()
            # Reply by reply: writing them all and closing stdin ends the server mid-flight.
            if "id" in request:
                replies.append(process.stdout.readline())
    finally:
        watchdog.cancel()
    rest, stderr = process.communicate(timeout=30)

    assert "" not in replies, "the watchdog killed a server that stopped replying"
    assert [json.loads(line)["jsonrpc"] for line in replies] == ["2.0"] * 3
    assert rest == ""
    assert "press ['A']" in stderr


def test_two_servers_hold_independent_state() -> None:
    moved, untouched = asyncio.run(moved_and_untouched())
    assert moved != untouched
    assert untouched == asdict(FakeReader().read(FakeBackend()))


def test_mcp_json_launches_a_server_that_lists_both_tools() -> None:
    (entry,) = json.loads(MCP_JSON.read_text())["mcpServers"].values()
    assert set(entry) == {"command", "args"}
    assert "${" not in MCP_JSON.read_text()
    launched = StdioServerParameters(
        command=entry["command"], args=entry["args"], cwd=MCP_JSON.parent
    )
    assert sorted(asyncio.run(listed_tools(launched))) == ["get_state", "press"]
