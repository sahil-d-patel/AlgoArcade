import asyncio
import base64
import json
import struct
import subprocess
import sys
import threading
import zlib
from dataclasses import asdict
from pathlib import Path
from typing import Any, get_args

import numpy as np
import pytest
from mcp import Client, StdioServerParameters
from mcp_types import CallToolResult
from numpy.typing import NDArray

from algoarcade.agent import server as agent_server
from algoarcade.agent.loop import server
from algoarcade.agent.server import SCALE, ButtonName, Game, build
from algoarcade.backends.base import Button, GameBackend
from algoarcade.backends.fake import FakeBackend
from algoarcade.backends.mgba import MgbaBackend
from algoarcade.games.base import StateNotReady
from algoarcade.games.fake import FakeField, FakeReader, FakeState
from algoarcade.games.firered.manifest import ROM_SHA1 as MANIFEST_ROM_SHA1
from algoarcade.games.firered.reader import FireRedReader
from algoarcade.rom import KNOWN_ROMS

PARAMS = StdioServerParameters(command=sys.executable, args=["-m", "algoarcade.agent.server"])
MCP_JSON = Path(__file__).parents[1] / ".mcp.json"
ROM = Path(__file__).parents[1] / "data" / "roms" / "firered.gba"
STATES = Path(__file__).parents[1] / "data" / "states"
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


def png_pixels(png: bytes) -> NDArray[np.uint8]:
    """Decode the server's PNG: 8-bit RGB, every row unfiltered."""
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    chunks: dict[bytes, bytes] = {}
    offset = 8
    while offset < len(png):
        (length,) = struct.unpack(">I", png[offset : offset + 4])
        kind = png[offset + 4 : offset + 8]
        chunks[kind] = chunks.get(kind, b"") + png[offset + 8 : offset + 8 + length]
        offset += 12 + length
    width, height, depth, colour, *_ = struct.unpack(">IIBBBBB", chunks[b"IHDR"])
    assert (depth, colour) == (8, 2)
    rows = np.frombuffer(zlib.decompress(chunks[b"IDAT"]), np.uint8).reshape(height, -1)
    assert not rows[:, 0].any()
    return rows[:, 1:].reshape(height, width, 3)


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
    screen = FakeBackend().screen()
    state, png = asyncio.run(state_and_png(PARAMS))
    assert state == asdict(FakeReader().read(FakeBackend()))
    assert SCALE == 2
    assert np.array_equal(png_pixels(png), screen.repeat(2, 0).repeat(2, 1))


def test_ten_tool_calls_over_stdio() -> None:
    states = asyncio.run(ten_calls())
    assert len(states) == 10
    assert [state["y"] for state in states if state] == [1, 2, 2, 3, 3, 4, 4, 4, 4, 4]


@pytest.mark.rom
@pytest.mark.parametrize("name", ["seg_01_house", "seg_02_pallet"])
def test_firered_get_state_matches_the_reader_and_the_png_is_480x320(name: str) -> None:
    """Two states, so a constant location fails."""
    state_file = STATES / f"{name}.state"
    if not ROM.is_file() or not state_file.is_file():
        pytest.skip(f"needs {ROM} and {state_file}")
    pytest.importorskip(
        "mgba.core", reason="mGBA binding did not load; see docs/setup.md", exc_type=ImportError
    )
    backend = MgbaBackend(ROM)
    try:
        backend.load_state(state_file.read_bytes())
        expected = FireRedReader(backend.rom.sha1).read(backend)
    finally:
        backend.close()
    state, png = asyncio.run(state_and_png(server(("--game", "firered", "--state", name))))
    assert state is not None
    assert (state["map_group"], state["map_num"], state["x"], state["y"]) == expected.location
    assert png_pixels(png).shape == (320, 480, 3)


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


class NotReadyPastX1:
    """The fake reader, raising once the player leaves x=1, as FireRed's can mid-warp."""

    def read(self, backend: GameBackend) -> FakeState:
        state = FakeReader().read(backend)
        if state.x > 1:
            raise StateNotReady("the player left x=1")
        return state


async def press_into_a_state_that_is_not_ready() -> tuple[CallToolResult, CallToolResult]:
    server = build(lambda: Game(FakeBackend(), NotReadyPastX1(), FakeField()))
    async with Client(server, raise_exceptions=True) as client:
        pressed = await client.call_tool("press", {"buttons": ["RIGHT"], "screenshot": True})
        read_back = await client.call_tool("get_state")
    return pressed, read_back


def test_a_state_that_is_not_ready_is_reported_not_raised() -> None:
    pressed, read_back = asyncio.run(press_into_a_state_that_is_not_ready())
    for result in (pressed, read_back):
        assert not result.is_error
        assert result.structured_content is None
        assert result.content[0].type == "text"
        assert result.content[0].text == "State not ready: the player left x=1"
    assert [block.type for block in pressed.content] == ["text", "image"]


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


@pytest.fixture
def states(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A data/ folder with no ROM, one manifest state with the wrong bytes and one of our own."""
    monkeypatch.setattr(agent_server, "DATA", tmp_path)
    (tmp_path / "states").mkdir()
    (tmp_path / "states" / "seg_01_house.state").write_bytes(b"not the captured state")
    (tmp_path / "states" / "mine.state").write_bytes(b"a state of my own")
    return tmp_path


def test_a_manifest_state_must_match_its_hash(
    states: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(agent_server, "identify", lambda _: KNOWN_ROMS[MANIFEST_ROM_SHA1])
    with pytest.raises(ValueError, match="does not match the manifest hash"):
        agent_server.start_state("seg_01_house")
    assert agent_server.start_state("mine") == b"a state of my own"


def test_a_manifest_state_needs_the_rom_it_was_captured_on(
    states: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (rev0,) = (
        rom
        for rom in KNOWN_ROMS.values()
        if rom.game == "firered" and rom.sha1 != MANIFEST_ROM_SHA1
    )
    monkeypatch.setattr(agent_server, "identify", lambda _: rev0)
    with pytest.raises(ValueError, match="captured on rev 1"):
        agent_server.start_state("seg_01_house")
    assert agent_server.start_state("mine") == b"a state of my own"


async def tools_and_results(level: int) -> tuple[list[str], list[CallToolResult]]:
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "algoarcade.agent.server", "--level", str(level)]
    )
    async with Client(params, raise_exceptions=True) as client:
        names = sorted(tool.name for tool in (await client.list_tools()).tools)
        results = [
            await client.call_tool("get_state"),
            await client.call_tool("press", {"buttons": ["RIGHT"]}),
        ]
    return names, results


@pytest.mark.parametrize(
    ("level", "names"),
    [
        (0, ["get_state", "press"]),
        (1, ["get_state", "press"]),
        (2, ["advance_dialogue", "get_state", "navigate_to", "press"]),
    ],
)
def test_each_level_lists_its_tools(level: int, names: list[str]) -> None:
    assert asyncio.run(tools_and_results(level))[0] == names


def test_level_0_returns_only_the_screen() -> None:
    _, results = asyncio.run(tools_and_results(0))
    for result in results:
        assert not result.is_error
        assert result.structured_content is None
        assert [block.type for block in result.content] == ["image"]
    pressed = FakeBackend()
    pressed.press([Button.RIGHT])
    (image,) = results[1].content
    assert image.type == "image"
    assert np.array_equal(
        png_pixels(base64.b64decode(image.data)), pressed.screen().repeat(2, 0).repeat(2, 1)
    )


async def navigate_and_talk(trace: Path) -> tuple[CallToolResult, CallToolResult]:
    server = build(lambda: Game(FakeBackend(), FakeReader(), FakeField()), level=2, trace=trace)
    async with Client(server, raise_exceptions=True) as client:
        walked = await client.call_tool("navigate_to", {"x": 4, "y": 1})
        talked = await client.call_tool("advance_dialogue")
    return walked, talked


def test_l2_tools_report_what_they_did_and_trace_every_press(tmp_path: Path) -> None:
    trace = tmp_path / "presses.jsonl"
    walked, talked = asyncio.run(navigate_and_talk(trace))
    backend = FakeBackend()
    for _ in range(3):
        backend.press([Button.RIGHT])
    after = asdict(FakeReader().read(backend))
    assert walked.structured_content == {"reason": "arrived", "presses": 3, "state": after}
    assert [json.loads(block.text) for block in walked.content if block.type == "text"] == [
        {"reason": "arrived", "presses": 3},
        after,
    ]
    assert talked.structured_content == {
        "reason": "closed",
        "presses": 0,
        "text": "",
        "state": after,
    }
    lines = [json.loads(line) for line in trace.read_text().splitlines()]
    assert [line["button"] for line in lines] == [None, "RIGHT", "RIGHT", "RIGHT"]
    assert lines[0]["state"] == asdict(FakeReader().read(FakeBackend()))
    assert lines[-1]["state"] == after
