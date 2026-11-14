import hashlib
import io
import itertools
import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from capture_states import Player

from algoarcade.agent.tools import (
    DIALOGUE_PRESSES,
    Game,
    advance_dialogue,
    navigate_to,
    shortest_path,
)
from algoarcade.backends.base import Button
from algoarcade.backends.mgba import MgbaBackend
from algoarcade.games.firered.field import FireRedField, decode
from algoarcade.games.firered.manifest import MANIFEST, ROM_SHA1
from algoarcade.games.firered.maps import MapGrid, load_maps
from algoarcade.games.firered.reader import FireRedReader

ROOT = Path(__file__).resolve().parents[1]
# Manifest states, seg_01_house_goal from scripts/capture_states.py and validate_rival_text from
# scripts/capture_validation_states.py, all rev 1.
HASHES = {entry.name: entry.sha256 for entry in MANIFEST} | {
    "seg_01_house_goal": "b3dadbaeb797824282cc74b8cf0013bd088a7358f0067e2d5b5f3854db4f09bf",
    "validate_rival_text": "5347edf9a63c170cfe341b42a3b3fa164ba08bc8f1e385e7e38849fa095e4473",
    "validate_potion": "202e53c584a704aa3e0bf695c734d604dff1b8d82aa0d6ccc8679437eb4b65d6",
}
ROUTE1 = (3, 19)


@pytest.fixture
def firered() -> Iterator[MgbaBackend]:
    rom = ROOT / "data/roms/firered.gba"
    if not rom.is_file():
        pytest.skip("Needs the local FireRed ROM")
    pytest.importorskip(
        "mgba.core", reason="mGBA binding did not load; see docs/setup.md", exc_type=ImportError
    )
    backend = MgbaBackend(rom)
    if backend.rom.sha1 != ROM_SHA1:
        backend.close()
        pytest.skip("The states were captured on rev 1")
    try:
        yield backend
    finally:
        backend.close()


@pytest.fixture
def maps() -> dict[tuple[int, int], MapGrid]:
    try:
        return load_maps()
    except FileNotFoundError:
        pytest.skip("Needs the map data: uv run python scripts/fetch_maps.py")


def load(
    backend: MgbaBackend, name: str, maps: dict[tuple[int, int], MapGrid] | None = None
) -> Game:
    path = ROOT / "data/states" / f"{name}.state"
    if not path.is_file():
        pytest.skip(f"Needs {path.name}")
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == HASHES[name]
    backend.load_state(data)
    field = FireRedField(backend.rom.sha1, maps)
    return Game(backend, FireRedReader(backend.rom.sha1), field, io.StringIO())


def test_decode_skips_control_codes_and_stops_at_the_terminator() -> None:
    # "BULBASAUR" with a highlight code before it and a colour code after, as gDisplayedStringBattle
    # held it in the rival battle, then the terminator and leftover bytes.
    raw = bytes.fromhex("fc0202bccfc6bcbbcdbbcfccfc010bb5ffbbbb")
    assert decode(raw) == "BULBASAUR♂"


# Seen on screen at seg_02_pallet: the player below the lab door at (16, 14), a woman three tiles
# left and three down, the pond's top-right water tile three down and six left. Walking up from
# (15, 14) bumped the lab wall; walking left from (16, 14) moved.
@pytest.mark.rom
@pytest.mark.parametrize(
    ("tile", "walkable"),
    [((16, 14), True), ((15, 14), True), ((15, 13), False), ((13, 17), False), ((10, 17), False)],
    ids=["player", "floor", "lab wall", "woman", "water"],
)
def test_pallet_town_walkability(
    firered: MgbaBackend, tile: tuple[int, int], walkable: bool
) -> None:
    game = load(firered, "seg_02_pallet")
    grid = game.field.walkable(firered)
    assert grid.shape == (20, 24)
    assert grid[tile[1], tile[0]] == walkable


@pytest.mark.rom
def test_navigate_to_a_tile_17_away_on_seg_02(firered: MgbaBackend) -> None:
    game = load(firered, "seg_02_pallet")
    result = navigate_to(game, 5, 8)
    state = game.read()
    assert result.reason == "arrived"
    assert (state.location, result.presses) == ((3, 0, 5, 8), 20)


@pytest.mark.rom
def test_a_door_is_reachable_and_warps_on_its_own(firered: MgbaBackend) -> None:
    """The lab door's block has collision 1. The player stands on it, then the game walks him in:
    69 frames after the press, measured on rev 1."""
    game = load(firered, "seg_02_pallet")
    result = navigate_to(game, 16, 13)
    assert (result.reason, result.presses) == ("arrived", 1)
    firered.step(120)
    assert game.read().location[:2] == (4, 3)  # MAP_PALLET_TOWN_PROFESSOR_OAKS_LAB


@pytest.mark.rom
def test_advance_dialogue_reads_the_rivals_challenge(firered: MgbaBackend) -> None:
    game = load(firered, "validate_rival_text")
    result = advance_dialogue(game)
    assert result.text == (
        "GREEN: Wait, A!\nLet\u2019s check out our POKéMON!\nCome on, I\u2019ll take you on!"
    )
    assert result.reason == "closed"
    assert 0 < result.presses < DIALOGUE_PRESSES
    assert not game.read().text_box_open


@pytest.mark.rom
def test_a_yes_no_menu_is_a_choice(firered: MgbaBackend) -> None:
    """At the middle ball in Oak's Lab, as scripts/capture_states.py plays it for seg_03_rival."""
    game = load(firered, "seg_01_house_goal")
    Player(firered).run("tRx6 tUx7 tD tRx3 U")
    firered.press([Button.A])
    state = game.read()
    assert state.text_box_open
    assert not game.field.choice(firered)
    result = advance_dialogue(game)
    assert result.reason == "choice"
    assert result.text.startswith("Hm! SQUIRTLE is your choice.")
    assert game.field.choice(firered)
    assert not game.read().text_box_open


@pytest.mark.rom
def test_navigate_to_jumps_a_route_1_ledge_without_a_failed_step(
    firered: MgbaBackend, maps: dict[tuple[int, int], MapGrid]
) -> None:
    """validate_potion stands at (7, 28), above the ledges on row 31. (3, 32) is below them, and
    every shortest way there lands on it from the ledge at (3, 31), out of the grass."""
    game = load(firered, "validate_potion", maps)
    state = game.read()
    assert state.location == (*ROUTE1, 7, 28)
    plan = shortest_path(game.field.walkable(firered), (7, 28), (3, 32), game.field.ledges(state))
    assert plan is not None
    tiles = [(7, 28), *(landing for _, landing in plan)]
    jumps = [(a, b) for a, b in itertools.pairwise(tiles) if b[1] - a[1] == 2]
    assert [(a[1], b[1]) for a, b in jumps] == [(30, 32)]
    result = navigate_to(game, 3, 32)
    assert (result.reason, game.read().location) == ("arrived", (*ROUTE1, 3, 32))
    assert isinstance(game.trace, io.StringIO)
    pressed = [json.loads(line)["state"] for line in game.trace.getvalue().splitlines()]
    # Every press landed on the plan or turned in place: no step failed, so nothing was replanned.
    assert {(state["x"], state["y"]) for state in pressed} <= set(tiles)
    # Seven steps on the plan and one press that only turned the player.
    assert (len(plan), result.presses) == (7, 8)


@pytest.mark.rom
def test_navigate_to_plans_around_ledges_going_up(
    firered: MgbaBackend, maps: dict[tuple[int, int], MapGrid]
) -> None:
    game = load(firered, "validate_potion", maps)
    plan = shortest_path(
        game.field.walkable(firered), (2, 32), (2, 29), game.field.ledges(game.read())
    )
    assert plan is not None
    # Up through the gap in row 31 at x 6 to 8, never across a ledge.
    crossing = [landing for _, landing in plan if landing[1] == 31]
    assert len(crossing) == 1
    assert 6 <= crossing[0][0] <= 8
