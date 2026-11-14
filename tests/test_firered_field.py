import hashlib
from collections.abc import Iterator
from pathlib import Path

import pytest
from capture_states import Player

from algoarcade.agent.tools import DIALOGUE_PRESSES, Game, advance_dialogue, navigate_to
from algoarcade.backends.base import Button
from algoarcade.backends.mgba import MgbaBackend
from algoarcade.games.firered.field import FireRedField, decode
from algoarcade.games.firered.manifest import MANIFEST, ROM_SHA1
from algoarcade.games.firered.reader import FireRedReader

ROOT = Path(__file__).resolve().parents[1]
# Manifest states, seg_01_house_goal from scripts/capture_states.py and validate_rival_text from
# scripts/capture_validation_states.py, all rev 1.
HASHES = {entry.name: entry.sha256 for entry in MANIFEST} | {
    "seg_01_house_goal": "b3dadbaeb797824282cc74b8cf0013bd088a7358f0067e2d5b5f3854db4f09bf",
    "validate_rival_text": "5347edf9a63c170cfe341b42a3b3fa164ba08bc8f1e385e7e38849fa095e4473",
}


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


def load(backend: MgbaBackend, name: str) -> Game:
    path = ROOT / "data/states" / f"{name}.state"
    if not path.is_file():
        pytest.skip(f"Needs {path.name}")
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == HASHES[name]
    backend.load_state(data)
    return Game(backend, FireRedReader(backend.rom.sha1), FireRedField(backend.rom.sha1))


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
    state = game.reader.read(firered)
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
    assert game.reader.read(firered).location[:2] == (4, 3)  # MAP_PALLET_TOWN_PROFESSOR_OAKS_LAB


@pytest.mark.rom
def test_advance_dialogue_reads_the_rivals_challenge(firered: MgbaBackend) -> None:
    game = load(firered, "validate_rival_text")
    result = advance_dialogue(game)
    assert result.text == (
        "GREEN: Wait, A!\nLet\u2019s check out our POKéMON!\nCome on, I\u2019ll take you on!"
    )
    assert result.reason == "closed"
    assert 0 < result.presses < DIALOGUE_PRESSES
    assert not game.reader.read(firered).text_box_open


@pytest.mark.rom
def test_a_yes_no_menu_is_a_choice(firered: MgbaBackend) -> None:
    """At the middle ball in Oak's Lab, as scripts/capture_states.py plays it for seg_03_rival."""
    game = load(firered, "seg_01_house_goal")
    Player(firered).run("tRx6 tUx7 tD tRx3 U")
    firered.press([Button.A])
    state = game.reader.read(firered)
    assert state.text_box_open
    assert not game.field.choice(firered)
    result = advance_dialogue(game)
    assert result.reason == "choice"
    assert result.text.startswith("Hm! SQUIRTLE is your choice.")
    assert game.field.choice(firered)
    assert not game.reader.read(firered).text_box_open
