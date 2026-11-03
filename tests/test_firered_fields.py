import hashlib
from collections.abc import Iterator
from pathlib import Path

import pytest

from algoarcade.backends.base import Button
from algoarcade.backends.mgba import MgbaBackend
from algoarcade.games.firered.reader import FireRedReader, FireRedState
from algoarcade.rom import identify

ROOT = Path(__file__).resolve().parents[1]
CAPTURE_SHA256 = {
    "env08-pokedex.state": "11b891fdc87972644c45198728edee6bc4d4f269d2713fb2881dd084c8bbe3bd",
    "env08-money-3000.state": "37ad9fefd6813f5f68362247ee276d6205dd2b4a0823403c3a69f41b325ae093",
    "env08-money-3080.state": "fc957ea809f0b2f687755a18ba3ca038a5184c6850ef79f433b01947fd18d7c8",
    "env08-before-starter.state": "ff47f57f7e3071786053af11d709be276c8a56cef5de0954c27e1cb0d6240fb5",
    "env08-starter-overworld.state": "a8f139228c9a17a798fe5b69caa18fe3dbc1b7ab116d385116288a452c015c9b",
    "env08-party.state": "43fd78f7ed377c79fa03c60e8573953347562edbf9d80bc958b60dbef5365dc4",
    "env08-moves.state": "f5cc22d91353dba7ccc873a8634df9b00bc59909f1cac201eb0c18027d6f7ac5",
    "env08-party-hp13.state": "3f0b4400db4d35ba73f19af7e5beaa575f8b63c3ece88102b240540fba20cf78",
    "env08-after-rival.state": "78b86e532f6174576198098cdbd83315f36a4d9051cf46f05d983f971d51df9a",
    "env08-rival-battle.state": "ec0e4d1ab834b444fad12e263bb9327648c1af9d32d8820228af8484bedfb3e6",
    "env08-oak-dialogue.state": "b54328c9856111da3cdb48f6c299d8dac4f1ef76c2f84ed47587d070677d80af",
    "env08-warp-before.state": "e8e6c7c190f299eb72ea1feaf41d42c173897c4fab28ad6dc67a567530191e3b",
    "env08-warp-after.state": "3a8f869e05afffafa706ec9da67a7e8fb13b33db21bff5d8961bae8f4b68538e",
    "env08-options-default.state": "da916d488fb064ad0e0bc3c837fea670f73f4684bcc94504a307e8a9bdbf4e97",
    "env08-options-fast-off.state": "ae91101713f18de3d2202d44d3a9dc104123c8aaec840937b88d77ee46f2f4e0",
}


@pytest.fixture
def firered_backend() -> Iterator[MgbaBackend]:
    rom = ROOT / "data/roms/firered.gba"
    if not rom.is_file():
        pytest.skip("Needs the local FireRed ROM")
    if identify(rom).sha1 != "41cb23d8dccc8ebd7c649cd8fbb58eeace6e2fdc":
        pytest.skip("These captures were recorded on FireRed USA rev 0")
    backend = MgbaBackend(rom)
    try:
        yield backend
    finally:
        backend.close()


def load_capture(backend: MgbaBackend, filename: str) -> FireRedState:
    path = ROOT / "data/states" / filename
    if not path.is_file():
        pytest.skip(f"Needs local capture {filename}")
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == CAPTURE_SHA256[filename]
    backend.load_state(data)
    return FireRedReader(backend.rom.sha1).read(backend)


# Money, badges, party, Pokedex and options were read from their screens.
# Progress counts describe the observed story events and trainer wins.
@pytest.mark.rom
@pytest.mark.parametrize(
    ("filename", "field", "expected"),
    [
        ("env08-money-3000.state", "money", 3000),
        ("env08-money-3080.state", "money", 3080),
        ("env08-money-3000.state", "badge_flags", 0),
        ("env08-money-3000.state", "badge_count", 0),
        ("env08-warp-before.state", "location", (4, 1, 10, 2)),
        ("env08-warp-after.state", "location", (4, 0, 10, 2)),
        ("env08-before-starter.state", "party_count", 0),
        ("env08-party.state", "party_count", 1),
        ("env08-party.state", "party_level_sum", 5),
        ("env08-after-rival.state", "party_level_sum", 6),
        ("env08-party.state", "party_hp_frac", 1.0),
        ("env08-party-hp13.state", "party_hp_frac", 13 / 19),
        ("env08-before-starter.state", "trainers_defeated", 0),
        ("env08-after-rival.state", "trainers_defeated", 1),
        ("env08-before-starter.state", "story_flags", 0),
        ("env08-after-rival.state", "story_flags", 2),
        ("env08-money-3000.state", "world_map_flags", 0),
        ("env08-after-rival.state", "world_map_flags", 1),
        ("env08-before-starter.state", "pokedex_owned", 0),
        ("env08-party.state", "pokedex_owned", 1),
        ("env08-party.state", "pokedex_seen", 1),
        ("env08-rival-battle.state", "pokedex_seen", 2),
        ("env08-pokedex.state", "pokedex_seen", 4),
        ("env08-pokedex.state", "pokedex_owned", 1),
        ("env08-pokedex.state", "story_flags", 3),
        ("env08-pokedex.state", "world_map_flags", 2),
        ("env08-starter-overworld.state", "in_battle", False),
        ("env08-rival-battle.state", "in_battle", True),
        ("env08-starter-overworld.state", "text_active", False),
        ("env08-oak-dialogue.state", "text_active", True),
        ("env08-starter-overworld.state", "script_active", False),
        ("env08-oak-dialogue.state", "script_active", True),
        ("env08-starter-overworld.state", "controls_locked", False),
        ("env08-oak-dialogue.state", "controls_locked", True),
        ("env08-options-default.state", "text_speed", 1),
        ("env08-options-fast-off.state", "text_speed", 2),
        ("env08-options-default.state", "battle_scene_off", False),
        ("env08-options-fast-off.state", "battle_scene_off", True),
    ],
)
def test_firered_fields_state(
    firered_backend: MgbaBackend, filename: str, field: str, expected: object
) -> None:
    assert getattr(load_capture(firered_backend, filename), field) == expected


@pytest.mark.rom
@pytest.mark.parametrize(
    ("filename", "field", "expected"),
    [
        ("env08-party.state", "level", 5),
        ("env08-after-rival.state", "level", 6),
        ("env08-party.state", "hp", 19),
        ("env08-party-hp13.state", "hp", 13),
        ("env08-party.state", "max_hp", 19),
        ("env08-after-rival.state", "max_hp", 21),
        ("env08-party.state", "status", 0),
        ("env08-party.state", "species", 7),
        ("env08-moves.state", "moves", (33, 39, 0, 0)),
    ],
)
def test_firered_fields_party(
    firered_backend: MgbaBackend, filename: str, field: str, expected: object
) -> None:
    # Party/summary screens show Squirtle, Tackle, Tail Whip and healthy status.
    assert getattr(load_capture(firered_backend, filename).party[0], field) == expected


@pytest.mark.rom
@pytest.mark.parametrize(
    ("filename", "valid"),
    [
        ("env08-options-default.state", False),
        ("env08-options-fast-off.state", True),
    ],
)
def test_firered_fields_options_check(
    firered_backend: MgbaBackend, filename: str, valid: bool
) -> None:
    load_capture(firered_backend, filename)
    reader = FireRedReader(firered_backend.rom.sha1)
    if valid:
        reader.check_options(firered_backend)
    else:
        with pytest.raises(ValueError, match="fast text speed and battle scene off"):
            reader.check_options(firered_backend)


@pytest.mark.rom
def test_firered_fields_warp_and_battle_trace(firered_backend: MgbaBackend) -> None:
    previous = load_capture(firered_backend, "env08-warp-before.state")
    reader = FireRedReader(firered_backend.rom.sha1)
    maps = {previous.location[:2]}
    battle_states = {previous.in_battle}
    money_values = {previous.money}
    # Recorded input from the house stairs to the lab rival victory: 163 decisions.
    inputs = [
        (Button.LEFT, 16, 90, 1),
        (Button.DOWN, 96, 8, 1),
        (Button.LEFT, 80, 8, 1),
        (Button.DOWN, 48, 90, 1),
        (Button.DOWN, 24, 100, 1),
        (Button.UP, 200, 60, 1),
        (Button.DOWN, 160, 120, 1),
        (Button.LEFT, 16, 8, 1),
        (Button.DOWN, 40, 120, 1),
        (Button.RIGHT, 64, 8, 1),
        (Button.UP, 144, 40, 1),
        (Button.RIGHT, 32, 8, 1),
        (Button.UP, 32, 120, 1),
        (None, 0, 30, 1),
        (Button.A, 8, 24, 3),
        (Button.A, 8, 200, 1),
        (Button.A, 8, 24, 3),
        (Button.A, 8, 300, 1),
        (None, 0, 240, 1),
        (Button.A, 8, 24, 2),
        (Button.A, 8, 120, 1),
        (Button.A, 8, 24, 8),
        (Button.A, 8, 90, 1),
        (Button.A, 8, 24, 3),
        (Button.A, 8, 60, 1),
        (Button.A, 8, 24, 7),
        (Button.A, 8, 60, 1),
        (Button.A, 8, 24, 1),
        (Button.A, 8, 90, 2),
        (Button.DOWN, 16, 8, 1),
        (Button.RIGHT, 64, 8, 1),
        (Button.UP, 8, 8, 1),
        (Button.A, 8, 80, 1),
        (Button.A, 8, 24, 24),
        (None, 0, 40, 1),
        (Button.DOWN, 16, 8, 1),
        (Button.RIGHT, 64, 8, 1),
        (Button.LEFT, 16, 8, 1),
        (Button.UP, 2, 8, 1),
        (Button.A, 8, 60, 1),
        (Button.A, 8, 24, 2),
        (Button.A, 8, 180, 1),
        (Button.A, 8, 120, 2),
        (Button.A, 8, 24, 1),
        (Button.B, 8, 24, 1),
        (Button.A, 8, 120, 1),
        (Button.A, 8, 24, 4),
        (None, 0, 90, 1),
        (Button.A, 8, 24, 1),
        (None, 0, 90, 1),
        (Button.DOWN, 48, 80, 1),
        (Button.DOWN, 16, 90, 1),
        (Button.LEFT, 48, 8, 1),
        (Button.DOWN, 32, 120, 1),
        (Button.A, 8, 24, 1),
        (Button.A, 8, 240, 1),
        (None, 0, 300, 1),
        (Button.A, 8, 120, 4),
        (Button.A, 8, 48, 12),
        (Button.A, 8, 120, 28),
        (Button.A, 8, 96, 7),
        (None, 0, 240, 1),
        (Button.A, 8, 24, 1),
        (Button.A, 8, 120, 1),
        (None, 0, 120, 1),
        (Button.A, 8, 160, 1),
        (None, 0, 160, 1),
    ]
    for button, hold, release, count in inputs:
        for _ in range(count):
            firered_backend.press([] if button is None else [button], hold=hold, release=release)
            current = reader.read(firered_backend)
            maps.add(current.location[:2])
            battle_states.add(current.in_battle)
            money_values.add(current.money)
            assert current.money in (3000, 3080)
            for field in (
                "badge_count",
                "trainers_defeated",
                "story_flags",
                "world_map_flags",
                "pokedex_owned",
                "pokedex_seen",
            ):
                assert getattr(current, field) >= getattr(previous, field), field
            previous = current
    assert maps == {(4, 1), (4, 0), (3, 0), (4, 3)}
    assert battle_states == {False, True}
    assert money_values == {3000, 3080}
    assert previous.money == 3080
    assert previous.trainers_defeated == 1
    assert previous.story_flags == 2
    assert not previous.in_battle
