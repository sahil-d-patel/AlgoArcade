import hashlib
from collections.abc import Iterator
from pathlib import Path

import pytest
from capture_validation_states import SEGMENTS, play

from algoarcade.backends.base import Button
from algoarcade.backends.mgba import MgbaBackend
from algoarcade.games.firered.reader import FireRedReader, FireRedState

ROOT = Path(__file__).resolve().parents[1]
REV0 = "41cb23d8dccc8ebd7c649cd8fbb58eeace6e2fdc"
REV1 = "dd5945db9b930750cb39d00c84da8571feebf417"
# Each capture runs only on the revision it was made with. Rev 0: Kumario1's captures. Rev 1:
# scripts/capture_validation_states.py, and brock_goal from scripts/capture_states.py (#69).
CAPTURES = {
    "env08-pokedex.state": (
        REV0,
        "11b891fdc87972644c45198728edee6bc4d4f269d2713fb2881dd084c8bbe3bd",
    ),
    "env08-money-3000.state": (
        REV0,
        "37ad9fefd6813f5f68362247ee276d6205dd2b4a0823403c3a69f41b325ae093",
    ),
    "env08-money-3080.state": (
        REV0,
        "fc957ea809f0b2f687755a18ba3ca038a5184c6850ef79f433b01947fd18d7c8",
    ),
    "env08-before-starter.state": (
        REV0,
        "ff47f57f7e3071786053af11d709be276c8a56cef5de0954c27e1cb0d6240fb5",
    ),
    "env08-starter-overworld.state": (
        REV0,
        "a8f139228c9a17a798fe5b69caa18fe3dbc1b7ab116d385116288a452c015c9b",
    ),
    "env08-party.state": (
        REV0,
        "43fd78f7ed377c79fa03c60e8573953347562edbf9d80bc958b60dbef5365dc4",
    ),
    "env08-moves.state": (
        REV0,
        "f5cc22d91353dba7ccc873a8634df9b00bc59909f1cac201eb0c18027d6f7ac5",
    ),
    "env08-party-hp13.state": (
        REV0,
        "3f0b4400db4d35ba73f19af7e5beaa575f8b63c3ece88102b240540fba20cf78",
    ),
    "env08-after-rival.state": (
        REV0,
        "78b86e532f6174576198098cdbd83315f36a4d9051cf46f05d983f971d51df9a",
    ),
    "env08-rival-battle.state": (
        REV0,
        "ec0e4d1ab834b444fad12e263bb9327648c1af9d32d8820228af8484bedfb3e6",
    ),
    "env08-oak-dialogue.state": (
        REV0,
        "b54328c9856111da3cdb48f6c299d8dac4f1ef76c2f84ed47587d070677d80af",
    ),
    "env08-warp-before.state": (
        REV0,
        "e8e6c7c190f299eb72ea1feaf41d42c173897c4fab28ad6dc67a567530191e3b",
    ),
    "env08-warp-after.state": (
        REV0,
        "3a8f869e05afffafa706ec9da67a7e8fb13b33db21bff5d8961bae8f4b68538e",
    ),
    "env08-options-default.state": (
        REV0,
        "da916d488fb064ad0e0bc3c837fea670f73f4684bcc94504a307e8a9bdbf4e97",
    ),
    "env08-options-fast-off.state": (
        REV0,
        "ae91101713f18de3d2202d44d3a9dc104123c8aaec840937b88d77ee46f2f4e0",
    ),
    "validate_bedroom_default.state": (
        REV1,
        "8dae7b01d5bac2f066fb06903547a1d941ee12935e48e2bede1249f8493a6a67",
    ),
    "validate_bedroom.state": (
        REV1,
        "ad03ebdf5ec4a93f306d76896f636df5eebb7639d85cbf86570845dbcfc43b1d",
    ),
    "validate_starter.state": (
        REV1,
        "4efa9f1a6dc6ea522ae72062c30a60f776be578235f43a6a5fa0a8b70e5f295c",
    ),
    "validate_rival_text.state": (
        REV1,
        "5347edf9a63c170cfe341b42a3b3fa164ba08bc8f1e385e7e38849fa095e4473",
    ),
    "validate_battle.state": (
        REV1,
        "f1cf692066eb5bb93742af5adbf8c2a1367c88b5d4573b37a8d4f33474865796",
    ),
    "validate_after_battle.state": (
        REV1,
        "678a504168ce0c52738ae5ca62d193d7161b8a9120b802269eba5d116c17c59f",
    ),
    "validate_pallet.state": (
        REV1,
        "ffe14ce0f7e7909433a1ee9851d31d959a24f49f91770409c155cc34fd8abe76",
    ),
    "validate_viridian.state": (
        REV1,
        "65f121b24f6762057d679d08322ad3f22434f8338c08e6127c47be7640ad8f2d",
    ),
    "validate_pokedex.state": (
        REV1,
        "e6335f08f060340552b23d7eab9f4427567df6334184a324a78a503bec0672d3",
    ),
    "validate_poisoned.state": (
        REV1,
        "07541dde2d66f665d4a22d34a33c9986f3575c1b580fbe8a83bfcd4bb89f2cd3",
    ),
    "brock_goal.state": (
        REV1,
        "1c789cc028a8f7d3ed2b7aa4a52f920ba253f86b8d303ef3b12a5684551f0f0a",
    ),
}


@pytest.fixture
def firered_backend() -> Iterator[MgbaBackend]:
    rom = ROOT / "data/roms/firered.gba"
    if not rom.is_file():
        pytest.skip("Needs the local FireRed ROM")
    backend = MgbaBackend(rom)
    try:
        yield backend
    finally:
        backend.close()


def load_capture(backend: MgbaBackend, filename: str) -> FireRedState:
    revision, sha256 = CAPTURES[filename]
    if backend.rom.sha1 != revision:
        pytest.skip(f"{filename} was captured on the other FireRed revision")
    path = ROOT / "data/states" / filename
    if not path.is_file():
        pytest.skip(f"Needs local capture {filename}")
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == sha256
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
        ("env08-warp-before.state", "story_flags", 0),
        ("env08-after-rival.state", "story_flags", 3),
        ("env08-money-3000.state", "world_map_flags", 0),
        ("env08-after-rival.state", "world_map_flags", 1),
        ("env08-before-starter.state", "pokedex_owned", 0),
        ("env08-party.state", "pokedex_owned", 1),
        ("env08-party.state", "pokedex_seen", 1),
        ("env08-rival-battle.state", "pokedex_seen", 2),
        ("env08-pokedex.state", "pokedex_seen", 4),
        ("env08-pokedex.state", "pokedex_owned", 1),
        ("env08-pokedex.state", "story_flags", 4),
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
        # Rev 1. Trainer card: money and badge slots. Map name popups and the new-game spawn
        # (src/new_game.c) and lab door (PalletTown/map.json) for location. Party screen: level
        # and HP. Pokedex contents: Seen and Owned. Story events, each seen on screen: entering
        # the lab, "A received the SQUIRTLE", the rival battle, the Pokedex, Brock's badge and
        # TM39 in the TM case. Battle screen: "A got 80 for winning!" for trainers_defeated.
        ("validate_bedroom_default.state", "money", 3000),
        ("validate_after_battle.state", "money", 3080),
        ("brock_goal.state", "money", 4588),
        ("validate_pokedex.state", "badge_flags", 0),
        ("brock_goal.state", "badge_flags", 1),
        ("validate_pokedex.state", "badge_count", 0),
        ("brock_goal.state", "badge_count", 1),
        ("validate_bedroom.state", "location", (4, 1, 6, 6)),
        ("validate_pallet.state", "location", (3, 0, 16, 14)),
        ("validate_viridian.state", "location", (3, 1, 25, 39)),
        ("validate_bedroom.state", "party_count", 0),
        ("validate_starter.state", "party_count", 1),
        ("validate_starter.state", "party_level_sum", 5),
        ("validate_after_battle.state", "party_level_sum", 6),
        ("brock_goal.state", "party_level_sum", 15),
        ("validate_starter.state", "party_hp_frac", 1.0),
        ("validate_poisoned.state", "party_hp_frac", 30 / 37),
        ("validate_battle.state", "trainers_defeated", 0),
        ("validate_after_battle.state", "trainers_defeated", 1),
        ("validate_bedroom.state", "story_flags", 0),
        ("validate_starter.state", "story_flags", 2),
        ("validate_after_battle.state", "story_flags", 3),
        ("validate_pokedex.state", "story_flags", 4),
        ("brock_goal.state", "story_flags", 6),
        ("validate_bedroom.state", "world_map_flags", 0),
        ("validate_pallet.state", "world_map_flags", 1),
        ("validate_viridian.state", "world_map_flags", 2),
        ("validate_bedroom.state", "pokedex_owned", 0),
        ("validate_pokedex.state", "pokedex_owned", 1),
        ("validate_bedroom.state", "pokedex_seen", 0),
        ("validate_pokedex.state", "pokedex_seen", 4),
        ("validate_starter.state", "in_battle", False),
        ("validate_battle.state", "in_battle", True),
        ("validate_starter.state", "text_active", False),
        ("validate_rival_text.state", "text_active", True),
        ("validate_starter.state", "script_active", False),
        ("validate_rival_text.state", "script_active", True),
        ("validate_starter.state", "controls_locked", False),
        ("validate_rival_text.state", "controls_locked", True),
        ("validate_bedroom_default.state", "text_speed", 1),
        ("validate_bedroom.state", "text_speed", 2),
        ("validate_bedroom_default.state", "battle_scene_off", False),
        ("validate_bedroom.state", "battle_scene_off", True),
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
        ("validate_starter.state", "level", 5),
        ("brock_goal.state", "level", 15),
        ("validate_starter.state", "hp", 19),
        ("validate_poisoned.state", "hp", 30),
        ("validate_starter.state", "max_hp", 19),
        ("brock_goal.state", "max_hp", 42),
        ("validate_starter.state", "status", 0),
        ("validate_poisoned.state", "status", 8),
        ("validate_starter.state", "species", 7),
        ("validate_starter.state", "moves", (33, 39, 0, 0)),
        ("brock_goal.state", "moves", (55, 39, 145, 110)),
    ],
)
def test_firered_fields_party(
    firered_backend: MgbaBackend, filename: str, field: str, expected: object
) -> None:
    # Party and summary screens: SQUIRTLE (7) with Tackle (33) and Tail Whip (39); at Brock,
    # Water Gun (55), Tail Whip, Bubble (145) and Withdraw (110). Status 8 is STATUS1_POISON
    # (include/constants/pokemon.h); the party screen shows PSN.
    assert getattr(load_capture(firered_backend, filename).party[0], field) == expected


@pytest.mark.rom
@pytest.mark.parametrize(
    ("filename", "valid"),
    [
        ("env08-options-default.state", False),
        ("env08-options-fast-off.state", True),
        ("validate_bedroom_default.state", False),
        ("validate_bedroom.state", True),
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


PROGRESS = (
    "badge_count",
    "trainers_defeated",
    "story_flags",
    "world_map_flags",
    "pokedex_owned",
    "pokedex_seen",
)


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
            for field in PROGRESS:
                assert getattr(current, field) >= getattr(previous, field), field
            previous = current
    assert maps == {(4, 1), (4, 0), (3, 0), (4, 3)}
    assert battle_states == {False, True}
    assert money_values == {3000, 3080}
    assert previous.money == 3080
    assert previous.trainers_defeated == 1
    assert previous.story_flags == 3
    assert not previous.in_battle


@pytest.mark.rom
def test_firered_fields_warp_and_battle_trace_rev1(firered_backend: MgbaBackend) -> None:
    previous = load_capture(firered_backend, "validate_starter.state")
    reader = FireRedReader(firered_backend.rom.sha1)
    segments = dict(SEGMENTS)
    maps, battle_reads = set(), 0
    # From the starter through the rival battle and out of the lab door into Pallet Town.
    for name in ("rival_text", "battle", "after_battle", "pallet"):
        for token in segments[name].split():
            button, _, repeat = token.partition("*")
            for _ in range(1 if token.startswith("w") else int(repeat or 1)):
                play(firered_backend, token if token.startswith("w") else button)
                current = reader.read(firered_backend)
                maps.add(current.location[:2])
                battle_reads += current.in_battle
                assert current.money in (3000, 3080)
                assert current.money >= previous.money
                for field in PROGRESS:
                    assert getattr(current, field) >= getattr(previous, field), field
                previous = current
    assert maps == {(4, 3), (3, 0)}
    assert battle_reads > 0
    assert previous.location == (3, 0, 16, 14)
    assert (previous.money, previous.trainers_defeated, previous.story_flags) == (3080, 1, 3)


@pytest.mark.rom
def test_firered_fields_text_progress_field(firered_backend: MgbaBackend) -> None:
    # "GREEN: Wait, A! Let's check out our POKeMON!" waits at its arrow. A prints the next page,
    # "Come on, I'll take you on!", one character a frame; the box stays up once it is done.
    waiting = load_capture(firered_backend, "validate_rival_text.state")
    reader = FireRedReader(firered_backend.rom.sha1)
    firered_backend.step(30)
    assert reader.read(firered_backend).text_progress == waiting.text_progress
    firered_backend.press([Button.A], hold=8, release=0)
    printing = []
    for _ in range(12):
        firered_backend.step(1)
        printing.append(reader.read(firered_backend))
    assert all(state.text_active for state in printing)
    progress = [waiting.text_progress] + [state.text_progress for state in printing]
    assert progress == sorted(set(progress))
    firered_backend.step(30)
    done = reader.read(firered_backend)
    assert (done.text_active, done.script_active, done.controls_locked) == (False, True, True)


@pytest.mark.rom
def test_firered_fields_text_progress_battle(firered_backend: MgbaBackend) -> None:
    # "RIVAL GREEN would like to battle!", then "RIVAL GREEN sent out BULBASAUR!" prints into the
    # same window. The field message box is not used in battle, so text_active stays False.
    first = load_capture(firered_backend, "validate_battle.state")
    reader = FireRedReader(firered_backend.rom.sha1)
    firered_backend.step(100)
    firered_backend.press([Button.A], hold=8, release=0)
    second = []
    for _ in range(12):
        firered_backend.step(1)
        second.append(reader.read(firered_backend))
    assert all(state.in_battle and not state.text_active for state in second)
    progress = [state.text_progress for state in second]
    assert progress == sorted(set(progress))
    assert first.text_progress not in progress
