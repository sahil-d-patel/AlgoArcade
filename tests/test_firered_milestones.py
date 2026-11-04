import hashlib
from dataclasses import replace
from pathlib import Path

import pytest
from capture_states import Player
from capture_validation_states import play

from algoarcade.backends.base import Button
from algoarcade.backends.mgba import MgbaBackend
from algoarcade.envs.env import AlgoArcadeEnv
from algoarcade.games.firered.features import FEATURES
from algoarcade.games.firered.manifest import MANIFEST, ROM_SHA1
from algoarcade.games.firered.milestones import MILESTONES, count
from algoarcade.games.firered.reader import FireRedReader, FireRedState, PartyPokemon
from algoarcade.rewards.registry import COMPONENTS, DEFAULT_WEIGHTS, RewardRegistry

ROOT = Path(__file__).resolve().parents[1]
STARTER = PartyPokemon(level=5, hp=20, max_hp=20, status=0, species=7, moves=(33, 39, 0, 0))
BLANK = FireRedState(
    map_group=4,
    map_num=3,
    x=6,
    y=8,
    party=(STARTER,),
    money=3000,
    badge_flags=0,
    trainers_defeated=0,
    pokedex_owned=0,
    pokedex_seen=0,
    in_battle=False,
    text_active=False,
    text_progress=0,
    script_active=False,
    controls_locked=False,
    story_flags=0,
    world_map_flags=0,
    text_speed=2,
    battle_scene_off=True,
    battle_own=None,
    battle_enemy=None,
    event_flags=frozenset(),
    parcel_delivered=False,
)
# Manifest states, goal states from scripts/capture_states.py and two from
# scripts/capture_validation_states.py.
HASHES = {entry.name: entry.sha256 for entry in MANIFEST} | {
    "seg_03_rival_goal": "c7d822061aa8fbf0b1922ff8a6ddb88714aa6f7364b86ff6c8b37b5be046ea2b",
    "seg_05_forest_goal": "6a4908eb8f19ef2de0b79ebb1a49906b03734c17c7a3cb8d9f22ce52ea0998f2",
    "brock_goal": "1c789cc028a8f7d3ed2b7aa4a52f920ba253f86b8d303ef3b12a5684551f0f0a",
    "validate_parcel": "2695da9e2b59db8e2ea70481b8f39da74a990c765631eb8f4029c09998fb98e4",
    "validate_pokedex": "e6335f08f060340552b23d7eab9f4427567df6334184a324a78a503bec0672d3",
}
RIVAL_DONE = replace(BLANK, event_flags=frozenset({"FLAG_BEAT_RIVAL_IN_OAKS_LAB"}))


def flags(*names: str) -> frozenset[str]:
    return frozenset(names)


# Before and after for each milestone, in the order of the list.
CASES = {
    "starter_chosen": (BLANK, replace(BLANK, event_flags=flags("FLAG_SYS_POKEMON_GET"))),
    # The battle is over either way; only a win levels the starter to 6.
    "rival_beaten": (RIVAL_DONE, replace(RIVAL_DONE, party=(replace(STARTER, level=6),))),
    "parcel_delivered": (BLANK, replace(BLANK, parcel_delivered=True)),
    "pokedex_received": (BLANK, replace(BLANK, event_flags=flags("FLAG_SYS_POKEDEX_GET"))),
    "forest_exited": (
        replace(BLANK, map_group=1, map_num=0, x=29, y=62),
        replace(BLANK, map_group=15, map_num=3, x=7, y=10),
    ),
    "pewter_reached": (BLANK, replace(BLANK, event_flags=flags("FLAG_WORLD_MAP_PEWTER_CITY"))),
    "gym_entered": (
        replace(BLANK, map_group=3, map_num=2, x=15, y=17),
        replace(BLANK, map_group=6, map_num=2, x=6, y=14),
    ),
    "brock_defeated": (BLANK, replace(BLANK, event_flags=flags("FLAG_DEFEATED_BROCK"))),
    "boulder_badge": (BLANK, replace(BLANK, event_flags=flags("FLAG_BADGE01_GET"))),
}


def test_milestones_are_the_nine_in_order() -> None:
    assert list(MILESTONES) == list(CASES)


@pytest.mark.parametrize("name", CASES)
def test_milestone_false_before_true_after(name: str) -> None:
    before, after = CASES[name]
    assert not MILESTONES[name](before)
    assert MILESTONES[name](after)
    assert count(after) == list(MILESTONES).index(name) + 1


def test_count_is_the_furthest_milestone_reached() -> None:
    assert count(BLANK) == 0
    lost_then_parcel = replace(
        RIVAL_DONE,
        event_flags=flags("FLAG_SYS_POKEMON_GET", "FLAG_BEAT_RIVAL_IN_OAKS_LAB"),
        parcel_delivered=True,
    )
    assert not MILESTONES["rival_beaten"](lost_then_parcel)
    assert lost_then_parcel.milestone_count == 3


def start(name: str) -> MgbaBackend:
    rom = ROOT / "data/roms/firered.gba"
    path = ROOT / "data/states" / f"{name}.state"
    if not rom.is_file() or not path.is_file():
        pytest.skip(f"Needs the local FireRed ROM and {path.name}")
    backend = MgbaBackend(rom, render=False)
    if backend.rom.sha1 != ROM_SHA1:
        backend.close()
        pytest.skip("The start states were captured on rev 1")
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == HASHES[name]
    backend.load_state(data)
    return backend


def reached(backend: MgbaBackend) -> list[str]:
    state = FireRedReader(backend.rom.sha1).read(backend)
    return [name for name, holds in MILESTONES.items() if holds(state)]


@pytest.mark.rom
@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("s0", []),
        # After the parcel: Oak has also handed over the Pokedex, so four hold.
        ("s1", ["starter_chosen", "rival_beaten", "parcel_delivered", "pokedex_received"]),
    ],
)
def test_milestones_on_s0_and_s1(name: str, expected: list[str]) -> None:
    backend = start(name)
    try:
        assert reached(backend) == expected
    finally:
        backend.close()


# Start states and ENV-08 captures along the route. validate_parcel holds the parcel in the
# Viridian Mart; validate_pokedex is the frame after Oak's Pokedex scene.
@pytest.mark.rom
@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("seg_03_rival", 1),
        ("seg_03_rival_goal", 2),
        ("validate_parcel", 2),
        ("validate_pokedex", 4),
        ("seg_05_forest", 4),
        ("seg_05_forest_goal", 5),
        ("gym_door", 6),
        ("brock", 7),
        ("brock_goal", 9),
    ],
)
def test_milestone_count_on_real_states(name: str, expected: int) -> None:
    backend = start(name)
    try:
        assert FireRedReader(backend.rom.sha1).read(backend).milestone_count == expected
    finally:
        backend.close()


@pytest.mark.rom
def test_pewter_reached_stays_true_on_route_2() -> None:
    backend = start("gym_door")
    try:
        assert "pewter_reached" in reached(backend)
        # South through Pewter City and onto Route 2.
        Player(backend).run("tLx7 tUx5 tRx14 tDx14 tLx5 tRx3 tDx3 tR tDx10 tD tL tDx2")
        assert FireRedReader(backend.rom.sha1).read(backend).location == (3, 20, 8, 2)
        assert reached(backend)[-1] == "pewter_reached"
    finally:
        backend.close()


@pytest.mark.rom
def test_rival_beaten_stays_false_after_a_loss() -> None:
    backend = start("seg_03_rival")
    try:
        player = Player(backend)
        player.run("tDx2 tLx3")
        backend.press([Button.DOWN], hold=20, release=20)
        player.run("ab")
        play(backend, "w60 B*60 w200")
        # Tail Whip every turn until Squirtle faints, then through the rival's lines.
        while player.reader.read(backend).in_battle:
            play(backend, "A w20 RIGHT w10 A w300 B*10 w60")
        player.run("bb")
        state = FireRedReader(backend.rom.sha1).read(backend)
        assert "FLAG_BEAT_RIVAL_IN_OAKS_LAB" in state.event_flags
        assert state.trainers_defeated == 1
        assert reached(backend) == ["starter_chosen"]
    finally:
        backend.close()


@pytest.mark.rom
def test_env_keeps_gym_entered_after_leaving_the_gym() -> None:
    backend = start("gym_door")
    env = AlgoArcadeEnv(
        backend,
        FireRedReader(backend.rom.sha1),
        RewardRegistry(COMPONENTS, DEFAULT_WEIGHTS),
        backend.save_state(),
        100,
        FEATURES,
    )
    try:
        _, info = env.reset(seed=0)
        counts, maps, paid = [info["milestone_count"]], [], 0.0
        for action in [0] * 4 + [1] * 6:
            _, _, _, _, info = env.step(action)
            counts.append(info["milestone_count"])
            maps.append(info["location"][:2])
            paid += info["reward/milestone"]
        assert (6, 2) in maps
        assert maps[-1] == (3, 2)
        assert counts == [6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7]
        assert paid == 1.0
    finally:
        env.close()
