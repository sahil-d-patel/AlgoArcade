import hashlib
from pathlib import Path

import pytest
from capture_states import ROM, STAGES, Player, options_ok, play

from algoarcade.backends.mgba import MgbaBackend
from algoarcade.games.firered.manifest import MANIFEST, ROM_SHA1, STATES, StartState, check
from algoarcade.rom import identify

# docs/harness-spec.md, Start states and segments. None is "set per run".
BUDGETS = {
    "s0": None,
    "s1": None,
    "seg_01_house": 2048,
    "seg_02_pallet": 4096,
    "seg_03_rival": 3072,
    "seg_04_route1": 6144,
    "seg_05_forest": 8192,
    "gym_door": None,
    "brock": None,
}
GOAL_STATES = ["seg_01_house", "seg_02_pallet", "seg_03_rival", "seg_04_route1", "seg_05_forest"]


def test_state_manifest_has_nine_entries() -> None:
    assert {entry.name: entry.step_budget for entry in MANIFEST} == BUDGETS
    for entry in MANIFEST:
        assert len(entry.sha256) == 64
        int(entry.sha256, 16)
        assert callable(entry.goal)


def test_state_manifest_check_reports_missing_and_mismatched(tmp_path: Path) -> None:
    (tmp_path / "s0.state").write_bytes(b"not the captured state")
    lines = check(tmp_path)
    assert len(lines) == 9
    assert lines[0].startswith("MISMATCH  s0")
    assert all(line.startswith("MISSING") for line in lines[1:])


@pytest.fixture
def backend():
    if not ROM.is_file():
        pytest.skip("Place your FireRed ROM in data/roms/firered.gba")
    if identify(ROM).sha1 != ROM_SHA1:
        pytest.skip("The states were captured on rev 1")
    backend = MgbaBackend(ROM)
    yield backend
    backend.close()


def load(backend, name: str, sha256: str | None = None) -> None:
    path = STATES / f"{name}.state"
    if not path.is_file():
        pytest.skip(f"{path} is missing")
    data = path.read_bytes()
    if sha256 is not None:
        assert hashlib.sha256(data).hexdigest() == sha256, f"{path} differs from the manifest"
    backend.load_state(data)


@pytest.mark.rom
@pytest.mark.parametrize("entry", MANIFEST, ids=lambda entry: entry.name)
def test_state_manifest_goal_false_on_start(backend, entry: StartState) -> None:
    load(backend, entry.name, entry.sha256)
    assert options_ok(backend, ROM_SHA1)
    assert not entry.goal(Player(backend).reader.read(backend))


@pytest.mark.rom
@pytest.mark.parametrize(
    "entry", [entry for entry in MANIFEST if entry.name in GOAL_STATES], ids=lambda e: e.name
)
def test_state_manifest_goal_true_on_goal_state(backend, entry: StartState) -> None:
    load(backend, f"{entry.name}_goal")
    assert entry.goal(Player(backend).reader.read(backend))


@pytest.mark.rom
def test_state_manifest_badge_goal_true_after_brock(backend) -> None:
    load(backend, "brock_goal")
    state = Player(backend).reader.read(backend)
    assert state.badge_flags == 0x01
    assert MANIFEST[-1].goal(state)


@pytest.mark.rom
def test_state_manifest_options_check_fails_on_defaults(backend) -> None:
    backend.step(1800)  # the title screen: text speed mid, battle scene on
    assert not options_ok(backend, ROM_SHA1)


@pytest.mark.rom
def test_state_manifest_s0_replay_from_power_on(backend) -> None:
    stage = STAGES[0]
    assert stage.source is None
    state = play(backend, stage)
    assert Player(backend).location()[:2] == (4, 1)  # MAP_PALLET_TOWN_PLAYERS_HOUSE_2F
    assert options_ok(backend, ROM_SHA1)
    assert hashlib.sha256(state).hexdigest() == MANIFEST[0].sha256
