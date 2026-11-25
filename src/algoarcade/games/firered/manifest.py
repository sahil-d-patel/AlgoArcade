"""FireRed start states and segments. Contract: docs/harness-spec.md, Start states and segments.

The files live in data/states/ as <name>.state. scripts/capture_states.py regenerates them from
power-on. Check yours against the hashes with:

    uv run python -m algoarcade.games.firered.manifest            # data/states/
    uv run python -m algoarcade.games.firered.manifest DIR        # e.g. a capture_states --out
"""

import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from algoarcade.backends.mgba import state_sha256
from algoarcade.games.firered.reader import FireRedState

# Captured on FireRed (USA) rev 1. A rev 1 state on a rev 0 ROM has not been tried.
ROM_SHA1 = "dd5945db9b930750cb39d00c84da8571feebf417"
STATES = Path(__file__).resolve().parents[4] / "data" / "states"


@dataclass(frozen=True)
class StartState:
    name: str
    sha256: str  # state_sha256 of the file, not the file's own SHA-256
    goal: Callable[[FireRedState], bool]
    # None means the run sets it, as the spec says for the gym states.
    step_budget: int | None


def left_house(state: FireRedState) -> bool:
    return state.location[:2] == (3, 0)  # MAP_PALLET_TOWN


def entered_route1(state: FireRedState) -> bool:
    return state.location[:2] == (3, 19)  # MAP_ROUTE1


def won_rival_battle(state: FireRedState) -> bool:
    # No flag tells a win from a loss here: CB2_EndTrainerBattle sets the trainer flag on both
    # outcomes of TRAINER_BATTLE_EARLY_RIVAL, and EndRivalBattle sets FLAG_BEAT_RIVAL_IN_OAKS_LAB
    # unconditionally. From seg_03_rival, Squirtle is level 5 and reaches 6 only from the
    # experience for knocking out the rival's Bulbasaur, and a loss gives none. The map check
    # keeps the goal inside the lab (MAP_PALLET_TOWN_PROFESSOR_OAKS_LAB).
    return (
        state.location[:2] == (4, 3)
        and bool(state.party)
        and state.party[0].level >= 6
        and not state.in_battle
    )


def entered_viridian(state: FireRedState) -> bool:
    return state.location[:2] == (3, 1)  # MAP_VIRIDIAN_CITY


def left_forest_north(state: FireRedState) -> bool:
    return state.location[:2] == (15, 3)  # MAP_ROUTE2_VIRIDIAN_FOREST_NORTH_ENTRANCE


def boulder_badge(state: FireRedState) -> bool:
    """FLAG_BADGE01_GET (0x820), the last milestone."""
    return bool(state.badge_flags & 1)


# Map ids are (group, number) from data/maps/map_groups.json in pret/pokefirered.
#
# s0's and s1's budgets are 11.2 and 11.3 times the scripted playthrough's decisions to
# brock_goal (docs/harness-spec.md, Start states and segments).
MANIFEST = (
    StartState(
        "s0",
        "614363f51439fdf9212ed231a75c453e91a30d280aa22f28095ef24aa0cc35f2",
        boulder_badge,
        87_040,
    ),
    StartState(
        "s1",
        "3d11fca855f9f4a4fad869d30bdfe15a42cd05a912c2431415d51e5e871467cb",
        boulder_badge,
        71_680,
    ),
    # The same bytes as s0.
    StartState(
        "seg_01_house",
        "614363f51439fdf9212ed231a75c453e91a30d280aa22f28095ef24aa0cc35f2",
        left_house,
        2048,
    ),
    # Outside Oak's Lab after the rival battle. Before the player has a Pokemon, Oak stops him at
    # the edge of town and Route 1 cannot be entered.
    StartState(
        "seg_02_pallet",
        "527591a12ea5fc6358fb308ad3cf15c7df8b9c02a7a6117b1ac8ee65cc03c281",
        entered_route1,
        4096,
    ),
    StartState(
        "seg_03_rival",
        "74bc5ffb5fa74bcc4c7dcf1a61daa178d85355f5fa6e5ca226b4246fd37a529c",
        won_rival_battle,
        3072,
    ),
    # The same bytes as seg_02_pallet's goal state: the first frame on Route 1.
    StartState(
        "seg_04_route1",
        "41ec4c32d3e8dc14be6a4ff29f82ab9be34c379f9e77579126fbb22a2e2a9b90",
        entered_viridian,
        6144,
    ),
    StartState(
        "seg_05_forest",
        "32077774dcc94f1f077a59b72899e699c6bde0a37f6c7c0526bf7b997aedbe63",
        left_forest_north,
        8192,
    ),
    # gym_door and brock carry a level 14 Squirtle with Water Gun, ground on Route 1 before the
    # forest. Pressing A every turn wins the Brock battle, so label results from these states.
    StartState(
        "gym_door",
        "f6e79eacde1a2cbba76db8b5ce27ca125a411cbd644fa8f797fe2a660ba3e4a8",
        boulder_badge,
        None,
    ),
    StartState(
        "brock",
        "270b2832bbb1ddd55b7bb81042f75da4871a1f7249ab41d234263f5cd829732c",
        boulder_badge,
        None,
    ),
)


def check(directory: Path = STATES) -> list[str]:
    lines = []
    for entry in MANIFEST:
        path = directory / f"{entry.name}.state"
        if not path.is_file():
            lines.append(f"MISSING   {entry.name}  {path}")
        elif state_sha256(path.read_bytes()) != entry.sha256:
            lines.append(f"MISMATCH  {entry.name}  {path}")
        else:
            lines.append(f"OK        {entry.name}")
    return lines


def main() -> None:
    lines = check(Path(sys.argv[1]) if len(sys.argv) > 1 else STATES)
    print("\n".join(lines))  # noqa: T201
    if not all(line.startswith("OK") for line in lines):
        sys.exit(1)


if __name__ == "__main__":
    main()
