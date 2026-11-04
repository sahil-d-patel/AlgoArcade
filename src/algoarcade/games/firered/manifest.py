"""FireRed start states and segments. Contract: docs/harness-spec.md, Start states and segments.

The files live in data/states/ as <name>.state. scripts/capture_states.py regenerates them from
power-on. Check yours against the hashes with:

    uv run python -m algoarcade.games.firered.manifest            # data/states/
    uv run python -m algoarcade.games.firered.manifest DIR        # e.g. a capture_states --out
"""

import hashlib
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from algoarcade.games.firered.reader import FireRedState

# Captured on FireRed (USA) rev 1. A rev 1 state on a rev 0 ROM has not been tried.
ROM_SHA1 = "dd5945db9b930750cb39d00c84da8571feebf417"
STATES = Path(__file__).resolve().parents[4] / "data" / "states"


@dataclass(frozen=True)
class StartState:
    name: str
    sha256: str
    goal: Callable[[FireRedState], bool]
    # None means the run sets it, as the spec says for S0, S1 and the gym states.
    step_budget: int | None


def no_goal(_state: FireRedState) -> bool:
    """S0 and S1 run for the furthest milestone, so the episode ends only on its step budget."""
    return False


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
    # TODO(#66): also require in_battle to be False once ENV-08 reads it.
    return state.location[:2] == (4, 3) and bool(state.party) and state.party[0].level >= 6


def entered_viridian(state: FireRedState) -> bool:
    return state.location[:2] == (3, 1)  # MAP_VIRIDIAN_CITY


def left_forest_north(state: FireRedState) -> bool:
    return state.location[:2] == (15, 3)  # MAP_ROUTE2_VIRIDIAN_FOREST_NORTH_ENTRANCE


def boulder_badge(state: FireRedState) -> bool:
    return bool(state.badge_flags & 1)


# Map ids are (group, number) from data/maps/map_groups.json in pret/pokefirered.
MANIFEST = (
    StartState(
        "s0", "1182e804a325282810cb810dfb6a354b199ca56f0b90f99cf08c049ff5290bbf", no_goal, None
    ),
    StartState(
        "s1", "ef6d2693b4d567092a3c89bbff5d2df1b8f8103e8124fb2e9d4e13221a56d2b2", no_goal, None
    ),
    # The same bytes as s0.
    StartState(
        "seg_01_house",
        "1182e804a325282810cb810dfb6a354b199ca56f0b90f99cf08c049ff5290bbf",
        left_house,
        2048,
    ),
    # Outside Oak's Lab after the rival battle. Before the player has a Pokemon, Oak stops him at
    # the edge of town and Route 1 cannot be entered.
    StartState(
        "seg_02_pallet",
        "f9b44a259496b85647ef3f6641f9684983b21aa691151a5f6bb4f027d0cc1001",
        entered_route1,
        4096,
    ),
    StartState(
        "seg_03_rival",
        "d64505bd62dfc764abe982b6dc9c0f174ca84195613e87539d92861fb00cd145",
        won_rival_battle,
        3072,
    ),
    # The same bytes as seg_02_pallet's goal state: the first frame on Route 1.
    StartState(
        "seg_04_route1",
        "cfeb7e5396c4bc8635cd68c53a4c2cb20c47c4ad6a38e11ae654ba5ca22c5700",
        entered_viridian,
        6144,
    ),
    StartState(
        "seg_05_forest",
        "77cb5f4db82e77a66b50a76e4371b24f154ed0e50eb8bcbd091835368264f814",
        left_forest_north,
        8192,
    ),
    # gym_door and brock carry a level 14 Squirtle with Water Gun, ground on Route 1 before the
    # forest. Pressing A every turn wins the Brock battle, so label results from these states.
    StartState(
        "gym_door",
        "ed1ff580b2b01aaaebd7407bec6843dbd714e85e55ad3398874c2501f2f9a318",
        boulder_badge,
        None,
    ),
    StartState(
        "brock",
        "74c3d3b78fc610989906f8de9d64c4443ce5deb5cea58845976fa444c2c3c58f",
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
        elif hashlib.sha256(path.read_bytes()).hexdigest() != entry.sha256:
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
