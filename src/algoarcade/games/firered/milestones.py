"""The ordered Pallet-to-Brock milestones. Contract: docs/harness-spec.md, FireRed GameState fields.

`count` is the position, 1 to 9, of the furthest milestone that holds, or 0, so a lost rival
battle does not hold back the rest. forest_exited and gym_entered stop holding when the player
walks off their map; AlgoArcadeEnv keeps each episode's best count, which latches them. Flag and
item ids are in addresses.IDS, map ids are (group, number) from data/maps/map_groups.json, and
scripts are under data/maps/, all in pret/pokefirered at addresses.LAYOUT_COMMIT.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from algoarcade.games.firered.reader import FireRedState

# The flags the reader puts in FireRedState.event_flags.
FLAGS = (
    "FLAG_SYS_POKEMON_GET",
    "FLAG_BEAT_RIVAL_IN_OAKS_LAB",
    "FLAG_SYS_POKEDEX_GET",
    "FLAG_WORLD_MAP_PEWTER_CITY",
    "FLAG_DEFEATED_BROCK",
    "FLAG_BADGE01_GET",
)

MILESTONES: dict[str, Callable[["FireRedState"], bool]] = {
    "starter_chosen": lambda s: "FLAG_SYS_POKEMON_GET" in s.event_flags,
    # A win, not just the end of the battle: CB2_EndTrainerBattle (src/battle_setup.c) sets the
    # trainer flag after a loss too, and PalletTown_ProfessorOaksLab/scripts.inc then sets
    # FLAG_BEAT_RIVAL_IN_OAKS_LAB either way. The level 5 starter reaches 6 only from the
    # experience for knocking out the rival's level 5 Pokemon. A loss followed by training to
    # level 6 on Route 1 also passes.
    "rival_beaten": lambda s: (
        "FLAG_BEAT_RIVAL_IN_OAKS_LAB" in s.event_flags and any(p.level >= 6 for p in s.party)
    ),
    "parcel_delivered": lambda s: s.parcel_delivered,
    "pokedex_received": lambda s: "FLAG_SYS_POKEDEX_GET" in s.event_flags,
    # MAP_ROUTE2_VIRIDIAN_FOREST_NORTH_ENTRANCE, the seg_05_forest goal.
    "forest_exited": lambda s: s.location[:2] == (15, 3),
    # PewterCity/scripts.inc sets it on entering Pewter City, and nothing clears it.
    "pewter_reached": lambda s: "FLAG_WORLD_MAP_PEWTER_CITY" in s.event_flags,
    # MAP_PEWTER_CITY_GYM.
    "gym_entered": lambda s: s.location[:2] == (6, 2),
    "brock_defeated": lambda s: "FLAG_DEFEATED_BROCK" in s.event_flags,
    "boulder_badge": lambda s: "FLAG_BADGE01_GET" in s.event_flags,
}


def count(state: "FireRedState") -> int:
    reached = [i for i, holds in enumerate(MILESTONES.values(), start=1) if holds(state)]
    return max(reached, default=0)
