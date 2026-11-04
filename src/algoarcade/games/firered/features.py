"""The observation vector AlgoArcadeEnv builds from a FireRedState, each value scaled into [0, 1].

Positions divide by 255, a guess at an upper bound on map ids and map sides; the env clips
anything above. The party bounds are PARTY_SIZE and MAX_LEVEL, the same as the level term in
algoarcade.rewards.registry.
"""

from collections.abc import Callable

from algoarcade.games.firered.reader import FireRedState

FEATURES: dict[str, Callable[[FireRedState], float]] = {
    "map_group": lambda s: s.map_group / 255,
    "map_num": lambda s: s.map_num / 255,
    "x": lambda s: s.x / 255,
    "y": lambda s: s.y / 255,
    "party_count": lambda s: s.party_count / 6,
    "party_level_sum": lambda s: sum(p.level for p in s.party) / 600,
    "party_hp_frac": lambda s: sum(p.hp for p in s.party) / max(1, sum(p.max_hp for p in s.party)),
}
