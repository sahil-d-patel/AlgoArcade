"""Builds reward components from a config of name to weight. Contract: docs/harness-spec.md."""

from collections.abc import Callable, Mapping

from algoarcade.games.base import GameState
from algoarcade.rewards.base import RewardComponent
from algoarcade.rewards.explore import Explore
from algoarcade.rewards.progress import NewMaximum, Progress

# FireRed's components. Another game passes its own table, so its state needs only the
# fields its components read.
#
# The bounds come from pret/pokefirered at ce6406d8: FLAGS_COUNT and NUM_BADGES in
# include/constants/flags.h, PARTY_SIZE in include/constants/global.h, MAX_LEVEL in
# include/constants/pokemon.h.
# - events: at most FLAGS_COUNT (0x900) until ENV-08 sets the whitelist length. No rise bound,
#   because one script can set several flags.
# - level: at most PARTY_SIZE * MAX_LEVEL. One Pokemon joins the party per step, so the sum
#   rises by at most MAX_LEVEL.
# - badge: at most NUM_BADGES. Each badge is set by its own gym leader's script, so a step
#   earns at most one. The rise bound is what rejects a misread 0xFF; its popcount is 8.
# - milestone: at most the nine milestones in ENV-13. No rise bound: the last two are
#   FLAG_DEFEATED_BROCK and FLAG_BADGE01_GET, which data/maps/PewterCity_Gym/scripts.inc sets
#   one after the other.
COMPONENTS: dict[str, Callable[[], RewardComponent[Progress]]] = {
    "events": lambda: NewMaximum("events", lambda s: s.story_flags, maximum=0x900),
    "explore": Explore,
    "level": lambda: NewMaximum(
        "level", lambda s: s.party_level_sum, maximum=600, max_rise=100, cap=1.0
    ),
    "badge": lambda: NewMaximum("badge", lambda s: s.badge_count, maximum=8, max_rise=1),
    "milestone": lambda: NewMaximum("milestone", lambda s: s.milestone_count, maximum=9),
}


class RewardRegistry[S: GameState]:
    def __init__(
        self,
        components: Mapping[str, Callable[[], RewardComponent[S]]],
        weights: Mapping[str, float],
    ) -> None:
        unknown = sorted(set(weights) - set(components))
        if unknown:
            raise ValueError(
                f"unknown reward component(s) {unknown}; valid names: {sorted(components)}"
            )
        self._weights = dict(weights)
        self._components = [components[name]() for name in weights]

    def reset(self, state: S) -> None:
        for component in self._components:
            component.reset(state)

    def step(self, prev: S, curr: S) -> tuple[float, dict[str, float]]:
        """The weighted reward, and each component's raw value as `reward/<name>`."""
        raw = {c.name: c.score(prev, curr) for c in self._components}
        reward = sum(self._weights[name] * value for name, value in raw.items())
        return reward, {f"reward/{name}": value for name, value in raw.items()}
