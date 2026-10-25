"""Builds reward components from a config of name to weight. Contract: docs/harness-spec.md."""

from collections.abc import Callable, Mapping

from algoarcade.rewards.base import RewardComponent
from algoarcade.rewards.explore import Explore
from algoarcade.rewards.progress import NewMaximum, Progress

COMPONENTS: dict[str, Callable[[], RewardComponent[Progress]]] = {
    "events": lambda: NewMaximum("events", lambda s: s.story_flags),
    "explore": Explore,
    "level": lambda: NewMaximum("level", lambda s: s.party_level_sum, cap=1.0),
    "badge": lambda: NewMaximum("badge", lambda s: s.badge_count),
    "milestone": lambda: NewMaximum("milestone", lambda s: s.milestone_count),
}


class RewardRegistry:
    def __init__(self, weights: Mapping[str, float]) -> None:
        unknown = sorted(set(weights) - set(COMPONENTS))
        if unknown:
            raise ValueError(
                f"unknown reward component(s) {unknown}; valid names: {sorted(COMPONENTS)}"
            )
        self._weights = dict(weights)
        self._components = [COMPONENTS[name]() for name in weights]

    def reset(self, state: Progress) -> None:
        for component in self._components:
            component.reset(state)

    def step(self, prev: Progress, curr: Progress) -> tuple[float, dict[str, float]]:
        """The weighted reward, and each component's raw value as `reward/<name>`."""
        raw = {c.name: c.score(prev, curr) for c in self._components}
        reward = sum(self._weights[name] * value for name, value in raw.items())
        return reward, {f"reward/{name}": value for name, value in raw.items()}
