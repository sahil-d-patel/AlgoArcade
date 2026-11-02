"""Terms that pay for a new maximum of one progress count: flags, levels, badges, milestones."""

from collections.abc import Callable
from typing import Protocol, override

from algoarcade.games.base import GameState
from algoarcade.rewards.base import RewardComponent


class Progress(GameState, Protocol):
    @property
    def story_flags(self) -> int: ...

    @property
    def party_level_sum(self) -> int: ...

    @property
    def badge_count(self) -> int: ...

    @property
    def milestone_count(self) -> int: ...


class NewMaximum(RewardComponent[Progress]):
    """Pays the amount a count rises above its best so far this episode, up to `cap` per step.
    A drop, or a climb back to the old best, pays nothing, so moving a count up and down cannot be
    farmed. The excess of a rise above the cap is never paid.

    A count above `maximum`, or more than `max_rise` above the best, is a bad read: it pays nothing
    and does not become the best, so it cannot silence the real gains that follow."""

    def __init__(
        self,
        name: str,
        count: Callable[[Progress], int],
        maximum: int,
        max_rise: float = float("inf"),
        cap: float = float("inf"),
    ) -> None:
        self.name = name
        self._count = count
        self._maximum = maximum
        self._max_rise = max_rise
        self._cap = cap
        self._best = 0

    @override
    def reset(self, state: Progress) -> None:
        self._best = self._count(state)

    @override
    def score(self, prev: Progress, curr: Progress) -> float:
        value = self._count(curr)
        if value > self._maximum or value - self._best > self._max_rise:
            return 0.0
        gain = max(0, value - self._best)
        self._best = max(self._best, value)
        return min(float(gain), self._cap)
