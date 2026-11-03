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

    A read above `maximum`, or more than `max_rise` above the best, pays nothing and does not
    become the best. The best comes only from two consecutive reads that agree: the second is 0 to
    `max_rise` above the first. On the first step the pair is the reset read and the next read,
    and their rise pays. Any later pair, after a rejected read, only sets the best and pays
    nothing. So a bad reset read is never trusted on its own, and a bad best is replaced once two
    reads agree. A bad read that repeats on two consecutive steps can therefore become the best
    and silence the term for the episode; only reader validation (ENV-05, ENV-08) prevents that."""

    def __init__(
        self,
        name: str,
        count: Callable[[Progress], int],
        maximum: int,
        max_rise: int,
        cap: float = float("inf"),
    ) -> None:
        self.name = name
        self._count = count
        self._maximum = maximum
        self._max_rise = max_rise
        self._cap = cap
        self._best: int | None = None
        self._first_step = True

    @override
    def reset(self, state: Progress) -> None:
        self._best = None
        self._first_step = True

    @override
    def score(self, prev: Progress, curr: Progress) -> float:
        value = self._count(curr)
        first_step, self._first_step = self._first_step, False
        if value > self._maximum:
            return 0.0
        if self._best is not None and value - self._best <= self._max_rise:
            gain = max(0, value - self._best)
            self._best = max(self._best, value)
            return min(float(gain), self._cap)
        rise = value - self._count(prev)
        if 0 <= rise <= self._max_rise:
            self._best = value
            return min(float(rise), self._cap) if first_step else 0.0
        return 0.0
