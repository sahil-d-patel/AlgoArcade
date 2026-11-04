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
    """Pays a rise in a count above its best so far this episode, once two consecutive reads
    agree on it, up to `cap` per step.

    FireRed's save blocks move on warps, battles and menus, and for one frame a count can read
    wrong, high or low. A step ignores its pair of reads if either is above `maximum` or `curr`
    is more than `max_rise` above `prev`. If `prev` is at most `max_rise` above the best, the step
    pays the rise from the best to `min(prev, curr)`, the count both reads agree on, and that
    becomes the best. A real rise pays one step late, a one-frame spike above the true count never
    pays, and a count that climbs on consecutive steps pays each rise as it is confirmed, so the
    cap does not swallow them. A drop, or a climb back to the old best, pays nothing. The excess
    of a rise above the cap is never paid, nor is any part of a rise above `ceiling`, so an
    episode pays at most `ceiling`.

    Otherwise there is no best yet, or `prev` is more than `max_rise` above it, and the step pays
    nothing. If `curr` is 0 to `max_rise` above `prev`, the best becomes `prev` on the first step
    and `curr` on a later one. So a bad reset read is never trusted on its own, and a bad best is
    replaced once two reads agree. A reset read up to `max_rise` too low does become the best, and
    the climb back from it pays. A bad read that repeats on two consecutive steps can become the
    best and silence the term for the episode; only reader validation (ENV-05, ENV-08) prevents
    that."""

    def __init__(  # noqa: PLR0913 two bounds on reads and two on payment
        self,
        name: str,
        count: Callable[[Progress], int],
        maximum: int,
        max_rise: int,
        cap: float = float("inf"),
        ceiling: float = float("inf"),
    ) -> None:
        self.name = name
        self._count = count
        self._maximum = maximum
        self._max_rise = max_rise
        self._cap = cap
        self._ceiling = ceiling
        self._best: int | None = None
        self._first_step = True

    @override
    def reset(self, state: Progress) -> None:
        self._best = None
        self._first_step = True

    @override
    def score(self, prev: Progress, curr: Progress) -> float:
        prev_value, value = self._count(prev), self._count(curr)
        first_step, self._first_step = self._first_step, False
        if max(prev_value, value) > self._maximum or value - prev_value > self._max_rise:
            return 0.0
        if self._best is not None and prev_value - self._best <= self._max_rise:
            confirmed = min(prev_value, value)
            paid = self._paid(self._best, confirmed)
            self._best = max(self._best, confirmed)
            return paid
        if value >= prev_value:
            self._best = prev_value if first_step else value
        return 0.0

    def _paid(self, low: int, high: int) -> float:
        rise = min(high, self._ceiling) - min(low, self._ceiling)
        return min(float(max(0, rise)), self._cap)
