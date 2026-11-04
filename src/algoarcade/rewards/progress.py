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
    """Pays a rise in a count above its best so far this episode, up to `cap` per step and
    nothing past `ceiling`. The rules are in docs/harness-spec.md, Reward registry.

    A rise pays one step late, on the read that confirms it: a step pays up to `min(prev, curr)`.
    FireRed's save blocks move on warps, battles and menus, and for one frame a count can read
    high, so a single read must never pay. The confirming step therefore pays even when nothing
    changed on it, and a rise on an episode's last step is never paid.

    On the first step the best becomes the reset read if the first read agrees with it, so a rise
    on the first step pays on the second; a reset read up to `max_rise` too low becomes the best
    too. With no best, or `prev` more than `max_rise` above it, an agreeing pair sets the best to
    `prev` and the next step pays nothing, so re-establishing the best never pays; a real rise
    inside that pair is lost with it, at most `max_rise`. A bad read
    repeated on two consecutive steps becomes the best this way and can silence the term for the
    episode; only reader validation (ENV-05, ENV-08) prevents that."""

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
        self._settling = False

    @override
    def reset(self, state: Progress) -> None:
        self._best = None
        self._first_step = True
        self._settling = False

    @override
    def score(self, prev: Progress, curr: Progress) -> float:
        prev_value, value = self._count(prev), self._count(curr)
        first_step, self._first_step = self._first_step, False
        settling, self._settling = self._settling, False
        if max(prev_value, value) > self._maximum or value - prev_value > self._max_rise:
            return 0.0
        if self._best is not None and prev_value - self._best <= self._max_rise:
            confirmed = min(prev_value, value)
            paid = 0.0 if settling else self._paid(self._best, confirmed)
            self._best = max(self._best, confirmed)
            return paid
        if value >= prev_value:
            self._best = prev_value
            self._settling = not first_step
        return 0.0

    def _paid(self, low: int, high: int) -> float:
        rise = min(high, self._ceiling) - min(low, self._ceiling)
        return min(float(max(0, rise)), self._cap)
