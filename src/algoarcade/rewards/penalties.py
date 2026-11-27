"""Penalties that are off by default. Each returns 1.0 when it fires; the weight is negative."""

from collections import deque
from collections.abc import Hashable
from typing import Protocol, override

from algoarcade.games.base import GameState
from algoarcade.rewards.base import RewardComponent

LOOP_WINDOW = 30  # steps; docs/technical-plan.md, Reward design


class AntiLoop(RewardComponent[GameState]):
    """Fires on a move onto a tile the player stood on in the last 30 steps. Standing still
    never fires it. This window is a penalty, not novelty: novelty keeps a permanent per-episode
    set (Explore)."""

    name = "anti_loop"

    def __init__(self) -> None:
        self._recent: deque[tuple[Hashable, ...]] = deque(maxlen=LOOP_WINDOW)

    @override
    def reset(self, state: GameState) -> None:
        self._recent = deque([state.location], maxlen=LOOP_WINDOW)

    @override
    def score(self, prev: GameState, curr: GameState) -> float:
        fired = curr.location != prev.location and curr.location in self._recent
        self._recent.append(curr.location)
        return 1.0 if fired else 0.0


class Dialogue(GameState, Protocol):
    @property
    def text_box_open(self) -> bool:
        """A text box is printing, waiting at its arrow, or finished and waiting for A or B."""
        ...

    @property
    def text_progress(self) -> int:
        """Any value that changes whenever the text in the box advances."""
        ...


class AntiSpam(RewardComponent[Dialogue]):
    """Fires on a press that left an open text box where it was: printing, waiting at its arrow,
    or finished and waiting for A or B, in the field or in battle. A step where nothing changed
    with a text box open is exactly what it charges, so it is not zero on every no-op step."""

    name = "anti_spam"

    @override
    def reset(self, state: Dialogue) -> None:
        pass

    @override
    def score(self, prev: Dialogue, curr: Dialogue) -> float:
        stuck = (
            prev.text_box_open and curr.text_box_open and curr.text_progress == prev.text_progress
        )
        return 1.0 if stuck else 0.0


class Time(RewardComponent[GameState]):
    """Fires on every step, no-op steps included. An episode ends only on its step budget, so
    every policy pays the same total; it only matters once something can end an episode early."""

    name = "time"

    @override
    def reset(self, state: GameState) -> None:
        pass

    @override
    def score(self, prev: GameState, curr: GameState) -> float:
        return 1.0
