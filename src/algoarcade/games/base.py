"""What every game provides. Contract: docs/harness-spec.md, StateReader and GameState."""

from collections.abc import Hashable
from typing import Protocol

from algoarcade.backends.base import GameBackend


class StateNotReady(Exception):
    """Game memory is not ready to provide a valid state."""


class GameState(Protocol):
    """A frozen record of one moment in a game. Each game adds its own fields."""

    @property
    def location(self) -> tuple[Hashable, ...]:
        """Where the player is. For FireRed: (map_group, map_num, x, y)."""
        ...


class StateReader(Protocol):
    def read(self, backend: GameBackend) -> GameState:
        """Raise StateNotReady when game memory cannot provide a valid state."""
        ...
