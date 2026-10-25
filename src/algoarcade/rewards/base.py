"""The shape of one reward term. Contract: docs/harness-spec.md, Reward registry."""

from typing import Protocol

from algoarcade.games.base import GameState


class RewardComponent[S: GameState](Protocol):
    name: str

    def reset(self, state: S) -> None:
        """Called with the first state of an episode."""
        ...

    def score(self, prev: S, curr: S) -> float:
        """Reward for moving from prev to curr. Zero when nothing changed."""
        ...
