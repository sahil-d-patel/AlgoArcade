"""Pays the first visit to each location in an episode."""

from collections.abc import Hashable
from typing import override

from algoarcade.games.base import GameState
from algoarcade.rewards.base import RewardComponent


class Explore(RewardComponent):
    name = "explore"

    def __init__(self) -> None:
        self._visited: set[tuple[Hashable, ...]] = set()

    @override
    def reset(self, state: GameState) -> None:
        self._visited = {state.location}

    @override
    def score(self, prev: GameState, curr: GameState) -> float:
        if curr.location in self._visited:
            return 0.0
        self._visited.add(curr.location)
        return 1.0
