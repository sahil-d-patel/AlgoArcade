"""The Gymnasium environment. Contract: docs/harness-spec.md, AlgoArcadeEnv."""

from collections.abc import Callable, Mapping
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from algoarcade.backends.base import Button, GameBackend
from algoarcade.games.base import GameState, StateNotReady, StateReader
from algoarcade.rewards.registry import RewardRegistry

ACTIONS: tuple[tuple[Button, ...], ...] = (
    (Button.UP,),
    (Button.DOWN,),
    (Button.LEFT,),
    (Button.RIGHT,),
    (Button.A,),
    (Button.B,),
)
HOLD = 8
RELEASE = 16
# A guess. A 24-frame step absorbs most of the offset, so it changes outcomes only through the
# game's RNG; measured in docs/harness-spec.md, AlgoArcadeEnv.
MAX_NOOP_FRAMES = 7
# Frames a not-ready read is retried, one frame apart, before StateNotReady propagates. FireRed's
# party checksum fails for one frame while the game rewrites an entry, never two in a row in the
# per-frame reads measured in docs/harness-spec.md, AlgoArcadeEnv. Eight is margin over that; a
# state that stays not ready, such as the title screen, still raises.
NOT_READY_FRAMES = 8
# Rows and columns of tiles around the player. A guess: FireRed shows 15 by 10 tiles.
CROP = (9, 15)


class AlgoArcadeEnv(gym.Env[dict[str, Any], np.int64]):
    """`features` maps a name to a function of the GameState scaled into [0, 1]; the vector
    clips to that range. The location's last two elements are the tile x and y, and the rest name
    the map. A read that raises StateNotReady is retried one frame later, up to NOT_READY_FRAMES
    times; then it propagates out of reset and step. The env never stands in a state for one the
    game cannot provide."""

    def __init__(  # noqa: PLR0913 the inputs harness-spec lists for the env
        self,
        backend: GameBackend,
        reader: StateReader,
        rewards: RewardRegistry[Any],
        start_state: bytes,
        step_budget: int,
        features: Mapping[str, Callable[[Any], float]],
        *,
        start: bool = False,
        noop: bool = False,
    ) -> None:
        self.backend = backend
        self._reader = reader
        self._rewards = rewards
        self._start_state = start_state
        self._step_budget = step_budget
        self._features = tuple(features.values())
        self._actions = ACTIONS + ((Button.START,),) * start + ((),) * noop
        self.action_space = spaces.Discrete(len(self._actions))
        self.observation_space = spaces.Dict(
            {
                "vector": spaces.Box(0.0, 1.0, (len(self._features),), np.float32),
                "visited": spaces.Box(0.0, 1.0, (1, *CROP), np.float32),
            }
        )
        self._visited: set[tuple[Any, ...]] = set()
        self._steps = 0
        self._milestone_count = 0
        self._last_milestone_count: int | None = None

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self.backend.load_state(self._start_state)
        self.backend.step(int(self.np_random.integers(MAX_NOOP_FRAMES + 1)))
        self._state = self._read()
        self._rewards.reset(self._state)
        self._visited = {self._state.location}
        self._steps = 0
        self._milestone_count = 0
        self._last_milestone_count = None
        return self._observe(self._state), self._info(self._state)

    def step(
        self, action: int | np.int64
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self.backend.press(self._actions[int(action)], HOLD, RELEASE)
        prev, self._state = self._state, self._read()
        reward, components = self._rewards.step(prev, self._state)
        self._visited.add(self._state.location)
        self._steps += 1
        truncated = self._steps >= self._step_budget
        info = components | self._info(self._state)
        return self._observe(self._state), float(reward), False, truncated, info

    def close(self) -> None:
        self.backend.close()

    def _read(self) -> GameState:
        for _ in range(NOT_READY_FRAMES):
            try:
                return self._reader.read(self.backend)
            except StateNotReady:
                self.backend.step(1)
        return self._reader.read(self.backend)

    def _observe(self, state: GameState) -> dict[str, Any]:
        vector = np.array([feature(state) for feature in self._features], dtype=np.float32)
        location: tuple[Any, ...] = state.location
        *place, x, y = location
        rows, cols = CROP
        visited = np.zeros((1, rows, cols), dtype=np.float32)
        for row in range(rows):
            for col in range(cols):
                tile = (*place, x + col - cols // 2, y + row - rows // 2)
                visited[0, row, col] = tile in self._visited
        return {"vector": np.clip(vector, 0.0, 1.0), "visited": visited}

    def _info(self, state: GameState) -> dict[str, Any]:
        info: dict[str, Any] = {"location": state.location}
        # Super Mario Land has no milestones. FireRed's count drops when the player walks off a
        # map-based milestone, so the episode keeps its best. A count can read high for one frame
        # while a save block moves, so the best rises only to the lower of two consecutive reads,
        # the reset read included: a real rise shows one step late, as in NewMaximum.
        milestone_count = getattr(state, "milestone_count", None)
        if milestone_count is not None:
            last, self._last_milestone_count = self._last_milestone_count, milestone_count
            if last is not None:
                self._milestone_count = max(self._milestone_count, min(last, milestone_count))
            info["milestone_count"] = self._milestone_count
        return info
