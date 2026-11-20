"""The Gymnasium environment. Contract: docs/harness-spec.md, AlgoArcadeEnv."""

import zlib
from collections.abc import Callable, Hashable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, get_args

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

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
# The grid's channels.
VISITED = 0
WALKABLE = 1
ObsMode = Literal["symbolic", "pixels", "both"]
# Rows and columns of a greyscale frame, and frames stacked: technical-plan.md, Observations.
PIXELS = (72, 80)
FRAMES = 3
# ITU-R BT.601 luma, which PIL's "L" mode and OpenCV's RGB2GRAY also use.
LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)
# Tiles per side of an archive cell. A starting guess from ENV-09.
CELL_TILES = 4
# zlib level 1 shrinks an mGBA save state from 528,448 bytes to about 50,000 in about 4 ms.
CELL_COMPRESSION = 1


@dataclass(frozen=True)
class Cell:
    """A cell for archive resets, from the step that first reached it: its key, the save state
    (zlib-compressed), the episode's visited locations, each progress term's best and the env's
    best milestone_count. The bests travel with the state because a count can sit below them:
    FireRed's forest_exited and gym_entered stop holding off their maps, and depositing a
    Pokemon lowers the level sum."""

    key: tuple[Hashable, ...]
    state: bytes
    visited: frozenset[tuple[Hashable, ...]]
    bests: Mapping[str, int]
    milestone_count: int


class AlgoArcadeEnv(gym.Env[dict[str, Any], np.int64]):
    """`features` maps a name to a function of the GameState scaled into [0, 1]; the vector
    clips to that range. The location's last two elements are the tile x and y, and the rest name
    the map. A read that raises StateNotReady is retried one frame later, up to NOT_READY_FRAMES
    times; then it propagates out of reset and step. The env never stands in a state for one the
    game cannot provide.

    With `archive`, a step that first reaches a cell in an episode puts a Cell in `info["cell"]`
    unless `add_archived` listed its key, and `set_next_start` makes the next reset load a cell
    instead of the start state. The key is
    (*map, x // CELL_TILES, y // CELL_TILES, badge_flags, the episode's best milestone_count);
    a game without a badge byte or milestones keys 0 for them.

    `obs_mode` picks the observation (docs/harness-spec.md, AlgoArcadeEnv). `walkable(location)`
    fills the grid's walkable channel; without it every tile reads walkable. A mode with pixels
    needs a backend that renders."""

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
        archive: bool = False,
        obs_mode: ObsMode = "symbolic",
        walkable: Callable[[Any], bool] = lambda _tile: True,
    ) -> None:
        if obs_mode not in get_args(ObsMode):
            raise ValueError(f"obs_mode {obs_mode!r}: use one of {get_args(ObsMode)}")
        self.backend = backend
        self._reader = reader
        self._rewards = rewards
        self._start_state = start_state
        self._step_budget = step_budget
        self._archive = archive
        self._next_start: Cell | None = None
        self._cells: set[tuple[Hashable, ...]] = set()
        self._archived: set[tuple[Hashable, ...]] = set()
        self._features = tuple(features.values())
        self._actions = ACTIONS + ((Button.START,),) * start + ((),) * noop
        self.action_space = spaces.Discrete(len(self._actions))
        self._walkable = walkable
        self._grid = obs_mode != "pixels"
        self._pixels = obs_mode != "symbolic"
        observation: dict[str, spaces.Space[Any]] = {
            "vector": spaces.Box(0.0, 1.0, (len(self._features),), np.float32)
        }
        if self._grid:
            observation["grid"] = spaces.Box(0.0, 1.0, (2, *CROP), np.float32)
        if self._pixels:
            observation["pixels"] = spaces.Box(0, 255, (FRAMES, *PIXELS), np.uint8)
            height, width, _ = backend.screen().shape
            self._rows = area_weights(height, PIXELS[0])
            self._cols = area_weights(width, PIXELS[1]).T
            self._frames = np.zeros((FRAMES, *PIXELS), np.uint8)
        self.observation_space = spaces.Dict(observation)
        self._visited: set[tuple[Any, ...]] = set()
        self._steps = 0
        self._milestone_count = 0
        self._last_milestone_count: int | None = None

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed, options=options)
        cell, self._next_start = self._next_start, None
        self.backend.load_state(self._start_state if cell is None else zlib.decompress(cell.state))
        self.backend.step(int(self.np_random.integers(MAX_NOOP_FRAMES + 1)))
        self._state = self._read()
        if self._pixels:
            self._frames[:] = self._frame()
        if cell is None:
            self._rewards.reset(self._state)
            self._visited = {self._state.location}
            self._milestone_count = 0
        else:
            self._rewards.reset(self._state, cell.visited, cell.bests)
            self._visited = {self._state.location, *cell.visited}
            self._milestone_count = cell.milestone_count
        self._cells = set()
        self._steps = 0
        self._last_milestone_count = None
        info = self._info(self._state) | {"archive_start": cell is not None}
        return self._observe(self._state), info

    def set_next_start(self, cell: Cell) -> None:
        """The next reset loads `cell` with its visited set. Ignored without `archive`, so an
        evaluation env always starts from the start state."""
        if self._archive:
            self._next_start = cell

    def add_archived(self, keys: Iterable[tuple[Hashable, ...]]) -> None:
        """Keys the archive already holds; reaching one reports no Cell, so its state is not
        saved and sent again."""
        self._archived.update(keys)

    def step(
        self, action: int | np.int64
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self.backend.press(self._actions[int(action)], HOLD, RELEASE)
        prev, self._state = self._state, self._read()
        if self._pixels:
            self._frames[:-1] = self._frames[1:]
            self._frames[-1] = self._frame()
        reward, components = self._rewards.step(prev, self._state)
        self._visited.add(self._state.location)
        self._steps += 1
        truncated = self._steps >= self._step_budget
        info = components | self._info(self._state)
        if self._archive:
            key = self._cell_key(self._state)
            if key not in self._cells:
                self._cells.add(key)
                if key not in self._archived:
                    info["cell"] = Cell(
                        key,
                        zlib.compress(self.backend.save_state(), CELL_COMPRESSION),
                        frozenset(self._visited),
                        self._rewards.bests(),
                        self._milestone_count,
                    )
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

    def _cell_key(self, state: GameState) -> tuple[Hashable, ...]:
        location: tuple[Any, ...] = state.location
        *place, x, y = location
        badges = getattr(state, "badge_flags", 0)
        return (*place, x // CELL_TILES, y // CELL_TILES, badges, self._milestone_count)

    def _frame(self) -> NDArray[np.uint8]:
        grey = self.backend.screen().astype(np.float32) @ LUMA
        return np.rint(self._rows @ grey @ self._cols).astype(np.uint8)

    def _observe(self, state: GameState) -> dict[str, Any]:
        vector = np.array([feature(state) for feature in self._features], dtype=np.float32)
        obs: dict[str, NDArray[Any]] = {"vector": np.clip(vector, 0.0, 1.0)}
        if self._grid:
            location: tuple[Any, ...] = state.location
            *place, x, y = location
            rows, cols = CROP
            grid = np.zeros((2, rows, cols), dtype=np.float32)
            for row in range(rows):
                for col in range(cols):
                    tile = (*place, x + col - cols // 2, y + row - rows // 2)
                    grid[VISITED, row, col] = tile in self._visited
                    grid[WALKABLE, row, col] = self._walkable(tile)
            obs["grid"] = grid
        if self._pixels:
            obs["pixels"] = self._frames.copy()
        return obs

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


def area_weights(size: int, out: int) -> NDArray[np.float32]:
    """An (out, size) matrix that resamples a line of `size` pixels to `out`, each output pixel
    the mean of the input it covers, split pixels weighted by the share covered. The GBA's 240 x
    160 frame goes to 80 x 72: 3 pixels across and 2.22 down per output pixel."""
    edges = np.linspace(0, size, out + 1)
    starts = np.arange(size)
    overlap = np.minimum(edges[1:, None], starts + 1) - np.maximum(edges[:-1, None], starts)
    weights = np.clip(overlap, 0, None)
    return (weights / weights.sum(axis=1, keepdims=True)).astype(np.float32)
