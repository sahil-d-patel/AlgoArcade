"""Curriculum resets (AGT-06, E8): training episodes start from states a successful LLM attempt
passed through, latest first. Contract: docs/harness-spec.md, Curriculum resets.

Evaluation never goes through the curriculum: its env is built without `curriculum`."""

import hashlib
import json
import zlib
from collections import deque
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper

from algoarcade.envs.env import CELL_COMPRESSION, Cell

ROOT = Path(__file__).resolve().parents[3] / "data" / "states" / "curriculum"


def load(folder: Path, start_state: str) -> list[Cell]:
    """The states in `folder`'s manifest (algoarcade.agent.curriculum), in press order, as
    cells: the archive-reset path loads them with their visited sets and bests."""
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest["start_state"] != start_state:
        raise ValueError(f"{folder} was captured from {manifest['start_state']}, not {start_state}")
    cells = []
    for entry in manifest["states"]:
        state = (folder / entry["file"]).read_bytes()
        if hashlib.sha256(state).hexdigest() != entry["sha256"]:
            raise ValueError(f"{folder / entry['file']} does not match the manifest hash")
        cells.append(
            Cell(
                (entry["action"],),
                zlib.compress(state, CELL_COMPRESSION),
                frozenset(tuple(location) for location in entry["visited"]),
                entry["bests"],
                entry["milestone_count"],
            )
        )
    return cells


class CurriculumResets(VecEnvWrapper):
    """Starts every training episode from the current state of `cells`, the latest at first.
    Once at least `threshold` of the last `window` episodes started from it reached the goal
    (`info["goal"]` on any step), it moves one state earlier, and after the first state to the
    fixed start state, where it stays. `stage` is the current state's index, -1 for the fixed
    start, and `success` the goal rate over the episodes counted so far at this stage.

    As in ArchiveResets, each worker's next start is sent ahead with `set_next_start`, since a
    vector env resets a worker inside the worker. So a worker whose episode ends on the step
    that moves the curriculum back has already started one more episode from the old state; it
    does not count. Wrap the SB3 vector env directly: `reset_infos` is passed on for wrappers
    above, such as ParquetLog."""

    def __init__(self, venv: VecEnv, cells: Sequence[Cell], threshold: float, window: int) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"curriculum_threshold is {threshold}; it must be in (0, 1]")
        super().__init__(venv)
        self._cells = list(cells)
        self._threshold = threshold
        self.stage = len(self._cells) - 1
        self.outcomes: deque[bool] = deque(maxlen=window)
        self._started = [self.stage] * venv.num_envs
        self._reached = [False] * venv.num_envs

    @property
    def success(self) -> float:
        return sum(self.outcomes) / len(self.outcomes) if self.outcomes else 0.0

    def reset(self) -> Any:
        self._send(range(self.num_envs))
        obs = self.venv.reset()
        self.reset_infos = self.venv.reset_infos
        self._started = [self.stage] * self.num_envs
        self._reached = [False] * self.num_envs
        self._send(range(self.num_envs))
        return obs

    def step_wait(self) -> Any:
        obs, rewards, dones, infos = self.venv.step_wait()
        self.reset_infos = self.venv.reset_infos
        ended = []
        for i, info in enumerate(infos):
            self._reached[i] = self._reached[i] or bool(info["goal"])
            if dones[i]:
                if self._started[i] == self.stage:
                    self.outcomes.append(self._reached[i])
                self._started[i] = self.stage
                self._reached[i] = False
                ended.append(i)
        full = len(self.outcomes) == self.outcomes.maxlen
        if self.stage >= 0 and full and self.success >= self._threshold:
            self.stage -= 1
            self.outcomes.clear()
            self._send(range(self.num_envs))
        else:
            self._send(ended)
        return obs, rewards, dones, infos

    def _send(self, envs: Iterable[int]) -> None:
        envs = list(envs)
        if envs:
            cell = self._cells[self.stage] if self.stage >= 0 else None
            self.venv.env_method("set_next_start", cell, indices=envs)
