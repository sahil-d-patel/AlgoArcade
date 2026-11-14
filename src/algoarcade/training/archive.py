"""Archive resets (ENV-09): one archive in the main process, and the SB3 wrapper that starts
training episodes from it. Contract: docs/harness-spec.md, Start states and segments.

Evaluation never goes through ArchiveResets, and its env is built without `archive`."""

from collections import Counter
from collections.abc import Hashable, Iterable
from typing import Any

import numpy as np
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper

from algoarcade.envs.env import Cell


class Archive:
    """Cells by key, each with the first state that reached it, and how often each was handed
    out. Independent of SB3, for the TRN-07 trainer too."""

    def __init__(self) -> None:
        self.cells: dict[tuple[Hashable, ...], Cell] = {}
        self.selections: Counter[tuple[Hashable, ...]] = Counter()

    def add(self, cell: Cell) -> None:
        self.cells.setdefault(cell.key, cell)

    def sample(self, rng: np.random.Generator) -> Cell:
        """A cell drawn with weight 1 / sqrt(1 + selections). The caller counts the selection."""
        keys = list(self.cells)
        weights = 1 / np.sqrt(1 + np.array([self.selections[key] for key in keys]))
        return self.cells[keys[rng.choice(len(keys), p=weights / weights.sum())]]


class ArchiveResets(VecEnvWrapper):
    """Adds every cell the envs report to the archive, and starts each training episode from an
    archived cell with probability `p`, otherwise from the fixed start state. Each new key goes
    to every env with `add_archived`, so no env sends that cell's state again.

    A vector env resets a worker inside the worker, so each worker's next start is chosen ahead
    of time and sent with `set_next_start`: as soon as the archive has a cell, and again after
    each of the worker's resets. `share` is the fraction of resets so far that loaded a cell, as
    the envs report it in `archive_start`. Wrap the SB3 vector env directly: `reset_infos` is
    passed on for wrappers above, such as ParquetLog."""

    def __init__(self, venv: VecEnv, p: float, seed: int) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"archive_p is {p}; it must be between 0 and 1")
        super().__init__(venv)
        self.archive = Archive()
        self._p = p
        self._rng = np.random.default_rng(seed)
        self._chosen = [False] * venv.num_envs
        self.starts = 0
        self.archive_starts = 0

    @property
    def share(self) -> float:
        return self.archive_starts / self.starts if self.starts else 0.0

    def reset(self) -> Any:
        self._choose()
        obs = self.venv.reset()
        self._started(range(self.num_envs))
        return obs

    def step_wait(self) -> Any:
        obs, rewards, dones, infos = self.venv.step_wait()
        new = []
        for info in infos:
            cell = info.get("cell")
            if cell is not None and cell.key not in self.archive.cells:
                self.archive.add(cell)
                new.append(cell.key)
        if new:
            self.venv.env_method("add_archived", new)
        self._started(np.flatnonzero(dones))
        return obs, rewards, dones, infos

    def _started(self, envs: Iterable[int]) -> None:
        self.reset_infos = self.venv.reset_infos
        for i in envs:
            self.starts += 1
            self.archive_starts += self.reset_infos[i]["archive_start"]
            self._chosen[i] = False
        self._choose()

    def _choose(self) -> None:
        if not self.archive.cells:
            return
        for i, chosen in enumerate(self._chosen):
            if chosen:
                continue
            self._chosen[i] = True
            if self._rng.random() < self._p:
                cell = self.archive.sample(self._rng)
                self.archive.selections[cell.key] += 1
                self.venv.env_method("set_next_start", cell, indices=[i])
