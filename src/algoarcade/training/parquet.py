"""Writes a run's steps, visits and episodes through the OPS-03 writer.

Contract: docs/harness-spec.md, Logging schema.
"""

import time
from collections import Counter
from collections.abc import Hashable, Iterable, Mapping
from pathlib import Path
from typing import Any

import pyarrow as pa
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper

from algoarcade.logs.schema import EPISODES, REWARD_COMPONENTS, STEPS, VISITS
from algoarcade.logs.writer import RollingWriter
from algoarcade.training.config import TrainConfig

STEP_BATCH = 4096
VISIT_WINDOW = 500_000
LLM_COLUMNS = ("llm_model", "llm_tool_level", "llm_tokens_in", "llm_tokens_out", "llm_cost_usd")


class ParquetLog(VecEnvWrapper):
    """Logs every decision of `venv` from the main process.

    A training log writes the steps, visits and episodes tables, and its `env_steps` matches
    SB3's num_timesteps. An eval log, built with `training=` the training log, writes only
    episodes, marked is_eval with the training log's env_steps as checkpoint_step. The run id
    is the run directory's name after `run_id=`."""

    def __init__(
        self,
        venv: VecEnv,
        config: TrainConfig,
        run: Path,
        start: float,
        training: "ParquetLog | None" = None,
    ) -> None:
        super().__init__(venv)
        self._ids = {"run_id": run.name.removeprefix("run_id="), "seed": config.seed}
        self._start_state = config.start_state
        self._components = set(config.reward_weights)
        self._step_log_every = config.step_log_every
        self._start = start
        self._training = training
        writer_id = "train" if training is None else "eval"
        self._episodes = RollingWriter(run, "episodes", writer_id)
        if training is None:
            self._steps = RollingWriter(run, "steps", writer_id)
            self._visits = RollingWriter(run, "visits", writer_id)
        self.env_steps = 0
        self._step_rows: list[dict[str, Any]] = []
        self._window = 0
        self._visit_counts: Counter[tuple[Hashable, ...]] = Counter()
        self._actions: Any = None
        n = venv.num_envs
        self._episode_id = [0] * n
        self._episode_step = [0] * n
        self._return = [0.0] * n
        self._totals: list[Counter[str]] = [Counter() for _ in range(n)]
        self._tiles: list[set[tuple[Hashable, ...]]] = [set() for _ in range(n)]
        self._goal_step: list[int | None] = [None] * n
        self._archive_start = [False] * n

    def _begin_episodes(self, envs: Iterable[int]) -> None:
        for i in envs:
            self._episode_step[i] = 0
            self._return[i] = 0.0
            self._totals[i] = Counter()
            reset_info = self.venv.reset_infos[i]
            self._tiles[i] = {tuple(reset_info["location"])}
            self._archive_start[i] = reset_info["archive_start"]
            self._goal_step[i] = None

    def reset(self) -> Any:
        obs = self.venv.reset()
        self._begin_episodes(range(self.num_envs))
        return obs

    def step_async(self, actions: Any) -> None:
        self._actions = actions
        self.venv.step_async(actions)

    def step_wait(self) -> Any:
        obs, rewards, dones, infos = self.venv.step_wait()
        clock = time.time() - self._start
        episode_rows = []
        for i, info in enumerate(infos):
            self._episode_step[i] += 1
            self._return[i] += float(rewards[i])
            self._totals[i].update({c: info[f"reward/{c}"] for c in self._components})
            location = tuple(info["location"])
            self._tiles[i].add(location)
            if info["goal"] and self._goal_step[i] is None:
                self._goal_step[i] = self._episode_step[i]
            if self._training is None:
                self.env_steps += 1
                self._log_step(i, info, location, float(rewards[i]), clock)
            if dones[i]:
                episode_rows.append(self._episode_row(i, info, clock))
                self._episode_id[i] += 1
        if episode_rows:
            self._episodes.write(pa.Table.from_pylist(episode_rows, schema=EPISODES))
            self._begin_episodes([i for i, done in enumerate(dones) if done])
        if len(self._step_rows) >= STEP_BATCH:
            self._flush_steps()
        return obs, rewards, dones, infos

    def _log_step(
        self, i: int, info: dict[str, Any], location: tuple[Any, ...], reward: float, clock: float
    ) -> None:
        window = (self.env_steps - 1) // VISIT_WINDOW * VISIT_WINDOW
        if window != self._window:
            self._flush_visits()
            self._window = window
        self._visit_counts[location] += 1
        if self._episode_step[i] % self._step_log_every:
            return
        map_group, map_num, x, y = _place(location)
        self._step_rows.append(
            self._ids
            | {
                "env_idx": i,
                "env_steps": self.env_steps,
                "wall_clock_s": clock,
                "episode_id": self._episode_id[i],
                "episode_step": self._episode_step[i],
                "action": int(self._actions[i]),
                "map_group": map_group,
                "map_num": map_num,
                "x": x,
                "y": y,
                "in_battle": info["in_battle"],
                "milestone_count": info["milestone_count"],
                "reward_total": reward,
            }
            | self._reward_columns({c: info[f"reward/{c}"] for c in self._components})
        )

    def _episode_row(self, i: int, info: dict[str, Any], clock: float) -> dict[str, Any]:
        training = self._training
        return (
            self._ids
            | {
                "env_idx": i,
                "episode_id": self._episode_id[i],
                "env_steps": self.env_steps if training is None else training.env_steps,
                "wall_clock_s": clock,
                "episode_return": self._return[i],
                "episode_length": self._episode_step[i],
                "badge_count": info["badge_count"],
                "party_level_sum": info["party_level_sum"],
                "story_flags": info["story_flags"],
                "unique_tiles": len(self._tiles[i]),
                "milestone_count": info["milestone_count"],
                "start_state": self._start_state,
                "archive_start": self._archive_start[i],
                # AlgoArcadeEnv ends episodes on the step budget only. A Super Mario Land
                # continue after a game over is not an end: the episode goes on.
                "end_reason": "truncated",
                "goal_reached": self._goal_step[i] is not None,
                "goal_step": self._goal_step[i],
                "is_eval": training is not None,
                "checkpoint_step": None if training is None else training.env_steps,
            }
            | self._reward_columns(self._totals[i])
            | dict.fromkeys(LLM_COLUMNS)
        )

    def _reward_columns(self, raw: Mapping[str, float]) -> dict[str, float | None]:
        return {f"reward_{c}": raw[c] if c in self._components else None for c in REWARD_COMPONENTS}

    def _flush_steps(self) -> None:
        if self._step_rows:
            self._steps.write(pa.Table.from_pylist(self._step_rows, schema=STEPS))
            self._step_rows = []

    def _flush_visits(self) -> None:
        rows = [
            {
                "run_id": self._ids["run_id"],
                "window_start_step": self._window,
                "map_group": map_group,
                "map_num": map_num,
                "x": x,
                "y": y,
                "visits": visits,
            }
            for location, visits in self._visit_counts.items()
            for map_group, map_num, x, y in [_place(location)]
        ]
        if rows:
            self._visits.write(pa.Table.from_pylist(rows, schema=VISITS))
        self._visit_counts = Counter()

    def close(self) -> None:
        if self._training is None:
            self._flush_steps()
            self._flush_visits()
            self._steps.close()
            self._visits.close()
        self._episodes.close()
        self.venv.close()


def _place(location: tuple[Any, ...]) -> tuple[Any, ...]:
    """(map_group, map_num, x, y). The fake game's (room, x, y) logs its room as map_num."""
    return (0,) * (4 - len(location)) + location
