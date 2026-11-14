"""Train Stable-Baselines3 PPO on a Gymnasium env id or an AlgoArcade game.

Contract: docs/technical-plan.md, Training.

    uv run python -m algoarcade.training.runner cartpole --seed 0
    uv run python -m algoarcade.training.runner firered --start-state seg_01_house --seed 0

The games (env_id firered or fake) also log to Parquet under the run directory; see
algoarcade.training.parquet.

Episodes end through Gymnasium's TimeLimit, which SB3 reports as TimeLimit.truncated and
bootstraps with the value of the final observation. Do not wrap the env in anything that turns a
truncation into a termination.
"""

import csv
import dataclasses
import hashlib
import json
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import tyro
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import CSVOutputFormat, KVWriter, Logger
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecEnv,
    VecNormalize,
    sync_envs_normalization,
)

from algoarcade.envs.env import HOLD, RELEASE
from algoarcade.logs.schema import RUNS
from algoarcade.logs.writer import RollingWriter
from algoarcade.training.archive import ArchiveResets
from algoarcade.training.config import PRESETS, TrainConfig, save_config
from algoarcade.training.games import env_factory
from algoarcade.training.parquet import ParquetLog

# Training envs take seeds seed .. seed + n_envs - 1; the eval env sits clear of them.
EVAL_SEED_OFFSET = 1_000
EVAL_FIELDS = ["step", "mean_return", "std_return", "wall_time_s"]
GAMES = ("fake", "firered", "super_mario_land")
SPAWN = {"vec_env_cls": SubprocVecEnv, "vec_env_kwargs": {"start_method": "spawn"}}


def build_model(config: TrainConfig, env: VecEnv) -> PPO:
    # AlgoArcadeEnv's observation is a Dict, which MlpPolicy refuses.
    policy = "MultiInputPolicy" if isinstance(env.observation_space, spaces.Dict) else "MlpPolicy"
    return PPO(policy, env, seed=config.seed, verbose=0, **dataclasses.asdict(config.ppo))


class CheckpointEval(BaseCallback):
    """Every `every` env steps and at the end: save a checkpoint, evaluate it, append to eval.csv.

    With VecNormalize, its statistics are saved beside the checkpoint as step_<n>.pkl and copied
    to the eval env, which never updates them. The eval env is reseeded before each evaluation,
    so every checkpoint plays the same episodes.
    _on_step runs during a rollout, before that rollout's update, so the final checkpoint waits
    for _on_training_end; a periodic one in the last rollout would miss the last update.

    When the env reports `continues` (Super Mario Land), the total across the training envs so
    far goes to progress.csv and W&B as rollout/continues. With archive resets, each rollout
    logs the share of training resets so far that loaded a cell, the number of cells and their
    mean save state size in bytes.
    """

    def __init__(
        self,
        run_dir: Path,
        config: TrainConfig,
        eval_env: VecEnv,
        resets: ArchiveResets | None = None,
    ) -> None:
        super().__init__()
        self.run_dir = run_dir
        self.config = config
        self.eval_env = eval_env
        self.resets = resets
        self.next_at = config.checkpoint_every
        self.start = time.time()
        self.continues: dict[int, int] = {}

    def _on_step(self) -> bool:
        for i, info in enumerate(self.locals["infos"]):
            if "continues" in info:
                self.continues[i] = info["continues"]
        if self.next_at <= self.num_timesteps < self.config.total_timesteps:
            self.next_at += self.config.checkpoint_every
            self.checkpoint()
        return True

    def _on_rollout_end(self) -> None:
        if self.continues:
            self.logger.record("rollout/continues", sum(self.continues.values()))
        if self.resets is not None:
            cells = self.resets.archive.cells.values()
            size = sum(len(cell.state) for cell in cells)
            self.logger.record("rollout/archive_share", self.resets.share)
            self.logger.record("rollout/archive_cells", len(cells))
            self.logger.record("rollout/archive_bytes_per_cell", size / len(cells) if cells else 0)

    def _on_training_end(self) -> None:
        self.checkpoint()
        self.logger.dump(self.num_timesteps)

    def checkpoint(self) -> None:
        path = self.run_dir / "checkpoints" / f"step_{self.num_timesteps}"
        self.model.save(path.with_suffix(".zip"))
        normalize = self.model.get_vec_normalize_env()
        if normalize is not None:
            normalize.save(str(path.with_suffix(".pkl")))
            sync_envs_normalization(self.training_env, self.eval_env)
        self.eval_env.seed(self.config.seed + EVAL_SEED_OFFSET)
        returns, _ = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=self.config.eval_episodes,
            deterministic=True,
            return_episode_rewards=True,
        )
        mean = float(np.mean(returns))
        self.logger.record("eval/mean_return", mean)
        path = self.run_dir / "eval.csv"
        new = not path.exists()
        with path.open("a", newline="") as f:
            writer = csv.DictWriter(f, EVAL_FIELDS)
            if new:
                writer.writeheader()
            writer.writerow(
                {
                    "step": self.num_timesteps,
                    "mean_return": mean,
                    "std_return": float(np.std(returns)),
                    "wall_time_s": round(time.time() - self.start, 1),
                }
            )


class WandbOutput(KVWriter):
    def __init__(self, **init: Any) -> None:
        import wandb

        self.run = wandb.init(**init)

    def write(
        self,
        key_values: dict[str, Any],
        key_excluded: dict[str, Any],  # noqa: ARG002 - SB3's KVWriter signature
        step: int = 0,
    ) -> None:
        self.run.log({k: v for k, v in key_values.items() if isinstance(v, int | float)}, step=step)

    def close(self) -> None:
        self.run.finish()


def git(*args: str) -> str:
    package = Path(__file__).resolve().parent
    return subprocess.run(
        ["git", *args], cwd=package, capture_output=True, text=True, check=True
    ).stdout.strip()


def run_row(
    config: TrainConfig, run_id: str, backend: str, rom_sha1: str, commit: str
) -> dict[str, Any]:
    """The per-run table's fields that do not change between the start and end rows."""
    return {
        "run_id": run_id,
        # Set by the run queue (OPS-04); a run launched by hand has neither.
        "experiment_id": "",
        "arm": "",
        "algorithm": "ppo",
        "trainer": "sb3",
        "seed": config.seed,
        "env_id": config.env_id,
        "backend": backend,
        "rom_sha1": rom_sha1,
        "start_state": config.start_state,
        "obs_mode": "symbolic",
        "frames_per_decision": HOLD + RELEASE,
        "num_envs": config.n_envs,
        "reward_config_hash": hashlib.sha256(
            json.dumps(config.reward_weights, sort_keys=True).encode()
        ).hexdigest(),
        "git_commit": commit,
        "hyperparameters": json.dumps(dataclasses.asdict(config), default=str),
        "step_log_every": config.step_log_every,
        "machine": socket.gethostname(),
    }


def write_run_row(run_dir: Path, row: dict[str, Any]) -> None:
    """Each row in its own closed file: an open Parquet file cannot be read, so a run killed
    before its end row still reads as running."""
    runs = RollingWriter(run_dir, "runs", "train", resume=True)
    runs.write(pa.Table.from_pylist([row], schema=RUNS))
    runs.close()


def train(config: TrainConfig) -> Path:  # noqa: C901, PLR0915 linear launch steps; a split gives single-use helpers
    name = f"{config.env_id}-{config.start_state}" if config.start_state else config.env_id
    run_id = f"{name}-seed{config.seed}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = config.runs_dir / f"run_id={run_id}"
    game = config.env_id in GAMES
    env_fn, backend, rom_sha1 = (
        env_factory(config.env_id, config.start_state, config.reward_weights)
        if game
        else (config.env_id, "", "")
    )
    (run_dir / "checkpoints").mkdir(parents=True)
    save_config(config, run_dir / "config.json")
    git_info = {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    (run_dir / "git.json").write_text(json.dumps(git_info) + "\n")

    # Wall clock, as the schema means it: perf_counter stops while macOS sleeps.
    start = time.time()
    env: VecEnv
    eval_env: VecEnv
    resets = None
    if game:
        venv = make_vec_env(
            env_fn,
            n_envs=config.n_envs,
            seed=config.seed,
            env_kwargs={"archive": config.archive},
            **SPAWN,
        )
        # The emulator for evaluation runs in its own process too, never in this one.
        eval_venv = make_vec_env(env_fn, n_envs=1, seed=config.seed + EVAL_SEED_OFFSET, **SPAWN)
        if config.archive:
            venv = resets = ArchiveResets(venv, config.archive_p, config.seed)
            # VecNormalize is outermost in both stacks, but sync_envs_normalization walks on
            # down both and raises at the first checkpoint unless the eval stack has a wrapper
            # wherever the training stack does. Its env is built without archive: it reports no
            # cells and ignores a next start, so evaluation always starts from the start state.
            eval_venv = ArchiveResets(eval_venv, config.archive_p, config.seed)
        log = ParquetLog(venv, config, run_dir, start)
        env = log
        eval_env = ParquetLog(eval_venv, config, run_dir, start, training=log)
    else:
        env = make_vec_env(config.env_id, n_envs=config.n_envs, seed=config.seed, **SPAWN)
        eval_env = make_vec_env(config.env_id, n_envs=1, seed=config.seed + EVAL_SEED_OFFSET)
    if config.norm_obs or config.norm_reward:
        env = VecNormalize(
            env, norm_obs=config.norm_obs, norm_reward=config.norm_reward, gamma=config.ppo.gamma
        )
        eval_env = VecNormalize(
            eval_env, training=False, norm_obs=config.norm_obs, norm_reward=False
        )
    model = build_model(config, env)

    outputs: list[KVWriter] = [CSVOutputFormat(str(run_dir / "progress.csv"))]
    if config.track:
        outputs.append(
            WandbOutput(
                project="algoarcade", name=run_id, dir=run_dir, config=dataclasses.asdict(config)
            )
        )
    logger = Logger(str(run_dir), outputs)
    model.set_logger(logger)

    fixed = run_row(config, run_id, backend, rom_sha1, git_info["commit"])
    if game:
        write_run_row(run_dir, fixed | {"status": "running", "env_steps": 0, "wall_clock_s": 0.0})
    status = "crashed"
    try:
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=CheckpointEval(run_dir, config, eval_env, resets),
        )
        status = "finished"
    except EOFError as error:
        # close() would wait forever on the workers that already answered this step.
        for vec in (env.unwrapped, eval_env.unwrapped):
            if isinstance(vec, SubprocVecEnv):
                for process in vec.processes:
                    process.terminate()
                vec.closed = True
        raise RuntimeError("A SubprocVecEnv worker died; its traceback is above") from error
    finally:
        if game:
            elapsed = time.time() - start
            end = {"status": status, "env_steps": model.num_timesteps, "wall_clock_s": elapsed}
            write_run_row(run_dir, fixed | end)
        logger.close()
        env.close()
        eval_env.close()
    return run_dir


def main() -> None:
    print(train(tyro.extras.overridable_config_cli(PRESETS)))  # noqa: T201


# SubprocVecEnv with spawn re-imports this module in every worker.
if __name__ == "__main__":
    main()
