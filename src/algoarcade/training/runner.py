"""Train Stable-Baselines3 PPO on a Gymnasium env id. Contract: docs/technical-plan.md, Training.

    uv run python -m algoarcade.training.runner cartpole --seed 0

Episodes end through Gymnasium's TimeLimit, which SB3 reports as TimeLimit.truncated and
bootstraps with the value of the final observation. Do not wrap the env in anything that turns a
truncation into a termination.
"""

import csv
import dataclasses
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import tyro
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import CSVOutputFormat, KVWriter, Logger
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv

from algoarcade.training.config import PRESETS, TrainConfig, save_config

# Training envs take seeds seed .. seed + n_envs - 1; the eval env sits clear of them.
EVAL_SEED_OFFSET = 1_000
EVAL_FIELDS = ["step", "mean_return", "std_return", "wall_time_s"]


def build_model(config: TrainConfig, env: VecEnv) -> PPO:
    return PPO("MlpPolicy", env, seed=config.seed, verbose=0, **dataclasses.asdict(config.ppo))


class CheckpointEval(BaseCallback):
    """Every `every` env steps and at the end: save a checkpoint, evaluate it, append to eval.csv.

    The eval env is reseeded before each evaluation, so every checkpoint plays the same episodes.
    """

    def __init__(
        self, run_dir: Path, every: int, eval_env: VecEnv, eval_seed: int, episodes: int
    ) -> None:
        super().__init__()
        self.run_dir = run_dir
        self.every = every
        self.eval_env = eval_env
        self.eval_seed = eval_seed
        self.episodes = episodes
        self.next_at = every
        self.last_at = 0
        self.start = time.perf_counter()

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_at:
            self.next_at += self.every
            self.checkpoint()
        return True

    def _on_training_end(self) -> None:
        if self.last_at != self.num_timesteps:
            self.checkpoint()

    def checkpoint(self) -> None:
        self.last_at = self.num_timesteps
        self.model.save(self.run_dir / "checkpoints" / f"step_{self.num_timesteps}.zip")
        self.eval_env.seed(self.eval_seed)
        returns, _ = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=self.episodes,
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
                    "wall_time_s": round(time.perf_counter() - self.start, 1),
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


def train(config: TrainConfig) -> Path:
    run_id = f"{config.env_id}-seed{config.seed}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = config.runs_dir / f"run_id={run_id}"
    (run_dir / "checkpoints").mkdir(parents=True)
    save_config(config, run_dir / "config.json")
    git_info = {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    (run_dir / "git.json").write_text(json.dumps(git_info) + "\n")

    env = make_vec_env(
        config.env_id,
        n_envs=config.n_envs,
        seed=config.seed,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "spawn"},
    )
    eval_seed = config.seed + EVAL_SEED_OFFSET
    eval_env = make_vec_env(config.env_id, n_envs=1, seed=eval_seed)
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

    callback = CheckpointEval(
        run_dir, config.checkpoint_every, eval_env, eval_seed, config.eval_episodes
    )
    model.learn(total_timesteps=config.total_timesteps, callback=callback)
    logger.close()
    env.close()
    eval_env.close()
    return run_dir


def main() -> None:
    print(train(tyro.extras.overridable_config_cli(PRESETS)))  # noqa: T201


# SubprocVecEnv with spawn re-imports this module in every worker.
if __name__ == "__main__":
    main()
