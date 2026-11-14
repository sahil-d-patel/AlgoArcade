"""Re-score a checkpoint: play its evaluation episodes as runner.evaluate() does and count goals.

    uv run python scripts/rescore_eval.py data/runs/run_id=<id> --step 1015808 --sampled

The env and the sampled actions start from the eval seed, through runner.eval_rng, so episode k
here is the episode `algoarcade.viz.render --eval-episode k` replays. TRN-08 scored the 1M-step
runs' final checkpoints with 20 sampled episodes this way (docs/decisions.md, 2026-11-14). Nothing
is written to the run directory.
"""

import dataclasses
from pathlib import Path

import tyro
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from algoarcade.training.config import load_config
from algoarcade.training.games import env_factory
from algoarcade.training.runner import EVAL_SEED_OFFSET, eval_rng


def main(run: Path, step: int, episodes: int = 20, sampled: bool = False) -> None:
    config = dataclasses.replace(load_config(run / "config.json"), eval_deterministic=not sampled)
    env_fn, _, _ = env_factory(config.env_id, config.start_state, config.reward_weights)
    checkpoint = run / "checkpoints" / f"step_{step}"
    env = VecNormalize.load(str(checkpoint.with_suffix(".pkl")), DummyVecEnv([env_fn]))
    env.training = False
    env.seed(config.seed + EVAL_SEED_OFFSET)
    model = PPO.load(checkpoint.with_suffix(".zip"), device="auto")
    goals = 0
    obs = env.reset()
    with eval_rng(config):
        for _ in range(episodes):
            reached, done = False, False
            while not done:
                action, _ = model.predict(obs, deterministic=config.eval_deterministic)
                obs, _, dones, infos = env.step(action)
                reached |= bool(infos[0]["goal"])
                done = bool(dones[0])
            goals += reached
    mode = "sampled" if sampled else "deterministic"
    print(f"{run.name} step {step} {mode}: {goals}/{episodes} reached the goal")


if __name__ == "__main__":
    tyro.cli(main)
