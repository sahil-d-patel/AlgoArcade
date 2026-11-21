"""Env steps per second in each obs_mode under a uniform random policy (ENV-12).

    uv run python scripts/obs_throughput.py firered --start-state seg_02_pallet --n-envs 16

The envs run in the runner's SubprocVecEnv with spawn, and no policy runs, so the difference
between modes is rendering, the greyscale resize and sending the frames between processes.
"""

import argparse
import os
import platform
import time
from typing import get_args

import numpy as np
from stable_baselines3.common.env_util import make_vec_env

from algoarcade.envs.env import ObsMode
from algoarcade.training.config import FIRERED, SUPER_MARIO_LAND
from algoarcade.training.games import env_factory
from algoarcade.training.runner import SPAWN

WARMUP = 32


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("env_id", choices=("fake", "firered", "super_mario_land"))
    parser.add_argument("--start-state", default="")
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=500, help="vector steps timed per mode")
    args = parser.parse_args()
    weights = (SUPER_MARIO_LAND if args.env_id == "super_mario_land" else FIRERED).reward_weights
    print(f"{platform.platform()}, {platform.machine()}, {os.cpu_count()} CPUs")
    rng = np.random.default_rng(0)
    for obs_mode in get_args(ObsMode):
        make, _, _ = env_factory(args.env_id, args.start_state, weights, obs_mode)
        venv = make_vec_env(make, n_envs=args.n_envs, seed=0, **SPAWN)
        venv.reset()
        for _ in range(WARMUP):
            venv.step(rng.integers(venv.action_space.n, size=args.n_envs))  # type: ignore[attr-defined]
        start = time.perf_counter()
        for _ in range(args.steps):
            venv.step(rng.integers(venv.action_space.n, size=args.n_envs))  # type: ignore[attr-defined]
        elapsed = time.perf_counter() - start
        venv.close()
        rate = args.steps * args.n_envs / elapsed
        print(f"{obs_mode:<8} {args.n_envs} envs  {rate:,.0f} env steps/s")


# SubprocVecEnv with spawn re-imports this module in every worker.
if __name__ == "__main__":
    main()
