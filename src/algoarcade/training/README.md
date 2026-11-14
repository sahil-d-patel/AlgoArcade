# training

The training runner, our PPO and PQN trainers, RND, and the statistics module.

`runner.py` trains Stable-Baselines3 PPO from a preset in `config.py` (TRN-03). On FireRed, Super Mario Land and the fake game (TRN-05) it builds the env from `games.py` and logs to Parquet through `parquet.py`. The command and the run directory layout are in `.claude/skills/experiment/SKILL.md`. `ppo.py` is our own PPO, derived from CleanRL's `ppo.py` (TRN-07). It takes the same presets and writes the same run directory, and its checkpoints load with `PPO.load` and `VecNormalize.load` like the runner's: `uv run python -m algoarcade.training.ppo cartpole --seed 0`. Its run ids carry `-owned-`. `scripts/parity.py` draws the parity gate against Stable-Baselines3. The statistics module, `stats.py`, arrives with OPS-05 (Akshaj).

Contract: `docs/technical-plan.md`, Training. Gamma is always an explicit config field. Five seeds for any headline claim.
