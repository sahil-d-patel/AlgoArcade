# training

The training runner, our PPO and PQN trainers, RND, and the statistics module.

`runner.py` trains Stable-Baselines3 PPO from a preset in `config.py` (TRN-03). On FireRed, Super Mario Land and the fake game (TRN-05) it builds the env from `games.py` and logs to Parquet through `parquet.py`. `--archive` starts training episodes from `archive.py`'s archive of reached cells (ENV-09; docs/harness-spec.md, Archive resets). The command and the run directory layout are in `.claude/skills/experiment/SKILL.md`. Our own PPO trainer follows in TRN-07, behind a parity gate against Stable-Baselines3. The statistics module, `stats.py`, arrives with OPS-05 (Akshaj).

Contract: `docs/technical-plan.md`, Training. Gamma is always an explicit config field. Five seeds for any headline claim.
