# training

The training runner, our PPO and PQN trainers, RND, and the statistics module.

`runner.py` trains Stable-Baselines3 PPO from a preset in `config.py` (TRN-03). On FireRed, Super Mario Land and the fake game (TRN-05) it builds the env from `games.py` and logs to Parquet through `parquet.py`. The command and the run directory layout are in `.claude/skills/experiment/SKILL.md`. Our own PPO trainer follows in TRN-07, behind a parity gate against Stable-Baselines3. `stats.py` (OPS-05) holds the numbers every comparison reports: `describe(scores)` prints the IQM over seeds with its 95% bootstrap interval and the seed count, as in `IQM 0.62 [0.48, 0.71] (n=5)`. It also has the probability of improvement and the performance profile. Every estimate warns that fewer than five seeds is exploratory, and an IQM of fewer than four scores warns that it equals the mean.

Contract: `docs/technical-plan.md`, Training. Gamma is always an explicit config field. Five seeds for any headline claim.
