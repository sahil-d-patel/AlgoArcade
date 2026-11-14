# envs

`AlgoArcadeEnv` in `env.py`, the Gymnasium environment: a backend, a reader, a reward registry, a start state, a step budget and the game's feature table. With `archive=True` it reports the cells an episode reaches and loads one on the next reset after `set_next_start`; the archive itself is in `training/archive.py`.

Owner: Rishi.

`scripts/env_trace.py` runs one seeded 500-step trace and prints its hash, so two processes or two machines can be compared.

Contract: `docs/harness-spec.md`, AlgoArcadeEnv.
