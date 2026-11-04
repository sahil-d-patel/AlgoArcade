# envs

`AlgoArcadeEnv` in `env.py`, the Gymnasium environment: a backend, a reader, a reward registry, a start state, a step budget and the game's feature table. Archive resets follow in ENV-09.

Owner: Rishi.

`scripts/env_trace.py` runs one seeded 500-step trace and prints its hash, so two processes or two machines can be compared.

Contract: `docs/harness-spec.md`, AlgoArcadeEnv.
