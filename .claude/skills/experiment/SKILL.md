---
name: experiment
description: Rules for defining, launching and reporting training runs in this repo. Use when writing an experiment definition, choosing hyperparameters or seeds, starting a run locally or on the training box, comparing two runs, or writing up a result with a number in it.
paths: src/algoarcade/training/**, src/algoarcade/runqueue/**, experiments/**
---

# Running an experiment

A result we cannot reproduce or compare is not a result. These rules exist so that every number in the writeup traces to runs that differ in exactly one way.

Change one factor. An experiment names the baseline config and the single factor it varies. If you need to change two things, that is two experiments.

Set gamma explicitly. The library default of 0.99 gives a horizon of about 100 steps, and our episodes are thousands of steps long. An agent at the default does not crash. It does not learn. We start at 0.997.

Use five seeds for anything you will claim, on the same seed set as the arm you compare against. Three seeds is exploratory and must be labeled that way. Report the interquartile mean with a bootstrap interval, and print the seed count next to it.

Treat a step-budget end as truncation, not termination, and bootstrap the value there.

Every run records its seed, git commit, ROM SHA-1, reward config hash and resolved config in its run directory. Commit your code before you launch, or the commit hash means nothing.

Evaluate from the fixed start state with the archive off, whatever the training used.

Debug locally with one environment and 1,000 steps. Nothing interactive happens on the training box. Once the queue exists, a run on the box goes through the queue, never a terminal.

Never change the logging schema to suit a run. It freezes on November 1.

Before you believe a curve, watch the agent. Check the per-component reward chart, check that unique tiles rise with return, and watch five minutes of video.

A negative result is a result. Write down what you ran and what happened, including the arm that lost.

## The runner

`src/algoarcade/training/runner.py` trains Stable-Baselines3 PPO from a preset in `config.py`. Any field can be overridden on the command line; `--help` lists them.

```
uv run python -m algoarcade.training.runner cartpole --seed 0
uv run python -m algoarcade.training.runner acrobot --seed 0
uv run python -m algoarcade.training.runner baseline --env-id <any registered env id> --seed 0
```

Debug run on CartPole: add `--total-timesteps 1000 --n-envs 1 --checkpoint-every 500 --ppo.batch-size 32 --runs-dir <scratch dir>`. `--track` sends SB3's scalars to Weights & Biases; it is off by default.

Each run writes one directory:

```
data/runs/run_id=<env_id>-seed<seed>-<YYYYmmdd-HHMMSS>/
├── config.json               resolved config; load_config() reads it back
├── git.json                  {"commit": ..., "dirty": ...}
├── checkpoints/step_<n>.zip  every checkpoint_every env steps, and after the last update
├── eval.csv                  one row per checkpoint: step, mean_return, std_return, wall_time_s
└── progress.csv              SB3's rollout and train scalars
```

Each `eval.csv` row is `eval_episodes` (default 10) deterministic episodes on an env reseeded to `seed + 1000` first, so every checkpoint plays the same episodes. The last row is the final result. `uv run python scripts/summarize_runs.py --out-dir <dir>` prints the table of runs and saves one learning-curve PNG per env.

## FireRed and the fake game

With `env_id` set to `firered` or `fake`, the runner trains through `AlgoArcadeEnv` with MultiInputPolicy and 16 SubprocVecEnv workers (spawn), each building its own emulator. Evaluation runs in a seventeenth process. The `firered` preset is the baseline at 1M steps with explore as the only reward (`--reward-weights` changes it). The ROM is `data/roms/firered.gba` and the start state comes from the manifest; the runner checks both hashes before starting workers.

The TRN-05 runs, launched one after another so a stop loses only the run in progress:

```
for state in seg_01_house seg_02_pallet; do
  for seed in 0 1 2; do
    WANDB_MODE=offline uv run python -m algoarcade.training.runner firered \
      --start-state $state --seed $seed --track
  done
done
```

Debug first with `--n-envs 1 --total-timesteps 1024 --checkpoint-every 512 --ppo.n-steps 512 --ppo.batch-size 64 --eval-episodes 1 --runs-dir <scratch dir>`. On macOS, do not launch through `/usr/bin/time` or `nohup`: they drop `DYLD_FALLBACK_LIBRARY_PATH` (docs/setup.md), and every worker dies at start as an `EOFError`. A worker that dies prints its traceback to stderr; the parent only sees `EOFError`.

Besides the files above, a game run writes:

```
├── checkpoints/step_<n>.pkl  VecNormalize statistics for step_<n>.zip
├── steps/ visits/ episodes/ runs/   Parquet through algoarcade.logs.writer
└── wandb/                    with --track; WANDB_MODE=offline keeps it local
```

To play a checkpoint, load both files and freeze the statistics:

```python
env = VecNormalize.load("step_<n>.pkl", DummyVecEnv([env_fn]))
env.training = False
model = PPO.load("step_<n>.zip")
```

where `env_fn` comes from `algoarcade.training.games.env_factory`. Until ENV-08 and ENV-13 land, FireRed logs `story_flags` and `milestone_count` as -1 and `in_battle` as False: the reader does not provide them.

Super Mario Land is not wired in. It needs PyBoyBackend and its reader (ENV-03, #58), the ROM, a reward for `level_progress`, and a place to log `level_progress`: the steps table has no column for it.

The experiment table and the compute budget are in `docs/technical-plan.md` under "Experiments". TRN-03 owns this skill.
