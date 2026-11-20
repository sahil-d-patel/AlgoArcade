---
name: experiment
description: Rules for defining, launching and reporting training runs in this repo. Use when writing an experiment definition, choosing hyperparameters or seeds, starting a run locally or on the training box, comparing two runs, or writing up a result with a number in it.
paths: src/algoarcade/training/**, src/algoarcade/runqueue/**, experiments/**
---

# Running an experiment

A result we cannot reproduce or compare is not a result. These rules exist so that every number in the writeup traces to runs that differ in exactly one way.

Change one factor from the baseline. The baseline is `BASELINE_V1` in `src/algoarcade/training/config.py` (preset `baseline_v1`): the `firered` preset trained for 15M steps, with a checkpoint every 500,000 steps and 20 sampled evaluation episodes every 1.5M steps and at the end. An experiment imports it by name, `from algoarcade.training.config import BASELINE_V1`, and changes one field. If you need to change two things, that is two experiments. A test pins every field, so changing the baseline fails CI until the test changes too; say why in the pull request.

Set gamma explicitly. The library default of 0.99 gives a horizon of about 100 steps, and our episodes are thousands of steps long. An agent at the default does not crash. It does not learn. We start at 0.997.

Use five seeds for anything you will claim, on the same seed set as the arm you compare against. Fewer than five is exploratory and must be labeled that way. Report every result as the interquartile mean over seeds with its 95% percentile bootstrap interval and the seed count, n. `algoarcade.training.stats.describe(scores)` prints it as `IQM 0.62 [0.48, 0.71] (n=5)` and warns below five seeds.

Treat a step-budget end as truncation, not termination, and bootstrap the value there.

Every run records its seed, git commit, ROM SHA-1, reward config hash and resolved config in its run directory. Commit your code before you launch, or the commit hash means nothing.

Evaluate from the fixed start state with the archive off, whatever the training used.

Debug locally with one environment and 1,000 steps. Nothing interactive happens on the training box. A run on the box goes through the run queue, never a terminal.

Never change the logging schema to suit a run. It freezes on November 8.

Before you believe a curve, watch the agent. Check the per-component reward chart, check that unique tiles rise with return, and watch five minutes of video.

A negative result is a result. Write down what you ran and what happened, including the arm that lost.

## The runner

`src/algoarcade/training/runner.py` trains Stable-Baselines3 PPO from a preset in `config.py`. Any field can be overridden on the command line; `--help` lists them.

```
uv run python -m algoarcade.training.runner cartpole --seed 0
uv run python -m algoarcade.training.runner acrobot --seed 0
uv run python -m algoarcade.training.runner baseline --env-id <any registered env id> --seed 0
```

Debug run on CartPole: add `--total-timesteps 1000 --n-envs 1 --checkpoint-every 500 --eval-every 500 --ppo.batch-size 32 --runs-dir <scratch dir>`. `--eval-every` must be a multiple of `--checkpoint-every`; set them equal to evaluate every checkpoint. `--track` sends SB3's scalars to Weights & Biases; it is off by default.

Each run writes one directory:

```
data/runs/run_id=<env_id>-seed<seed>-<YYYYmmdd-HHMMSS>/
├── config.json               resolved config; load_config() reads it back
├── git.json                  {"commit": ..., "dirty": ...}
├── checkpoints/step_<n>.zip  every checkpoint_every env steps, and after the last update
├── eval.csv                  one row per evaluated checkpoint: step, mean_return, std_return, wall_time_s
└── progress.csv              SB3's rollout and train scalars
```

A run from the queue is `run_id=<experiment>-<arm>-seed<n>-<YYYYmmdd-HHMMSS>` and has `experiment_id` and `arm` set in its runs table.

Each `eval.csv` row is `eval_episodes` (default 10) episodes on an env reseeded to `seed + 1000` first. A checkpoint is evaluated when its step is a multiple of `eval_every`, and the final checkpoint always is. With `eval_deterministic` True the policy takes its argmax, which from a segment's start state replays one episode. `BASELINE_V1` sets it False: actions are sampled from torch's generator seeded to `seed + 1000` (`runner.eval_rng`), so evaluation never moves training and `algoarcade.viz.render --eval-episode k` replays episode k. The last row is the final result. `uv run python scripts/summarize_runs.py --out-dir <dir>` prints the table of runs and saves one learning-curve PNG per env.

## The games

With `env_id` set to `firered`, `super_mario_land` or `fake`, the runner trains through `AlgoArcadeEnv` with MultiInputPolicy and 16 SubprocVecEnv workers (spawn), each building its own emulator. Evaluation runs in a seventeenth process. The `firered` preset uses `BASELINE`'s PPO values for 1M steps on the default reward: events, explore, level, badge and milestone at `DEFAULT_WEIGHTS`, penalties off (`--reward-weights` changes it). The ROM is `data/roms/firered.gba` and the start state comes from the manifest; the runner checks both hashes before starting workers, and each worker checks the state's options.

The TRN-05 runs, launched one after another so a stop loses only the run in progress:

```
for state in seg_01_house seg_02_pallet; do
  for seed in 0 1 2; do
    WANDB_MODE=offline uv run python -m algoarcade.training.runner firered \
      --start-state $state --seed $seed --track
  done
done
```

Debug first with `--n-envs 1 --total-timesteps 1024 --checkpoint-every 512 --eval-every 512 --ppo.n-steps 512 --ppo.batch-size 64 --eval-episodes 1 --runs-dir <scratch dir>`; without `--eval-every 512` the preset's `eval_every` is not a multiple of 512 and the config is rejected. On macOS, do not launch through `/usr/bin/time` or `nohup`: they drop `DYLD_FALLBACK_LIBRARY_PATH` (docs/setup.md), and every worker dies at start as an `EOFError`. A worker that dies prints its traceback to stderr; the parent only sees `EOFError`.

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

where `env_fn` comes from `algoarcade.training.games.env_factory`. The episodes table's `badge_count`, `party_level_sum` and `story_flags` are each the latest value two consecutive reads agreed on, because a read on the frame a save block moves can be wrong; `milestone_count` is the env's episode best.

With `env_id` set to `super_mario_land`, the runner trains Super Mario Land on PyBoy from the start state `level_1_1` (regenerate it with `uv run python scripts/capture_sml_states.py`). The `super_mario_land` preset uses the FireRed PPO values for 2M steps with `explore` at 0.02 as the only reward. A game over continues from the start state instead of showing the title menu and its demo (docs/harness-spec.md, Reward registry). The steps table's `x` holds `x_tile`, which is `level_progress` at 8-pixel resolution; use it for progress. Rows log `in_battle` False and the FireRed counts as 0. The TRN-05 runs:

```
for seed in 0 1 2; do
  WANDB_MODE=offline uv run python -m algoarcade.training.runner super_mario_land --seed $seed --track
done
```

`--archive` starts training episodes from cells the run already reached (ENV-09; docs/harness-spec.md, Archive resets); `--archive-p` is the chance per reset, default 0.5. It is off in the baseline, and evaluation never uses it. progress.csv gains `rollout/archive_share`, `rollout/archive_cells` and `rollout/archive_bytes_per_cell` (zlib-compressed), and the episodes table fills `archive_start`.

## The run queue

On the box, runs go through `algoarcade.runqueue`. An experiment file under `experiments/` sets `EXPERIMENT`, `BASE` (a `TrainConfig`), `ARMS` (arm name to the fields it replaces) and `SEEDS`; commit it, then:

```
uv run python -m algoarcade.runqueue submit experiments/<file>.py --dry-run
uv run python -m algoarcade.runqueue submit experiments/<file>.py
uv run python -m algoarcade.runqueue status
```

The nightly task runs `uv run python -m algoarcade.runqueue run`. A crash is retried once from step 0, then marked failed and posted to the channel. A run whose config sets `second_value_head`, which RND's `int_coef` needs, trains on the owned trainer (`algoarcade.training.ppo`); every other run on the SB3 runner. Set `track=True` in `BASE` for W&B, where the experiment id is the group. The file format, queue directory, webhook variable and nightly task are in docs/setup.md, Training box.

The experiment table and the compute budget are in `docs/technical-plan.md` under "Experiments". TRN-03 owns this skill.
