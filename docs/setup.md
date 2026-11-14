# Setup

From a fresh clone to a working environment. Commands marked "not run yet" depend on an issue that has not landed.

## Install

You need git and [uv](https://docs.astral.sh/uv/getting-started/installation/). uv installs the right Python for you.

Fork https://github.com/sahil-d-patel/AlgoArcade on GitHub, then:

```sh
git clone https://github.com/<your-username>/AlgoArcade.git
cd AlgoArcade
git remote add upstream https://github.com/sahil-d-patel/AlgoArcade.git
uv sync
uv run poe check
uv run prek install
```

Members work from a fork because only the PMs have write access. `origin` is your fork and `upstream` is the team repository. PMs can clone the team repository directly.

- `uv sync` creates `.venv` with Python 3.12 and everything in the lockfile, including PyTorch. The first run downloads a few hundred megabytes.
- `uv run poe check` should end with every step passing. If it does not, your setup is broken and nothing else will work. Ask in the channel.
- `uv run prek install` adds the git hook that runs the same checks on each commit.

Run every project command through `uv run`. Do not activate the virtual environment and do not `pip install` into it.

### The mGBA binding

Install your OS's system packages for the mGBA binding, then run the fetch script, which installs the pinned binding into the project environment. Run on macOS arm64 and on Ubuntu 24.04 x64 (in a container); Windows is not run yet.

| OS | First | Then |
|---|---|---|
| Windows x64 | Nothing | `uv run python scripts/fetch_mgba.py` |
| macOS (Apple Silicon) | `brew install mgba ffmpeg@8`, then the export below | `uv run python scripts/fetch_mgba.py` |
| Ubuntu 24.04 or later | `sudo apt install libmgba0.10t64` | `uv run python scripts/fetch_mgba.py` |

On macOS, Homebrew's mgba is linked against ffmpeg 8 while Homebrew now ships ffmpeg 9, and the binding does not search Homebrew's library folder. Until Homebrew rebuilds mgba, put both folders on the fallback path in `~/.zshrc`:

```sh
export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib:/opt/homebrew/opt/ffmpeg@8/lib
```

dyld reads it only at process start, and macOS strips it when a system binary such as `/bin/sh` is in the launch chain, so run Python through `uv run`. Do not `brew link ffmpeg@8`.

Ubuntu 22.04 only packages an older libmgba, so use 24.04 or later. Nobody compiles mGBA. If the binding fails on your machine, post the error on ENV-01 (or ENV-04 after week 1) instead of working around it.

With the ROM at `data/roms/firered.gba`, check the backend with `uv run pytest tests/conformance -k mgba -q`; the FireRed tests skip when the file is absent. States are native mGBA bytes, with flash and RTC data and, when rendering is on, a screenshot. They are not battery saves. `MgbaBackend(..., render=False)` turns video off, and `screen()` then raises `RuntimeError`. After loading a state without a screenshot into a rendered backend, advance one frame before calling `screen()`.

## ROMs

We do not share ROMs. Each member supplies their own dumps and puts them in `data/roms/`, which git ignores. Create the folder if it is not there. Save states go in `data/states/` the same way; see [Save states](#save-states).

Check that yours is a revision the project supports:

```sh
uv run python -m algoarcade.rom data/roms/firered.gba
```

It prints the hash and the name, or exits with an error that lists the hashes we accept. FireRed (USA) rev 0 is the target and rev 1 also works. Super Mario Land joins the table when ENV-03 lands.

The harness runs the same check every time it loads a ROM. A wrong revision would not crash. It would read the wrong addresses and produce believable, wrong rewards.

## Save states

The FireRed start states are not downloadable: save states hold game data and the repository is public. Sahil (PM) sends the files directly. Put them in `data/states/` and check them against the manifest:

```sh
uv run python -m algoarcade.games.firered.manifest
```

It prints one line per state, `OK` or `MISSING` or `MISMATCH`, and exits non-zero unless all nine are OK. Never post the files anywhere public.

The states were captured on FireRed rev 1. With a rev 1 ROM you can regenerate them yourself instead; it takes about 90 seconds:

```sh
uv run python scripts/capture_states.py
```

The script replays recorded inputs from power-on, checks the location and the options after each stage, and writes the nine states plus the `_goal` states the tests use. On macOS arm64 with libmgba-py 0.2.0, the output hashes matched the manifest on every run; other platforms are not run yet. `--from gym_door` replays that stage and the ones after it, starting from the states already in the folder. `--out DIR` reads and writes `DIR` instead of `data/states/`, and the script writes `data/states/` only with the rev 1 ROM. To check a file, regenerate into a scratch folder and run `uv run python -m algoarcade.games.firered.manifest DIR`.

With a rev 0 ROM, use `--out`. The address table is the same for both revisions, but nobody has run the script on rev 0.

## Checks

`uv run poe check` runs, in order:

1. `ruff format --check`: formatting.
2. `ruff check`: lint.
3. `pyrefly check`: types.
4. `pytest`: tests. Tests that need a ROM are marked and skip when the ROM is absent.

Useful on their own:

```sh
uv run poe fmt                          # format in place
uv run pytest tests/conformance -q      # one folder
uv run pytest -k save_then_load         # tests matching a word
uv run poe items                        # validate the work item files
```

CI runs the same check on Linux, Windows and macOS for every pull request.

## Training box

We train on two machines: a Windows desktop (the box) and a Mac. OPS-01 sets up access. Three rules, from the first day:

- Nobody debugs on the box. Debug on your own machine with one environment and 1,000 steps.
- Runs go through the run queue. Do not launch runs by hand.
- A stray process costs a night of training. If you started something, stop it before you log out.

### The run queue

An experiment is a Python file under `experiments/`, committed before you submit it. It sets `EXPERIMENT` (the id), `BASE` (a `TrainConfig`), `ARMS` (each arm's name to the `TrainConfig` fields it replaces in `BASE`) and `SEEDS`:

```python
import dataclasses

from algoarcade.rewards.registry import DEFAULT_WEIGHTS, WEIGHTS
from algoarcade.training.config import PRESETS

EXPERIMENT = "E5"
BASE = dataclasses.replace(PRESETS["firered"][1], start_state="seg_02_pallet", track=True)
PENALTIES = ("blackout", "anti_loop", "anti_spam", "time")
ARMS = {
    "positive": {},
    "penalties": {"reward_weights": DEFAULT_WEIGHTS | {n: WEIGHTS[n] for n in PENALTIES}},
}
SEEDS = [0, 1, 2]
```

Several files can share an experiment id, so E1 to E4 can go in arm by arm as each arm's code lands. The commands, from the clone:

```sh
uv run python -m algoarcade.runqueue submit experiments/e5.py --dry-run   # print the runs only
uv run python -m algoarcade.runqueue submit experiments/e5.py
uv run python -m algoarcade.runqueue status
uv run python -m algoarcade.runqueue run
```

- `submit` queues one run per arm and seed, named `<experiment>-<arm>-seed<n>`: seed 0 of every arm, then seed 1, so a queue cut short leaves the arms with the same seeds. It refuses a name already in the queue. The config is fixed at submit. The code is whatever the clone has checked out when the run starts, and the run records that commit.
- `run` trains the queued runs in submit order, each in its own process, until none is left. A run that exits with an error is retried once, from step 0, then marked failed, and the next run starts. An attempt still running after 1 hour plus `total_timesteps` / 100 seconds (43 hours for 15M steps; `run --min-steps-per-s` changes the 100) is killed with its workers and counts as a crash. A second `run` exits at once while one is working. Ctrl+C stops the queue and puts the current run back without using up an attempt.
- `status` prints each run's status (queued, running, done or failed), attempts, start and end time, then a count per experiment.

Each attempt trains into `data/runs/run_id=<run>-<time>/` and writes its experiment id and arm to the runs table. With `track=True` the experiment id is also the run's W&B group; log in once on the box with `uv run wandb login`.

The queue lives in `data/runs/queue/`: `queue.json`, each run's config under `configs/`, and each attempt's output in `logs/<run>-<attempt>.log`. If the box reboots or `run` is killed mid-run, the next `run` counts that attempt as a crash. On Windows a trainer dies with its `run`, however `run` ends. On macOS and Linux, Ctrl+C or an error in `run` kills the trainer, but a `run` killed with SIGKILL leaves it training.

A failed run posts its experiment id, run name, the exception line of each traceback (each once) and the end of its log to the team channel, in at most 1,900 characters. The message carries both Discord's `content` and Slack's `text`. The webhook URL is a secret: set it as a user environment variable on the box, never in a file in the repo. The trainer does not get it, and a failed post prints only the error type. Without the variable, `run` prints a line and goes on.

```powershell
setx ALGOARCADE_WEBHOOK_URL "<webhook URL>"
```

`setx` changes only processes started after it. Set the variable before you register the task below, then sign out and back in so the task's next start sees it. Do the same after changing it.

`run` starts every night from Task Scheduler. Register the task once, in PowerShell from the clone:

```powershell
$action = New-ScheduledTaskAction -Execute (Get-Command uv).Source `
  -Argument "run python -m algoarcade.runqueue run" -WorkingDirectory (Get-Location).Path
$trigger = New-ScheduledTaskTrigger -Daily -At 1am
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) `
  -MultipleInstances IgnoreNew -StartWhenAvailable -Priority 4
Register-ScheduledTask -TaskName "AlgoArcade run queue" -Action $action -Trigger $trigger -Settings $settings
```

A zero `ExecutionTimeLimit` removes Task Scheduler's default three-day limit, which would stop the queue partway through a run. `-Priority 4` is normal priority; the default, 7, is below normal and would lower the steps per second the runs measure. The task runs only while you are signed in: lock the box, do not sign out. `Start-ScheduledTask "AlgoArcade run queue"` starts it now. Not run on the box yet.

## Gamma

> **Never leave gamma at the Stable-Baselines3 default of 0.99.** The effective horizon is about 1 / (1 - gamma), which is 100 steps at 0.99. Our segments run 2,000 to 8,000 steps. An agent at the default cannot value its goal. It throws no error. It does not learn. We start at 0.997, and gamma is always an explicit field in the config.
