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

## ROMs

We do not share ROMs. Each member supplies their own dumps and puts them in `data/roms/`, which git ignores. Create the folder if it is not there. Save states go in `data/states/` the same way.

Check that yours is a revision the project supports:

```sh
uv run python -m algoarcade.rom data/roms/firered.gba
```

It prints the hash and the name, or exits with an error that lists the hashes we accept. FireRed (USA) rev 0 is the target and rev 1 also works. Super Mario Land joins the table when ENV-03 lands.

The harness runs the same check every time it loads a ROM. A wrong revision would not crash. It would read the wrong addresses and produce believable, wrong rewards.

## The mGBA binding

Not run yet. ENV-01 lands the fetch script and ENV-04 proves it on all three systems. This is what it will be:

| OS | First | Then |
|---|---|---|
| Windows x64 | Nothing | `uv run python scripts/fetch_mgba.py` |
| macOS (Apple Silicon) | `brew install mgba` | `uv run python scripts/fetch_mgba.py` |
| Ubuntu 24.04 or later | `sudo apt install libmgba0.10t64` | `uv run python scripts/fetch_mgba.py` |

Ubuntu 22.04 only packages an older libmgba, so use 24.04 or later. Nobody compiles mGBA. If the binding fails on your machine, post the error on ENV-01 (or ENV-04 after week 1) instead of working around it.

Until then, everything that does not need FireRed runs on the fake backend, which needs no ROM.

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

We train on two machines: a Windows desktop (the box) and a Mac. You reach the box over Tailscale, a private network, with the SSH server that ships with Windows.

1. Install [Tailscale](https://tailscale.com/download) and sign in. Accept the share Sahil sends you for the box.
2. If you have no SSH key, make one with `ssh-keygen -t ed25519`. Send Sahil the `.pub` file it prints the path of. Never send the file without `.pub`.
3. Once Sahil confirms your key is added, connect with `ssh <user>@<box>`, using the user and machine name from the share. `hostname` should print the box's name.

The shell on the box is Windows `cmd`. The repo is cloned in the home folder, and runs write to its `data\runs\`. Copy results off with `scp -r <user>@<box>:AlgoArcade/data/runs/<run_id> .`

Three rules, from the first day:

- Nobody debugs on the box. Debug on your own machine with one environment and 1,000 steps.
- Runs go through the queue once OPS-04 lands. Do not launch runs by hand after that.
- A stray process costs a night of training. If you started something, stop it before you log out.

## Gamma

> **Never leave gamma at the Stable-Baselines3 default of 0.99.** The effective horizon is about 1 / (1 - gamma), which is 100 steps at 0.99. Our segments run 2,000 to 8,000 steps. An agent at the default cannot value its goal. It throws no error. It does not learn. We start at 0.997, and gamma is always an explicit field in the config.
