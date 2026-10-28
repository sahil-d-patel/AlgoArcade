# Decisions

The design decisions behind the plan, and why we made each one. Add a dated line to the log whenever a deliverable is cut, a gate is missed, or a stack choice changes.

Sources are listed at the end of [technical-plan.md](technical-plan.md).

## Design decisions

| Area | Decision | Why |
|---|---|---|
| Goals | The Brock runs and the second game are on the schedule | We would rather aim high and report a miss than plan a shallow result. |
| Team | Two floating PMs, two on Environment, two on Training, one on the LLM agent | Environment is the critical path. The LLM arm is decoupled, so one engineer owns it with Akshaj as backup. |
| Visualization | Built in weeks 6 and 7 by everyone | Before that, every hour goes to the harness and the experiments. The schema freeze and the fake logger make building late safe. |
| GBA binding | Prebuilt libmgba-py archives, fetched and checksummed by a script | No compile on any OS. The `mgba` wheel on PyPI has no Windows or Apple Silicon build. |
| Install | Native, with uv. No container. | Nothing needs compiling, so an image would only add a layer. |
| Second backend | Super Mario Land on PyBoy, from week 2 | It is the non-Pokémon game, so the framework claim is tested all semester. Its dense reward gives the Training pair a learning curve in week 3. |
| Test backend | A ROM-free fake backend | CI can run every test that does not need a ROM, and three lanes can start before the emulator works. |
| Addresses | Symbol files published by the decompilation's CI, pinned to a commit | The linker's own output, with no toolchain to install. Never a wiki. |
| Progress signal | A whitelist of story flags, trainer flags and the badge byte | Temporary flags reset on every map load, so a count of all flags goes down as well as up. |
| Start state | S1 (after the parcel, with Squirtle) for development; S0 (the bedroom) for the cold-start runs | Published Red agents start after the parcel because the backtrack blocked learning. Squirtle wins the Brock fight far more often than Charmander. |
| Actions | Six buttons by default; Start and a no-op optional | Start opens SAVE and OPTIONS, and nothing before Brock needs it. |
| Rewards | Five positive-only components on; penalties and heal off | Published and open-source results agree that penalties and heal rewards backfire. Whether that holds on FireRed is experiment E5. |
| Value-based arm | PQN | Built for parallel environments, with no replay buffer, and it consumes the same rollouts as PPO. |
| Exploration | Count-based against RND | Counts are the proven baseline, and RND on Pokémon is untested in the literature. ICM is the first arm to add if time allows. |
| Frontier return | Archive resets | Needs only save and load, which every backend has. |
| Trainer | Stable-Baselines3 for weeks 1 to 3, then our own PPO behind a parity gate | RND needs two value heads and PQN is not in the library. If the gate fails on November 8 we stay on Stable-Baselines3. |
| Gamma | 0.997 | The value published Red projects tuned to. Never the 0.99 default. |
| Statistics | IQM and bootstrap intervals implemented in the repo over scipy | The reference library is archived, and the metrics are a few lines. |
| Configs | Frozen dataclasses with a generated command line | Type-checked, with nothing hidden in YAML. |
| Map data | World coordinates and walkability generated from the decompilation's map files | One source serves the symbolic observation, the LLM's pathfinding and the heatmap. |
| Heatmap base | A schematic drawn from collision data | The game's tile art is not ours to publish. |
| Logs | Sampled per-step rows plus an exact per-tile visit count | Sampled rows cannot give exact heatmap counts. |
| LLM tools | Four tools in three levels | The tool ladder is the experiment. |
| LLM evaluation | Ten isolated attempts per configuration, with intervals and cost per success | LLM runs vary, and one attempt is an anecdote. |
| Dashboard | Quarto, rendered to static HTML | It has to work at the showcase with no server. |
| Tracking | Weights & Biases for the live view, Parquet as the source of truth, Trackio as the fallback | Every published figure is rebuilt from files we hold. |

## Project setup

Settled on October 11, 2026.

- The repository is public. Rulesets on `main` are enforced, and CI runs on Linux, Windows and macOS.
- Every issue gets one branch cut from `main` and one pull request back into `main`.
- Only the PMs have write access. Members work from forks and open pull requests into `main`; the PMs are the code owners and merge.
- Training runs on a Windows desktop (the training box) and a Mac.
- Save states live in `data/states/` on each machine and are handed to members directly. They never go in git.
- Sahil leads and Akshaj co-leads, both floating. Rishi and Prince are on Environment, Isaac and Pri on Training, Jayden on Agent. Each person has one assigned issue a week.

## Choices a PM can veto

AGENTS.md treats names, dependencies and architecture as PM decisions. These were made to get the repo ready for kickoff.

- Names: the package `algoarcade`; its subpackages `backends`, `games`, `rewards`, `envs`, `logs`, `training`, `runqueue`, `agent`, `viz`; the issue id prefixes OPS, ENV, TRN, AGT, VIZ, DOC.
- Dependencies: numpy, gymnasium, pyboy, pyarrow, tyro, cffi; for training torch, stable-baselines3, wandb; for the agent mcp, anthropic; for development ruff, pyrefly, pytest, poethepoet, prek, typos.
- Tooling: Python 3.12, uv with a committed lockfile.
- Scope: the repo ships the backend protocol, the fake backend and the conformance suite already written, so week 1 has something to build against. They are drafts until the November 1 freeze, and the Environment pair owns them from here.

## Log

- 2026-10-11: Plan fine-tuned before kickoff.
- 2026-10-27: Two Super Mario Land addresses have no label in the kaspermeerts/supermarioland disassembly (commit 618d00e): the level block at 0xC0AB and Mario's screen x at 0xC202. The reader cites PyBoy 2.8.1's `game_wrapper_super_mario_land.py` for both (ENV-03).
