# Port diary: Super Mario Land

Every file three pull requests touched to add Super Mario Land: ENV-03 (#58, the PyBoy backend and the reader), TRN-05 (#76, the start state, observation, reward table and runner wiring) and ENV-11 (#31, the button hold and E9). [adding-a-game.md](adding-a-game.md) turns it into steps.

## Totals

From `git log --no-merges --numstat` over each pull request's own commits, summed per file. A line changed in two commits counts twice. ENV-11's branch has `main` merged in, so its range follows the first parent. This diary is left out of its own count.

| Pull request | Range | Files | Added | Deleted |
|---|---|---|---|---|
| #58 ENV-03 | `13a0838^1..13a0838^2` | 13 | 532 | 130 |
| #76 TRN-05 | `c09abc7^1..c09abc7^2` | 16 | 458 | 49 |
| #31 ENV-11 | `27bcbe8..3c27a51`, first parent | 18 | 498 | 90 |
| All three | | 36 distinct | 1,488 | 269 |

Inside `src/algoarcade/games/` and `src/algoarcade/backends/pyboy.py`: 6 files, +265 −31. Outside: 30 files, +1,223 −238. Of the lines added outside, 696 are tests (10 files), 65 Markdown (8 files) and 462 code (12 files, the scripts and the experiment file included).

The commands, from the repo root:

```
git log --no-merges --numstat --format= 13a0838^1..13a0838^2
git log --no-merges --numstat --format= c09abc7^1..c09abc7^2
git log --first-parent --no-merges --numstat --format= 27bcbe8..3c27a51
```

## Inside the game's folder and the backend

| File | Pull request | Lines | What |
|---|---|---|---|
| `src/algoarcade/backends/pyboy.py` | #58 | +77 −7 | The backend, pinned to the rev A dump, with rendering switchable |
| `src/algoarcade/games/super_mario_land/reader.py` | #58, #76 | +110 −22 | The reader and its address table; #76 added the game-state and menu addresses |
| `src/algoarcade/games/super_mario_land/manifest.py` | #76 | +39 −0 | Start state `level_1_1`, its goal and budget |
| `src/algoarcade/games/super_mario_land/features.py` | #76 | +16 −0 | The observation vector |
| `src/algoarcade/games/super_mario_land/rewards.py` | #76 | +20 −0 | The reward table: existing components only |
| `src/algoarcade/games/README.md` | #58, #76 | +3 −2 | Lists the new files |

## Outside them, with the reason

| File | Pull request | Lines | Why it changed |
|---|---|---|---|
| `src/algoarcade/rom.py` | #58 | +5 −0 | Pins the dump's SHA-1; every ROM is pinned here |
| `src/algoarcade/backends/README.md` | #58 | +1 −1 | Lists the backend |
| `tests/conformance/test_backend.py` | #58 | +42 −1 | A new backend joins the conformance registry, plus the ROM-free probe CI runs |
| `tests/test_rom.py` | #58 | +14 −1 | The ROM guard for the new hash |
| `tests/test_super_mario_land.py` | #58 | +267 −79 | Reader tests and the PyBoy wrapper oracle |
| `scripts/env_trace.py` | #58 | +9 −7 | The env trace script's `pyboy` option imported the modules once they existed, and starts after the game's Init |
| `DATA.md` | #58 | +2 −2 | The ROM's hash was TBD in the source register |
| `docs/harness-spec.md` | #58, #76, #31 | +17 −12 | The location rule, start state, continue rule and the hold are contract |
| `docs/decisions.md` | #58, #76, #31 | +10 −5 | Two unlabelled addresses, the continue rule, the hold and E9's scoring |
| `tools/github/issues/ENV-03-pyboy-mario-land.md` | #58 | +1 −1 | The oracle criterion as approved on #10 |
| `scripts/capture_sml_states.py` | #76 | +44 −0 | Regenerates `level_1_1`; state files stay out of git, so each game needs its own capture script |
| `src/algoarcade/training/games.py` | #76, #31 | +161 −43 | The env factory and its `env_factory` entry; `Continues`, which reloads the start state after a game over; `Measures` taught to log 0 for FireRed counts the game lacks; the 16-frame hold and `game_hold` |
| `src/algoarcade/training/config.py` | #76, #31 | +25 −3 | The `super_mario_land` preset, FireRed's PPO values on this game; the `hold` field and its backfill for older configs |
| `src/algoarcade/training/runner.py` | #76, #31 | +23 −5 | The runner's list of game env ids, `rollout/continues` in `progress.csv`, and the hold it used in config.json and the runs table |
| `src/algoarcade/training/parquet.py` | #76 | +2 −1 | A comment: a continue does not end an episode |
| `src/algoarcade/training/README.md` | #76 | +1 −1 | Names the game |
| `src/algoarcade/viz/render.py` | #31 | +1 −1 | Replays a checkpoint at the hold it trained with |
| `tests/training/test_super_mario_land_training.py` | #76, #31 | +197 −13 | Continues, the capture script, a game over, pause, the hold in the runs table, and the pipe at each hold |
| `tests/training/test_training_config.py` | #76, #31 | +21 −2 | The preset keeps gamma 0.997 and the reward unnormalized; older configs load with hold 8 |
| `tests/training/test_game_runner.py` | #31 | +12 −2 | The runs table records the hold; a stand-in `env_factory` takes it |
| `tests/training/test_ppo.py` | #31 | +2 −2 | Two stand-in `env_factory` functions take the hold |
| `.claude/skills/experiment/SKILL.md` | #76 | +9 −3 | The launch command |
| `src/algoarcade/envs/env.py` | #31 | +6 −2 | The env takes a per-game `hold`; the decision stays 24 frames |
| `tests/test_env.py` | #31 | +11 −1 | A longer hold keeps the step at 24 frames |
| `experiments/e9_super_mario_land.py` | #31 | +18 −2 | The E9 runs for the queue, at hold 16 and hold 8 |
| `tests/training/test_e9.py` | #31 | +26 −1 | E9's PPO config equals FireRed's field by field, and each arm's hold |
| `scripts/e9.py` | #31 | +105 −25 | E9's schema check, final `level_progress` per arm, throughput and dropped runs |
| `tests/test_e9_script.py` | #31 | +104 −21 | Tests for that script on fake-logger data |
| `scripts/sml_pipe_probe.py` | #31 | +63 −0 | Reruns the evidence for the hold |
| `docs/adding-a-game.md` | #31 | +24 −1 | The guide drafted from this diary |

## What it says about the claim

The plan says a new game costs a reader, a reward config and start states, with nothing downstream changing. Three downstream changes were needed. A game over returns the game to its title, whose demo plays on its own (`Continues`). The schema's FireRed counts needed a 0 (`Measures`). Mario's jump needs A held longer than FireRed's menus do (`hold`, which also had to reach the config, the logs and the renderer). The trainer and the PPO config did not change. The rest of the outside lines are registration (`rom.py`, the conformance registry, the runner's game list, the preset), tests and docs.
