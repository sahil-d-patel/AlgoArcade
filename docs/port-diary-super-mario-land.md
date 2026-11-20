# Port diary: Super Mario Land

Every file three pull requests touched to add Super Mario Land: ENV-03 (#58, the PyBoy backend and the reader), TRN-05 (#76, the start state, observation, reward table and runner wiring) and ENV-11 (#31, the button hold and E9). [adding-a-game.md](adding-a-game.md) turns it into steps.

## Totals

From `git log --no-merges --numstat` over each pull request's own commits, summed per file. A line changed in two commits counts twice. The ENV-11 row is this branch before this diary, `27bcbe8..9f4bb7f`.

| Pull request | Range | Files | Added | Deleted |
|---|---|---|---|---|
| #58 ENV-03 | `13a0838^1..13a0838^2` | 13 | 532 | 130 |
| #76 TRN-05 | `c09abc7^1..c09abc7^2` | 16 | 458 | 49 |
| #31 ENV-11 | `27bcbe8..9f4bb7f` | 11 | 247 | 13 |
| All three | | 32 distinct | 1,237 | 192 |

Inside `src/algoarcade/games/` and `src/algoarcade/backends/pyboy.py`: 6 files, +265 −31. Outside: 26 files, +972 −161. Of the lines added outside, 604 are tests (8 files), 61 Markdown (8 files) and 307 code (10 files, the scripts and the experiment file included).

The commands, from the repo root:

```
git log --no-merges --numstat --format= 13a0838^1..13a0838^2
git log --no-merges --numstat --format= c09abc7^1..c09abc7^2
git log --no-merges --numstat --format= 27bcbe8..9f4bb7f
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
| `docs/harness-spec.md` | #58, #76, #31 | +16 −11 | The location rule, start state, continue rule and the hold are contract |
| `docs/decisions.md` | #58, #76, #31 | +8 −3 | Two unlabelled addresses, the continue rule, the hold and E9's scoring |
| `tools/github/issues/ENV-03-pyboy-mario-land.md` | #58 | +1 −1 | The oracle criterion as approved on #10 |
| `scripts/capture_sml_states.py` | #76 | +44 −0 | Regenerates `level_1_1`; state files stay out of git, so each game needs its own capture script |
| `src/algoarcade/training/games.py` | #76, #31 | +124 −31 | The env factory and its `env_factory` entry; `Continues`, which reloads the start state after a game over; `Measures` taught to log 0 for FireRed counts the game lacks; the 16-frame hold |
| `src/algoarcade/training/config.py` | #76 | +18 −3 | The `super_mario_land` preset: FireRed's PPO values on this game |
| `src/algoarcade/training/runner.py` | #76 | +12 −1 | The runner's list of game env ids, and `rollout/continues` in `progress.csv` |
| `src/algoarcade/training/parquet.py` | #76 | +2 −1 | A comment: a continue does not end an episode |
| `src/algoarcade/training/README.md` | #76 | +1 −1 | Names the game |
| `tests/training/test_super_mario_land_training.py` | #76, #31 | +179 −2 | Continues, the capture script, a game over, pause, and the pipe at the new hold |
| `tests/training/test_training_config.py` | #76 | +7 −2 | The preset keeps gamma 0.997 and the reward unnormalized |
| `.claude/skills/experiment/SKILL.md` | #76 | +9 −3 | The launch command |
| `src/algoarcade/envs/env.py` | #31 | +6 −2 | The env takes a per-game `hold`; the decision stays 24 frames |
| `tests/test_env.py` | #31 | +11 −1 | A longer hold keeps the step at 24 frames |
| `experiments/e9_super_mario_land.py` | #31 | +14 −0 | The E9 runs for the queue |
| `tests/training/test_e9.py` | #31 | +23 −0 | E9's PPO config equals FireRed's field by field |
| `scripts/e9.py` | #31 | +73 −8 | E9's schema check, final `level_progress`, throughput and IQM |
| `tests/test_e9_script.py` | #31 | +61 −0 | Tests for that script on fake-logger data |
| `docs/adding-a-game.md` | #31 | +23 −0 | The guide drafted from this diary |

## What it says about the claim

The plan says a new game costs a reader, a reward config and start states, with nothing downstream changing. Three downstream changes were needed. A game over returns the game to its title, whose demo plays on its own (`Continues`). The schema's FireRed counts needed a 0 (`Measures`). Mario's jump needs A held longer than FireRed's menus do (`hold`). The trainer and the PPO config did not change. The rest of the outside lines are registration (`rom.py`, the conformance registry, the runner's game list, the preset), tests and docs.
