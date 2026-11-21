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
- Dependencies: numpy, gymnasium, pyboy, pyarrow, tyro, cffi; for training torch, stable-baselines3, wandb, matplotlib (added in TRN-03 for learning curves); for the agent mcp, anthropic; for development ruff, pyrefly, pytest, poethepoet, prek, typos.
- Tooling: Python 3.12, uv with a committed lockfile.
- Scope: the repo ships the backend protocol, the fake backend and the conformance suite already written, so week 1 has something to build against. They are drafts until the November 1 freeze, and the Environment pair owns them from here.

## Log

- 2026-10-11: Plan fine-tuned before kickoff.
- 2026-10-25: On macOS arm64, Homebrew's mgba links ffmpeg 8 but Homebrew ships ffmpeg 9. We pin `ffmpeg@8` and put `/opt/homebrew/lib:/opt/homebrew/opt/ffmpeg@8/lib` on `DYLD_FALLBACK_LIBRARY_PATH` until Homebrew rebuilds mgba. libretro.py stays the fallback (#4).
- 2026-10-27: Two Super Mario Land addresses have no label in the kaspermeerts/supermarioland disassembly (commit 618d00e): the level block at 0xC0AB and Mario's screen x at 0xC202. The reader cites PyBoy v2.8.1, commit f97d75162df5dafa82db50850d04497013cf215b, `pyboy/plugins/game_wrapper_super_mario_land.py` lines 360-361, for both (ENV-03).
- 2026-11-02: The schema freeze moves to Nov 8. ENV-08 was not validated on real memory by Nov 1, which is the rule's trigger (#20).
- 2026-11-02: ENV-04 is built on Kumario1's `issue-11/mgba-backend` branch, with their commits kept, and opened from the team repo (#11).
- 2026-11-02: The three libmgba-py 0.2.0-2 zips (MPL-2.0) are mirrored as a release on this repo, because upstream has had no push since December 2024 (#11).
- 2026-11-02: matplotlib is allowed as a dependency in the `train` group only, for learning-curve figures (#7, #14).
- 2026-11-02: `StateNotReady` is the shared way a reader says memory is not ready. It is raised when the reader detects that; on the title screen and intro the FireRed v0 reader returned zeros instead, a known gap (#12) that was closed before #57 merged: an empty save block now raises.
- 2026-11-02: The LLM attempt loop's dollar cap is hard. It stops before a request whose upper-bound cost would pass the cap (#16).
- 2026-11-03: FireRed's `story_flags` counts seven flags from Pallet Town to Pewter: 0x2CF (visited Oak's Lab), 0x828 (starter), 0x258 (lab rival battle finished), 0x230 (Route 1 potion), 0x829 (Pokédex), 0x4B0 (Brock defeated), 0x254 (TM39). This list defines the frozen column (#19).
- 2026-11-04: `AlgoArcadeEnv` retries a read that raises `StateNotReady` one frame later, up to 8 times, before letting it propagate. This reverses the earlier rule that a retry would hide reader bugs: on rev 1 the party checksum fails on isolated frames while the game rewrites the entry (about 1 in 10,000 per-frame reads in battle, never two frames in a row), which would otherwise kill a training worker (#19).
- 2026-11-04: Progress terms (events, level, badge, milestone) pay a rise only once two consecutive reads agree on it, so a rise pays one step late. The save blocks move on warps, battles and menus, and for one frame a count can read high; badge counts 0, 0, 0, 1, 0, 0, 1 paid the spike and not the real badge. This amends the zero-on-no-op rule: a no-op step that is the second read confirming the previous step's rise pays that rise, which any two-read confirmation must do (Sahil, Lead PM). A rise on an episode's last step is never paid, which loses at most one step's pay per term per episode (#22, #73).
- 2026-11-04: The Super Mario Land location is `(world, level, x_tile, mario_y // 8)` in 8-pixel tiles (Sahil, #58), so it fills the four location columns like FireRed's and jumping in place cannot farm `explore` pixel by pixel. `x_tile` counts the columns `Call_2198` has loaded instead of dividing `level_progress`, which dips by 15 at block boundaries. Two more addresses have no label at 618d00e: Mario's screen y at 0xC201, from the disassembly's comments ("Mario Y pos", `bank0.asm` lines 1185 and 1430), and the column bit at 0xFFA3, from `Call_2198` (`bank0.asm` lines 5098-5113). PyBoy's wrapper reads neither; the ROM oracle test bounds `x_tile` against it, and nothing checks `mario_y` beyond changing on a jump (ENV-03).
- 2026-11-04: In TRN-05's Super Mario Land runs a game over continues: the backend reloads the start state when the title menu comes up. AlgoArcadeEnv ends episodes only on the step budget, the schema's `end_reason` has no value for a game over, and the menu's demo would pay `explore`. A continue is not a terminal transition: the episode goes on, PPO bootstraps across it, and the episode still ends as `truncated`. The count goes to `progress.csv` and W&B as `rollout/continues`, not to the Parquet schema. The menu flag at 0xFF9F has no label at 618d00e. Only two instructions in the four banks write it: the menu init stores 0x28 (`bank0.asm` lines 591-593) and `.startLevel` clears it when START begins a game (lines 760-762). Its other reads skip score, coins, lives and pause in the demo, or feed the demo its recorded input (`bank0.asm` lines 185, 320, 339, 798, 4212, 4246, 5062; `bank3.asm` line 439). So pause, pipes, level clears and bosses leave it alone; a soft reset (A+B+START+SELECT, which the action set cannot press) goes through `Init` and the menu. Measured on our dump: 0x28 on the title, 0 from the first frame of play and through a pause (`test_pausing_leaves_the_menu_flag_clear`), 0x28 again after a game over. Pipes, a level clear and a boss are untested on the game.
- 2026-11-13: The logging schema is frozen as `schema-v1`, five days after the November 8 date; no lane asked for a field by November 7 (#20). Games without a FireRed field log its column as False or 0 (Super Mario Land logs `in_battle` False and the counts as 0) instead of nullable columns, so any analysis that pools games filters by `env_id`. Frames per decision stay 24, environments 16, and the steps table keeps every fourth decision (harness-spec, Settled at the freeze).
- 2026-11-13: An LLM attempt whose model stops calling tools logs `end_reason` `truncated`: the frozen `schema-v1` set has no other value for it (Sahil, Lead PM; #23).
- 2026-11-13: For LLM rows the `action` column holds the Button value (A 0 to L 9), the index into the MCP server's ten buttons, not the RL env's six-action index, so a pooled analysis of actions filters by `algorithm` (Sahil, Lead PM; #23).
- 2026-11-14: The TRN-07 parity gate passes: at five seeds per trainer we found no detectable difference between the owned PPO (`src/algoarcade/training/ppo.py`) and SB3 PPO on the same presets. Sahil (Lead PM) accepts the switch to the owned trainer for the experiments that need it (PQN, RND); SB3 stays the default runner. The measure is the IQM of the deterministic eval return with 95% bootstrap bands over seeds 0 to 4 on both sides ([figure](figures/trn07-parity.png)). On every task the bands meet at all 10 evenly spaced checkpoints, the last included. Final IQM, SB3 against owned: CartPole 500.0 [500.0, 500.0] against 500.0 [480.0, 500.0]; Super Mario Land at 2M steps 1.9 [1.5, 2.4] against 1.7 [0.6, 2.5]; seg_02_pallet at 1M steps 0.2 [0.1, 0.2] against 0.2 [0.2, 0.2]. Neither trainer learns seg_02 in 1M steps, so that task tests little. The gate was due Nov 8; it ran Nov 13 and 14 on the Mac, not the box (#28).
- 2026-11-14: Success rates come from sampled actions. In TRN-08's 15 runs at 1M steps, the 20 deterministic evaluation episodes at each of the 30 checkpoints were one episode replayed, and none reached its goal, while sampled training episodes reached it 8 to 78% of the time in each run's last tenth. Re-scored with 20 sampled episodes each (`scripts/rescore_eval.py --sampled`), the same final checkpoints reached the goal in 54, 18 and 35 of 100 on seg_01, seg_02 and seg_03. `TrainConfig.eval_deterministic` is a required field: False in `BASELINE_V1`, True in every other preset. `BASELINE_V1` trains for 15M steps, the experiment run length in the technical plan's compute budget; 1M was TRN-05's guess. It still saves a checkpoint every 500,000 steps but evaluates only every 1.5M steps and at the end, 10 times per run (`eval_every`, a required multiple of `checkpoint_every`), which cuts evaluation across TRN-08's 15 runs from 27.6M to 9.2M single-env steps. The explore weight stays as it is and is the first knob if the box run still misses (Sahil, Lead PM, #29).
- 2026-11-15: Feature freeze (OPS-06, #34). Cut or deferred: E8, the LLM curriculum (#98), is not queued and does not merge before the results freeze. #95 (E1's PQN arm), #96 (E2's pixels arm) and #91 (E9's `hold` field) add features and wait for the Lead PM to merge them as freeze exceptions; until then E1 and E2 have only the baseline arm. The week's video review is deferred: no run had finished. The Brock runs (E6) run past the November 22 results freeze. At the Mac benchmark's rates no E1 to E5 arm finishes by November 22, because evaluation on S1 takes 34.7 of a run's 38.9 hours; the cut order is in technical-plan.md, Compute budget, and its first step, a lighter evaluation for every arm, is the Lead PM's to take. S0 and S1 get step budgets of 87,040 and 71,680 and end on the Boulder Badge flag (harness-spec, Start states and segments).
