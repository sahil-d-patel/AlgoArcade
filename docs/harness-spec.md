# AlgoArcade harness spec

Fine-tuned October 11, 2026. This defines the interfaces every part of AlgoArcade builds against, and the logging schema. Interfaces can change until November 1. The logging schema freezes on November 8, moved from November 1 (#20), and any change after that means rerunning training.

The reasoning behind these choices is in [technical-plan.md](technical-plan.md). This file is the contract.

## Purpose and principles

1. Backend-agnostic. Training and visualization code never knows which console it is talking to. One interface, three implementations.
2. RAM is the source of truth. Everything we reward or log comes from emulator memory. Pixels are an observation option, not a measurement tool.
3. Reward is built from parts. Every term has a name and its own log column. If we cannot chart a term's contribution, we cannot tell when the agent is exploiting it.
4. Log everything the visuals will need. Visuals are built in weeks 6 and 7, so anything not logged during training is gone.
5. Tests need no ROM. Anything that can be tested on the fake backend is, so CI runs on every pull request.

## GameBackend

GameBackend is the only layer that knows which console we are emulating. The protocol lives in `src/algoarcade/backends/base.py`. There are three implementations: FakeBackend, PyBoyBackend and MgbaBackend.

Loading a ROM, with an optional save state, is the constructor's job. Every backend then provides the same operations:

| Operation | What it does |
|---|---|
| step | Advances the emulator a given number of frames with no input |
| press | Holds a set of buttons for a number of frames, then releases them for a number of frames |
| screen | Returns the current frame as an RGB array. It is a copy, safe to keep. |
| read | Returns a run of bytes from a memory address |
| save_state | Returns the full emulator state as bytes |
| load_state | Restores a state from bytes |
| close | Shuts the emulator down |

Helpers read 8, 16 and 32-bit little-endian values through `read`.

Rules for implementers:

- One emulator per process. Create it inside the worker, never in the parent.
- `read` must be cheap. On mGBA, map the two RAM regions once as views and slice them. Do not go through the CPU bus on every call.
- States are bytes in memory. Writing a state to disk is the caller's choice.
- Rendering can be switched off when the observation does not need pixels.

## ROM pinning

FireRed shipped in several revisions, and addresses differ between them. On load, the backend hashes the ROM and refuses to run unless the hash is in the table in `src/algoarcade/rom.py`.

| ROM | SHA-1 |
|---|---|
| Pokémon FireRed (USA) rev 0, the target | 41cb23d8dccc8ebd7c649cd8fbb58eeace6e2fdc |
| Pokémon FireRed (USA) rev 1, accepted | dd5945db9b930750cb39d00c84da8571feebf417 |
| Super Mario Land | Recorded by ENV-03 once two members confirm the same dump |

Address tables are keyed by this hash. Rev 0 and rev 1 share every RAM address we use, but ROM function addresses differ.

A wrong ROM is an error, never a warning.

## StateReader and GameState

A StateReader is written per game. It reads memory through a GameBackend and returns a GameState: a frozen record of everything we care about. The protocols live in `src/algoarcade/games/base.py`. A reader raises `StateNotReady` from that module when the reader detects memory is not ready; callers must handle this as not ready rather than substitute zero-valued fields. On the title screen and intro, FireRed's save-block pointer is valid but the block is empty; the reader detects this (position (0, 0, 0, 0) with an empty party, which no overworld state has) and raises.

The framework asks one thing of every GameState: a `location`, a hashable tuple that identifies where the player is. Exploration rewards, the archive and the heatmap all key on it. For FireRed it is `(map_group, map_num, x, y)`.

### FireRed GameState fields

| Group | Field | Meaning |
|---|---|---|
| Position | map_group, map_num | Which map the player is on |
| Position | x, y | Tile coordinates on that map |
| Progress | badge_flags, badge_count | The badge byte and its popcount |
| Progress | story_flags | Count of set flags on the story whitelist |
| Progress | trainers_defeated | Popcount of the trainer flag range |
| Progress | world_map_flags | Count of the visited-map flags, 0x890 through 0x8C2 |
| Progress | milestone_count | How far down the ordered Pallet-to-Brock list |
| Progress | pokedex_owned, pokedex_seen | Pokédex counts |
| Party | party | Up to six Pokémon: species, level, HP, max HP, status, moves |
| Party | party_level_sum, party_hp_frac | Totals across the party |
| Context | money | Decrypted |
| Context | in_battle | Whether a battle is active |
| Context | text_active | Field message text is printing. It turns false when printing ends, while the box stays up until A, and stays false in battle |
| Context | text_box_open | A text box is printing, waiting at its arrow, or finished and waiting for A or B, in the field and in battle. `anti_spam` reads this |
| Context | text_progress | A value that changes whenever text in the open box advances, in the field and in battle |
| Context | script_active, controls_locked | A script is running; the player cannot move |
| Context | text_speed, battle_scene_off | Saved options; fast text is 2 and battle scenes must be off |
| Battle | own and enemy HP, level, moves, PP | Only meaningful while in_battle is true |

Fields arrive in stages: position and plaintext party fields in ENV-05; badges, encrypted party
fields and the remaining context in ENV-08; detailed battle fields and milestones in ENV-13.

Every field has a validation test against a captured save state with a known value.

## FireRed memory map

These come from the pret/pokefirered symbol files at commit `3b401eaf14f193946899eb7c056d5ac02a470126` and were checked against the decompilation's source. The address table in the code is the source of truth. This section is the map.

| Symbol | Address (rev 0) | Notes |
|---|---|---|
| gSaveBlock1 | 0x0202552C | The block. gSaveBlock1Ptr points 0 to 0x7C bytes above it, in steps of 4. |
| gSaveBlock1Ptr | 0x03005008 | Pointer. Dereference on every read. |
| gSaveBlock2 | 0x02024588 | The block. gSaveBlock2Ptr follows the same rule, at the same offset as gSaveBlock1Ptr. |
| gSaveBlock2Ptr | 0x0300500C | Pointer. Dereference on every read. |
| gPlayerPartyCount | 0x02024029 | |
| gPlayerParty | 0x02024284 | Six entries of 100 bytes |
| gBattleMons | 0x02023BE4 | Plaintext battle data |
| gMain | 0x030030F0 | inBattle is bit 1 of the byte at +0x439 |
| sMessageBoxType | 0x0203709C | 0 hidden, 2 normal, 3 auto-scroll |
| sGlobalScriptContextStatus | 0x03000EA8 | 0 running, 1 waiting, 2 shut down |
| sLockFieldControls | 0x03000F9C | |
| sTextPrinters | 0x02020034 | 32 TextPrinters of 0x24 bytes, indexed by window id |
| sGlobalScriptContext | 0x03000EB0 | The overworld script's context: mode at +1, nativePtr at +4 |
| WaitForAorBPress | 0x0806B898 (rev 1: 0x0806B8AC) | ROM function; the only address that differs between the revisions |

Offsets inside SaveBlock1: x at 0x00, y at 0x02, map group at 0x04, map number at 0x05, money at 0x290, flags at 0xEE0. The eight badge flags are one byte at 0xFE4. Trainer flags run from 0xF80 to 0xFDF. The money key is in SaveBlock2 at 0xF20.

Inside a party entry: status at 0x50, level at 0x54, HP at 0x56, max HP at 0x58, all plain. The encrypted block is at 0x20 to 0x4F.

ENV-08 source layouts below were checked at `e93eb12a336fb97df420b0961753d8a3713c9fcc`,
the source commit used for the pinned symbol build:

| Fields | Symbol and layout | Source |
|---|---|---|
| money | `gSaveBlock1Ptr` + 0x290, XOR `gSaveBlock2Ptr` + 0xF20 | `include/global.h`, `src/money.c` |
| badge_flags, badge_count | `gSaveBlock1Ptr` + 0xFE4; flags 0x820 through 0x827 | `include/constants/flags.h`, `src/event_data.c` |
| trainers_defeated | `gSaveBlock1Ptr` + 0xEE0; flags 0x500 through 0x7FF | Same flag definitions and indexing |
| story_flags | Seven persistent flags listed below, indexed from `gSaveBlock1Ptr` +0xEE0 | `include/constants/flags.h`, `src/event_data.c` |
| world_map_flags | `gSaveBlock1Ptr` +0xFF2; 51 bits for flags 0x890 through 0x8C2 | Same flag definitions and indexing |
| text_speed, battle_scene_off | `gSaveBlock2Ptr` +0x14 bits 0–2; +0x15 bit 2 | `include/global.h` |
| pokedex_owned, pokedex_seen | `gSaveBlock2Ptr` + 0x28 and +0x5C; seen copies at `gSaveBlock1Ptr` +0x5F8 and +0x3A18 | `include/global.h`, `src/pokedex_screen.c` |
| in_battle | `gMain` +0x439, bit 1 | `include/main.h` |
| text_active | `sMessageBoxType` is not hidden (0). `Task_DrawFieldMessageBox` sets it to hidden when printer 0 finishes, before the player closes the box | `src/field_message_box.c` |
| text_box_open | Printer 0 or 24 active (+0x1B), or `sGlobalScriptContext` in native mode (2) with `nativePtr` at `WaitForAorBPress`, which `waitbuttonpress` runs while a finished message stays up. A stopped script can keep a stale `nativePtr`, so the mode is checked too. A yes/no menu reads false | `include/text.h`, `src/text_printer.c`, `include/script.h`, `src/scrcmd.c` |
| text_progress | `currentChar` (+0x00) of printer 24 while it is active, else of printer 0. Printer 0 draws window 0, the field message box and `B_WIN_MSG` in battle; 24 is `B_WIN_OAK_OLD_MAN`, Oak's battle tutorial and the old man's catching demo. It is a position in the string buffer, so two finished messages of the same length read the same | `include/text.h`, `src/text_printer.c`, `include/constants/battle.h` |
| script_active | `sGlobalScriptContextStatus` is running (0); waiting (1) and shut down (2) are false | `src/script.c`, `ScriptContext_IsEnabled` |
| controls_locked | `sLockFieldControls` is nonzero | `src/script.c` |
| party species, moves | `gPlayerParty`, 100-byte entries; XOR twelve words at +0x20 with personality (+0x00) XOR trainer id (+0x04), then check the 16-bit sum against +0x1C | `include/pokemon.h`, `src/pokemon.c` |

The Pokédex screen counts only the 386 national entries. Seen bits must agree in all three copies; an owned bit also needs that seen bit. Party substructures have 24 physical orders, selected by personality modulo 24: species is in growth, four move ids are in attacks, and `is_egg` is bit 30 of the word at +4 in misc (`MON_DATA_IS_EGG`). `party_level_sum` sums the live party's levels and `party_hp_frac` is its total HP over its total maximum HP, or 0 when that total is zero; both leave eggs out, while `party_count` counts them. Species and move ids use the game's internal ids. `FireRedReader.check_options(backend)` raises `ValueError` unless saved text speed is fast (2) and battle scenes are off. Call it at environment reset.

The story whitelist, approved in `docs/decisions.md` on 2026-11-03, counts each of these persistent events once:

| Symbol | Flag id | Event |
|---|---|---|
| FLAG_VISITED_OAKS_LAB | 0x2CF | Entered Oak's Lab |
| FLAG_SYS_POKEMON_GET | 0x828 | Starter received |
| FLAG_BEAT_RIVAL_IN_OAKS_LAB | 0x258 | Lab rival battle finished, won or lost; `trainers_defeated` counts a win |
| FLAG_GOT_POTION_ON_ROUTE_1 | 0x230 | Route 1 gift potion received |
| FLAG_SYS_POKEDEX_GET | 0x829 | Pokedex received |
| FLAG_DEFEATED_BROCK | 0x4B0 | Brock defeated |
| FLAG_GOT_TM39_FROM_BROCK | 0x254 | Brock's TM received |

`tests/test_firered_fields.py` checks each field against hash-pinned local captures whose
values were read off the screen: trainer card, party and summary screens, options, dialogue,
battle and Pokedex. Each capture runs only on the revision it was made with. Rev 0 captures are
Kumario1's. Rev 1 captures come from `scripts/capture_validation_states.py`, including a
poisoned Squirtle (status 8), plus `brock_goal` from `scripts/capture_states.py` (Boulder Badge).
On each revision a replay crosses a warp and the lab rival battle, checking map ids, money and
that no progress count falls. Capture tests carry the `rom` marker and skip without the ROM or
the capture.

Rules:

- The save blocks relocate on every warp and every return from a battle or full-screen menu. The offset is one of 32 values. Never keep a dereferenced address across steps.
- Check that each pointer lands inside its expected range before using it. A null or out-of-range pointer means "not ready", not zero.
- Both blocks move by the same offset (`SetSaveBlocksPointers`). Two different offsets mean "not ready".
- Validate the party checksum before trusting species or moves.
- File-local symbols carry a numeric suffix that differs between builds. Match on the base name.
- No address enters the code without a symbol name and the commit it came from.

## Reward registry

A reward component is a small object with a name, a reset step that runs at the start of an episode, and a scoring step. The scoring step looks at the previous GameState and the current one and returns a number. The protocol lives in `src/algoarcade/rewards/base.py`.

Components are combined from a config that gives each a weight. Each game passes the registry its own table of components, so its GameState needs only the fields those components read. The environment returns the weighted total as the reward and reports each component's raw value in the step info, so each can be logged on its own.

FireRed's ten components are in `src/algoarcade/rewards/registry.py`: `COMPONENTS` builds them, `WEIGHTS` holds the starting weights from the technical plan (Reward design), and `DEFAULT_WEIGHTS` turns on the first five. The weights and the level cap are guesses until TRN-08 tunes them. Each raw value is zero or more; a penalty's sign is in its weight.

| Component | Pays or charges | Weight | Default | Predicted exploit, and what stops it |
|---|---|---|---|---|
| events | Each new story-whitelist flag above the episode's best | +1.0 | on | Setting and clearing a flag: only a new best pays. |
| explore | First visit to a location this episode | +0.02 | on | Two-tile oscillation: the visited set is permanent for the episode. |
| level | Rise in party level sum, at most 1.0 a step, nothing past a sum of 20 | +0.5 | on | Grass camping: the cap holds an episode to 10.0, one badge. |
| badge | A new badge | +10.0 | on | Misread badge byte: a read above 8 or a rise over 1 is ignored. |
| milestone | A new entry on the Pallet-to-Brock list | +2.0 | on | Same as events. |
| heal | Rise in party HP fraction between two reads that agree | +0.3 | off | Fainting for a free heal: a rise from a fainted party, a party size change or one bad read never pays. Damage then a Pokémon Center heal still pays; that is what E5 measures. A level-up raises a damaged Pokémon's HP fraction, so with heal on, grass camping keeps paying heal past the level cap. |
| blackout | The party goes from all fainted to healed | -5.0 | off | Fleeing or avoiding every battle to dodge it, as in RL-emerald. Nothing stops this; it is why the term is off. |
| anti_loop | A move onto a tile stood on in the last 30 steps | -0.01 | off | Standing still is free, so the agent may stop moving, or walk loops longer than 30 steps. |
| anti_spam | A press that left an open text box where it was | -0.02 | off | Avoiding NPCs and signs. It reads ENV-08's `text_box_open` and `text_progress`, so it charges presses on a finished box that is still up and on battle text at its arrow. A yes/no menu reads as no box open, so moving its cursor is not charged. |
| time | Every step | -0.001 | off | None while episodes end only on a step budget: every policy pays the same total. |

`heal` and `blackout` read `party_hp_frac`, which ENV-08 computes as party HP over party max HP, leaving eggs out: an egg has no HP to heal.

The five scripted exploit policies are in `tests/test_exploits.py`, and `uv run pytest -k exploit` runs exactly those.

Rules:

- Every component returns exactly zero on a no-op step and on the reset step, except the penalties `time` and `anti_spam`, which never pay and are off by default. `time` charges every step; `anti_spam` charges a step where `text_progress` did not change while `text_box_open` held, including a reset into one.
- Novelty uses coordinate tuples in a permanent per-episode set. Never screen hashes, never a sliding window.
- A component that reads a field validates it first. Garbage memory must not pay, and must not become the best value that later real progress has to beat. A count term ignores a read above the game's maximum or a rise faster than the game allows, and takes its best only from two consecutive reads that agree, so one bad read, at reset or later, cannot set it. A bad read that repeats on two consecutive steps can; the reader has to catch that.
- Each known exploit has a scripted policy in the tests that tries it and must stay under a stated reward.
- A pull request that adds or changes a component says how an agent would exploit it.

## AlgoArcadeEnv

AlgoArcadeEnv is the Gymnasium environment, in `src/algoarcade/envs/env.py`. It is built from a backend, a StateReader, a `RewardRegistry`, the start state's bytes, a step budget and the game's feature table. The seed comes through `reset(seed=s)`. Create the backend inside the env factory: SubprocVecEnv spawns its workers.

| Setting | Value |
|---|---|
| Decision length | 24 frames: hold 8, release 16 |
| Actions | `Discrete(6)`: 0 up, 1 down, 2 left, 3 right, 4 A, 5 B. `start=True` appends Start, then `noop=True` appends a no-op that holds nothing for 24 frames. |
| Episode end | The step budget only. `truncated` is True on the budget's last step; `terminated` is always False. |
| Reset | Loads the start state, then steps 0 to 7 no-op frames drawn from the env's seeded `np_random`. The range is a guess; its effect is measured below. Reset pays no reward. |
| Not ready | A read that raises `StateNotReady` is retried one frame later, up to `NOT_READY_FRAMES` (8) times, and then propagates out of `reset` or `step`. The env never substitutes a state. A retry adds frames only to that step, so the same seed and actions still replay the same episode. |

Reset noise does little. A 24-frame step absorbs most of a 0 to 7 frame offset, so the offset changes outcomes only through the game's RNG, such as a wild encounter. With the actions fixed (`scripts/env_trace.py --action-seed 0`), 128 steps from reset seeds 0 to 9 gave one location trace and one return from `s1.state` (13.0) and from `seg_02_pallet.state` (11.0), and two location traces from `seg_04_route1.state`. On `s1.state` the ten runs still ended in 4 different EWRAM states. So a deterministic evaluation from a state with no encounters nearby, such as these two, replays the same episode; TRN-05 and ENV-09 should not count on reset noise for variety there.

For TRN-05:

- A `StateNotReady` that survives the retries kills a SubprocVecEnv worker, and the main process sees only `EOFError`. Log the worker's traceback so a reader bug is visible. Measured on rev 1: FireRed's party checksum failed for single frames while the game rewrote an entry, never on two frames in a row: 3 times in 26,186 per-frame reads replaying `scripts/capture_validation_states.py` from the new game to the Pokédex, and 4 times in 5,630 from `validate_starter` through the rival battle to Pallet Town. 18,000 random env steps, 6,000 each from `s1`, `seg_02_pallet` and `seg_04_route1`, read not ready 0 times.
- The observation is a `Dict`, so SB3 needs `MultiInputPolicy`, not `MlpPolicy`.

The observation is a `Dict` of two float32 boxes:

| Key | Shape | Contents |
|---|---|---|
| `vector` | (number of features,) in [0, 1] | One value per entry in the game's `FEATURES` table (`games/fake.py`, `games/firered/features.py`), clipped to [0, 1]. FireRed has 7: map_group, map_num, x, y, party_count, party_level_sum, party_hp_frac. |
| `visited` | (1, 9, 15) in {0, 1} | Tiles visited this episode, in a crop of 9 rows by 15 columns centred on the player. The location's last two elements are x and y; the rest name the map. The crop size is a guess from FireRed's 15 by 10 tile screen. |

`info` on every step:

| Key | Value |
|---|---|
| `reward/<name>` | Each configured component's raw value, before its weight. The logs store it as `reward_<name>`. The step's reward is the sum of weight times raw value. |
| `location` | The GameState's location after the decision. For FireRed `(map_group, map_num, x, y)`. |
| `milestone_count` | Present when the GameState has the field. FireRed gains it in ENV-13; Super Mario Land has none. |

`reset` returns `location` and `milestone_count` only. ENV-09 adds a save state for the archive.

Observation modes:

- `symbolic`: the only mode so far, so the env takes no mode argument yet. The walkability channel arrives in ENV-12, as a second channel of the crop.
- `pixels`: the screen at 72 by 80 greyscale, three frames stacked, plus the same state vector (ENV-12).
- `both` (ENV-12).

The environment must pass Gymnasium's environment checker on all three backends. `uv run python scripts/env_trace.py <fake|mgba|pyboy> --state <file>` prints a hash of a 500-step trace; two processes with the same seed must print the same hash.

## Start states and segments

Episodes start from save states listed in a manifest in the game's package. Each entry has a name, the SHA-256 of the state file, a goal, and a step budget. The goal is a condition on GameState, such as "map is Route 1".

| Name | Start | Goal | Step budget |
|---|---|---|---|
| S0 | Player's bedroom, new game | Furthest milestone | Set per run |
| S1 | After Oak's Parcel, Squirtle | Furthest milestone | Set per run |
| seg_01_house | Player's bedroom | Leave the house | 2,048 |
| seg_02_pallet | Pallet Town | Enter Route 1 | 4,096 |
| seg_03_rival | Oak's Lab | Win the rival battle | 3,072 |
| seg_04_route1 | South end of Route 1 | Enter Viridian City | 6,144 |
| seg_05_forest | Viridian Forest entrance | Exit the north side | 8,192 |
| gym_door, brock | Pewter Gym door; the Brock battle | Badge flag set | Set per run |

State files are not in git. They live in `data/states/` on each machine. Sahil (PM) regenerates them with `scripts/capture_states.py` and hands the files to members directly. The manifest hash proves everyone has the same file.

The states were captured on FireRed rev 1. The address table is the same for rev 0, so a rev 0 user can regenerate the states into another folder with the script's `--out`, but nobody has run that.

gym_door and brock carry a level 14 Squirtle that knows Water Gun, ground on Route 1 before the forest. Pressing A every turn wins the Brock battle from either state, so label any result from them with that.

Every state has text speed fast and battle animations off. The reader asserts both at reset.

Evaluation always starts from the fixed state with the archive off, even when training used archive resets.

## Logging schema

Freezes November 8 under the tag `schema-v1`. It moved from November 1 because ENV-08 was not validated on real memory by then (#20). Before the freeze, each lane confirms the logs hold every field its week 6 visual needs. Owner: Sahil (PM), in OPS-03. The code is `src/algoarcade/logs/schema.py`, and it must match this section.

Logs are Parquet files under `data/runs/run_id=<id>/<table>/`, one folder per table: `steps`, `visits`, `episodes`, `runs`. Each writing process owns its own files, named `<writer_id>-<n>.parquet`, and rolls to a new file every 100,000 rows by default, because a Parquet file cannot be read until it is closed. The open file is `_<writer_id>-<n>.parquet.inprogress`, so a glob on `*.parquet` and a pyarrow folder read (`pq.read_table`, `ds.dataset`) see only closed files.

A writer refuses a table folder that already holds files from its writer id, so reusing a run id cannot overwrite or mix data. A resumed run says so (`resume=True`), and its writer numbers on after its highest file, including one left open by a crash.

The writer validates every batch. A missing, extra or duplicated column, a wrong type, a null in a required column, or a value outside a column's listed set (`end_reason`, `llm_tool_level`, `status`) raises `SchemaError`. Column order is free.

Types: Text is `string`, Integer is `int32`, 64-bit integer is `int64`, Small integer is `int16`, Decimal is `float64`, Boolean is `bool`. A column is required unless its row says nullable.

### Reward columns

Each of the ten components in the technical plan has a column `reward_<name>`: events, explore, level, badge, milestone, heal, blackout, anti_loop, anti_spam, time. It holds the raw value the registry reports, before the weight. It is null when the run's config leaves the component out. The weights are in the per-run `hyperparameters` under `reward_weights`, and `reward_total` is the sum of weight times raw value. The registry's step-info key `reward/<name>` becomes the column `reward_<name>`. A new component after the freeze adds a column, which is a PM decision.

### Per-step table

Written every Nth step: each environment writes the row of a decision whose `episode_step` is a multiple of N. N is `step_log_every` in the per-run table, at least 1. The default is 4; LLM runs use 1. Evaluation episodes write no per-step or visit rows.

| Field | Type | Notes |
|---|---|---|
| run_id | Text | |
| seed | Integer | The run's seed |
| env_idx | Integer | Which parallel environment |
| env_steps | 64-bit integer | Total agent decisions across all environments, counted after this one |
| wall_clock_s | Decimal | Seconds since the run started |
| episode_id | Integer | Counts from 0 within each environment |
| episode_step | Integer | Counts from 1 at the episode's first decision |
| action | Small integer | Index into the environment's action set |
| map_group, map_num, x, y | Small integer | The location after the decision |
| in_battle | Boolean | |
| milestone_count | Small integer | |
| reward_total | Decimal | The weighted reward the environment returned |
| reward_ plus component name | Decimal, nullable | One column per component |

Both `env_steps` and `wall_clock_s` are required. Sample efficiency and wall-clock efficiency are separate claims.

### Visit-count table

The per-step table is sampled, so the heatmap cannot use it for exact counts. Each environment counts one visit per decision, on the tile the player stands on after it, and flushes its counts once per checkpoint window of 500,000 `env_steps`. A run's visits sum to its `env_steps`. Several environments can write a row for the same tile and window; readers sum them. This table is exact and small.

| Field | Type | Notes |
|---|---|---|
| run_id | Text | |
| window_start_step | 64-bit integer | 0, 500,000, 1,000,000 and so on. A decision with `env_steps` e falls in the window starting at (e - 1) // 500,000 * 500,000 |
| map_group, map_num, x, y | Small integer | |
| visits | Integer | |

### Per-episode table

One row per finished episode. An episode the end of the run cuts off is not written. LLM attempts are episodes too.

| Field | Type | Notes |
|---|---|---|
| run_id, seed, env_idx | As in the per-step table | |
| episode_id | Integer | Counts from 0 per environment, separately for evaluation. With run_id, is_eval and env_idx, names the episode |
| env_steps | 64-bit integer | Training decisions so far, at the episode's last decision |
| wall_clock_s | Decimal | At the episode's last decision |
| episode_return | Decimal | Sum of `reward_total` over the episode |
| episode_length | Integer | Decisions in the episode |
| reward_ plus component name | Decimal, nullable | The component's raw total over the episode |
| badge_count, party_level_sum, story_flags | Small integer | At the episode's end |
| unique_tiles | Integer | Distinct `(map_group, map_num, x, y)` this episode |
| milestone_count | Small integer | At the episode's end |
| start_state | Text | The manifest name |
| archive_start | Boolean | Whether the start came from the archive |
| end_reason | Text | `truncated` for the step budget. LLM attempts also use `goal`, `action_cap` and `dollar_cap` |
| goal_reached | Boolean | Whether the start state's goal became true at any point in the episode. False when the start state has no goal |
| goal_step | Integer, nullable | The `episode_step` at which the goal first became true. Null when it never did |
| is_eval | Boolean | An evaluation episode: fixed start state, archive off |
| checkpoint_step | 64-bit integer, nullable | The training `env_steps` of the checkpoint being evaluated. Null for training episodes and LLM attempts |
| llm_model | Text, nullable | Null for RL runs, like the four below |
| llm_tool_level | Small integer, nullable | 0, 1 or 2 for L0 to L2 |
| llm_tokens_in, llm_tokens_out | 64-bit integer, nullable | |
| llm_cost_usd | Decimal, nullable | |

### Per-run table

Two rows per run, because a Parquet file cannot be updated in place. The start row is written at launch with status `running` and `env_steps` and `wall_clock_s` at 0. The end row repeats every field with status `finished` or `crashed` and the final totals. A run's current row is its end row if one exists, else its start row. A run killed without a chance to write its end row stays `running`; a stale `wall_clock_s` against the clock says it died.

| Field | Type | Notes |
|---|---|---|
| run_id | Text | What every other table joins on |
| status | Text | `running`, `finished` or `crashed` |
| experiment_id, arm, algorithm, trainer | Text | |
| seed | Integer | |
| env_id, backend, rom_sha1, start_state, obs_mode | Text | |
| frames_per_decision, num_envs | Small integer | |
| reward_config_hash, git_commit | Text | |
| hyperparameters | Text | The resolved config as JSON, with gamma and `reward_weights` (name to weight) |
| step_log_every | Small integer | N for the per-step table |
| machine | Text | The machine name from config, or the hostname when none is set |
| env_steps | 64-bit integer | Training decisions. The end row's visit counts sum to it |
| wall_clock_s | Decimal | Total wall-clock time in seconds |

### Checkpoints

Every run saves a model checkpoint every 500,000 steps, with normalization statistics, named by run id and step. Checkpoints and per-step logs are never deleted before the showcase.

## Conformance tests

These are blocking. Nothing merges to `main` until they pass.

1. Determinism: the same state and the same inputs produce the same RAM hash at every step.
2. Save state round trip: saving, stepping and loading restores the saved memory exactly.
3. Screen: the frame has the right shape and type, and is a copy.
4. ROM guard: a ROM with an unknown hash raises an error.
5. Reader sanity: a captured state with a known situation reads back exactly those values.
6. Component isolation: on a step where nothing changes, every reward component returns zero. Same on the reset step.
7. Exploit bounds: each scripted exploit policy earns less than its stated bound.
8. Schema validation: every logged row matches the frozen schema.

Tests 1 to 3 live in `tests/conformance/` and run against every backend in the registry. A new backend adds one registry entry and must pass them. The ROM guard is tested in `tests/test_rom.py`. Tests that need a ROM are marked and skip when the ROM is absent, so CI runs the fake backend and members run the real ones.

## Fake backend and fake logger

The fake backend is a small grid game written in Python with a memory map of its own: a player position, two rooms joined by a door, a chest that sets a flag, and an RNG that advances every frame. It also has five bytes that nothing in the game writes (party level sum, badge byte, milestone count, party HP percentage, text box); tests set them directly. It exists so that the reward registry, the environment, the MCP server and the logging pipeline can all be built and tested without a ROM.

The fake logger writes synthetic runs in the exact frozen schema: a random walker over real FireRed map ids, from the player's bedroom to Pewter Gym, that earns the badge, its goal, in about one episode in four. At each checkpoint it runs two evaluation episodes. The same seed gives identical tables, and a second run under the same run id is refused. `uv run python -m algoarcade.logs.fake --run-id fake0 --seed 0` writes a 2,000,000-step run with four environments under `data/runs/`. In week 6 everyone builds their panel against it before pointing at real runs.

## MCP tool surface

The MCP server exposes the same harness to a language model. It uses the official MCP Python SDK over stdio. One server process owns one emulator, created when the server starts, and every evaluation attempt gets a fresh process. State belongs to the process, not to a session: the 2026-07-28 protocol revision has none. Nothing but protocol messages may reach standard output, and logging goes to stderr.

| Tool | Arguments | Returns | Level |
|---|---|---|---|
| press | `buttons`: a list of button names, each one of A, B, SELECT, START, RIGHT, LEFT, UP, DOWN, R, L | The GameState after pressing them in order, one decision each | L0 |
| get_state | none | The GameState | L1 |
| navigate_to | — (AGT-04) | The state after walking to a tile on the current map by A*. Stops on a battle, dialogue or warp and says which. | L2 |
| advance_dialogue | — (AGT-04) | The text shown, after pressing A until the box closes or a choice appears | L2 |

`get_state` and `press` return the GameState as structured JSON, one field per name. On the fake backend that is `room`, `x`, `y`, `chest_opened` and `frame`. The SDK sends a text copy of the same object beside it, indented two spaces by a constant no option reaches, so structured content is the compact payload a client reads. Screenshots arrive with the level option in AGT-04.

A level includes every tool below it. The server takes the level as a launch option, so the same code serves all three rungs of the ladder.

`.mcp.json` at the repo root launches the server through `uv`, so Claude Code and VS Code list the tools after `uv sync`.

Jayden updates this section when AGT-04 lands.

## Open questions before the freeze

- Frames per decision: does a 24-frame decision still land menu inputs reliably? OPS-02 checks 8, 16, 24 and 48.
- Number of environments: 12 or 16 on the training box? Set from the benchmark.
- Per-step sampling: is every fourth step enough for the reward decomposition chart? Decide before the freeze.
