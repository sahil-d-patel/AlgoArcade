# AlgoArcade harness spec

Fine-tuned October 11, 2026. This defines the interfaces every part of AlgoArcade builds against, and the logging schema. Interfaces can change until November 1. The logging schema freezes on November 1, and any change after that means rerunning training.

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

A StateReader is written per game. It reads memory through a GameBackend and returns a GameState: a frozen record of everything we care about. The protocols live in `src/algoarcade/games/base.py`.

The framework asks one thing of every GameState: a `location`, a hashable tuple that identifies where the player is. Exploration rewards, the archive and the heatmap all key on it. For FireRed it is `(map_group, map_num, x, y)`.

### FireRed GameState fields

| Group | Field | Meaning |
|---|---|---|
| Position | map_group, map_num | Which map the player is on |
| Position | x, y | Tile coordinates on that map |
| Progress | badge_flags, badge_count | The badge byte and its popcount |
| Progress | story_flags | Count of set flags on the story whitelist |
| Progress | trainers_defeated | Popcount of the trainer flag range |
| Progress | milestone_count | How far down the ordered Pallet-to-Brock list |
| Progress | pokedex_owned, pokedex_seen | Pokédex counts |
| Party | party | Up to six Pokémon: species, level, HP, max HP, status, moves |
| Party | party_level_sum, party_hp_frac | Totals across the party |
| Context | money | Decrypted |
| Context | in_battle | Whether a battle is active |
| Context | text_active, script_active, controls_locked | What the game is waiting on |
| Battle | own and enemy HP, level, moves, PP | Only meaningful while in_battle is true |

Fields arrive in stages: position, party and badges in week 2 (ENV-05), the rest of the table in week 3 (ENV-08), battle and milestones in week 5 (ENV-13).

Every field has a validation test against a captured save state with a known value.

## FireRed memory map

These come from the pret/pokefirered symbol files at commit `3b401eaf14f193946899eb7c056d5ac02a470126` and were checked against the decompilation's source. The address table in the code is the source of truth. This section is the map.

| Symbol | Address (rev 0) | Notes |
|---|---|---|
| gSaveBlock1Ptr | 0x03005008 | Pointer. Dereference on every read. |
| gSaveBlock2Ptr | 0x0300500C | Pointer. Dereference on every read. |
| gPlayerPartyCount | 0x02024029 | |
| gPlayerParty | 0x02024284 | Six entries of 100 bytes |
| gBattleMons | 0x02023BE4 | Plaintext battle data |
| gMain | 0x030030F0 | inBattle is bit 1 of the byte at +0x439 |
| sMessageBoxType | 0x0203709C | 0 hidden, 2 normal, 3 auto-scroll |
| sGlobalScriptContextStatus | 0x03000EA8 | 0 running, 1 waiting, 2 shut down |
| sLockFieldControls | 0x03000F9C | |

Offsets inside SaveBlock1: x at 0x00, y at 0x02, map group at 0x04, map number at 0x05, money at 0x290, flags at 0xEE0. The eight badge flags are one byte at 0xFE4. Trainer flags run from 0xF80 to 0xFDF. The money key is in SaveBlock2 at 0xF20.

Inside a party entry: status at 0x50, level at 0x54, HP at 0x56, max HP at 0x58, all plain. The encrypted block is at 0x20 to 0x4F.

Rules:

- The save blocks relocate on every warp and every return from a battle or full-screen menu. The offset is one of 32 values. Never keep a dereferenced address across steps.
- Check that each pointer lands inside its expected range before using it. A null or out-of-range pointer means "not ready", not zero.
- Validate the party checksum before trusting species or moves.
- File-local symbols carry a numeric suffix that differs between builds. Match on the base name.
- No address enters the code without a symbol name and the commit it came from.

## Reward registry

A reward component is a small object with a name, a reset step that runs at the start of an episode, and a scoring step. The scoring step looks at the previous GameState and the current one and returns a number. The protocol lives in `src/algoarcade/rewards/base.py`.

Components are combined from a config that gives each a weight. The environment returns the weighted total as the reward and reports each component's raw value in the step info, so each can be logged on its own.

The components and their starting weights are in the technical plan, under Reward design. Five are on by default (events, explore, level, badge, milestone) and five are off (heal, blackout, anti_loop, anti_spam, time).

Rules:

- Every component returns exactly zero on a step where nothing changed.
- Every component returns exactly zero on the reset step.
- Novelty uses coordinate tuples in a permanent per-episode set. Never screen hashes, never a sliding window.
- A component that reads a field validates it first. Garbage memory must not pay.
- Each known exploit has a scripted policy in the tests that tries it and must stay under a stated reward.
- A pull request that adds or changes a component says how an agent would exploit it.

## AlgoArcadeEnv

AlgoArcadeEnv is the Gymnasium environment. It is built from a backend, a StateReader, a list of reward components, a start state, an observation mode, and a seed.

| Setting | Value |
|---|---|
| Decision length | 24 frames: hold 8, release 16 |
| Actions | Up, down, left, right, A, B. Start and a no-op are options. |
| Episode end | A step budget only. Reported as truncation. |
| Reset | Loads the start state, then waits a seeded random number of frames |
| Info | Each reward component, the location, the milestone count, and a save state when the archive asks for one |

Observation modes:

- `symbolic`: a scaled state vector, plus a grid crop centered on the player with a visited channel (week 3) and a walkability channel (week 5).
- `pixels`: the screen at 72 by 80 greyscale, three frames stacked, plus the same state vector.
- `both`.

The environment must pass Gymnasium's environment checker on all three backends.

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

State files are not in git. They live in `data/states/` on each machine, and Akshaj (Co-PM), who captures them, hands the files to members directly. The manifest hash proves everyone has the same file.

Every state has text speed fast and battle animations off. The reader asserts both at reset.

Evaluation always starts from the fixed state with the archive off, even when training used archive resets.

## Logging schema

Frozen November 1. Before the freeze, each lane confirms the logs hold every field its week 6 visual needs. Owner: Sahil (PM), in OPS-03.

Logs are Parquet files under `data/runs/run_id=<id>/`. Each writing process owns its own files and rolls to a new file every so many rows, because a Parquet file cannot be read until it is closed.

### Per-step table

Written every Nth step. The default N is 4.

| Field | Type | Notes |
|---|---|---|
| run_id | Text | |
| seed | Integer | |
| env_idx | Integer | Which parallel environment |
| env_steps | 64-bit integer | Total agent decisions across all environments |
| wall_clock_s | Decimal | Seconds since the run started |
| episode_id, episode_step | Integer | |
| action | Small integer | |
| map_group, map_num, x, y | Small integer | |
| in_battle | Boolean | |
| milestone_count | Small integer | |
| reward_total | Decimal | |
| reward_ plus component name | Decimal | One column per component |

Both `env_steps` and `wall_clock_s` are required. Sample efficiency and wall-clock efficiency are separate claims.

### Visit-count table

The per-step table is sampled, so the heatmap cannot use it for exact counts. Each environment also counts visits per tile and flushes them once per checkpoint window: run_id, window start step, map_group, map_num, x, y, visits. This table is exact and small.

### Per-episode table

Return, length, each component's total, badge count, party level sum, story flags, unique tiles, milestone count, start state name, whether the start came from the archive, and why the episode ended.

LLM attempts are episodes too. Four columns are empty for RL runs and filled for the LLM: model, tool level, tokens in and out, and cost in dollars.

### Per-run table

Written once per run: experiment id, arm, algorithm, trainer, seed, environment id, backend, ROM SHA-1, start state, observation mode, frames per decision, number of environments, reward config hash, git commit, hyperparameters, and total wall-clock time.

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

The fake backend is a small grid game written in Python with a memory map of its own: a player position, two rooms joined by a door, a chest that sets a flag, and an RNG that advances every frame. It exists so that the reward registry, the environment, the MCP server and the logging pipeline can all be built and tested without a ROM.

The fake logger writes synthetic runs in the exact frozen schema: a random walker over real FireRed map ids that sometimes earns a badge. It arrives with the logging pipeline in week 3 (OPS-03). In week 6 everyone builds their panel against it before pointing at real runs.

## MCP tool surface

The MCP server exposes the same harness to a language model. It uses the official MCP Python SDK over stdio. One server process owns one emulator, created when the server starts, and every evaluation attempt gets a fresh process. State belongs to the process, not to a session: the 2026-07-28 protocol revision has none. Nothing but protocol messages may reach standard output, and logging goes to stderr.

| Tool | Arguments | Returns | Level |
|---|---|---|---|
| press | `buttons`: a list of button names, each one of A, B, SELECT, START, RIGHT, LEFT, UP, DOWN, R, L. `screenshot`: optional, default false | The GameState after pressing them in order, one decision each | L0 |
| get_state | `screenshot`: optional, default false | The GameState | L1 |
| navigate_to | — (AGT-04) | The state after walking to a tile on the current map by A*. Stops on a battle, dialogue or warp and says which. | L2 |
| advance_dialogue | — (AGT-04) | The text shown, after pressing A until the box closes or a choice appears | L2 |

`get_state` and `press` return the GameState twice: as structured content and as one compact JSON text block, one field per name. On the fake backend that is `room`, `x`, `y`, `chest_opened` and `frame`. With `screenshot` true, a PNG of the screen upscaled 2x (480x320 on FireRed) follows the text. The server takes the game as a launch option, `--game`; `fake` is the only entry until ENV-04 and ENV-05 land.

The loop in `src/algoarcade/agent/loop.py` drives one attempt: it lists the tools once, keeps an append-only history with a cache breakpoint on the system prompt and the automatic one on the last turn, prices every response from `usage`, and stops at the goal, the action cap, the dollar cap, or when the model stops calling tools. One action is one button press. Each request is one line of `steps.jsonl` under `data/llm/<name>/`: tokens by kind, dollars, buttons pressed, the model's text and the state after.

A level includes every tool below it. The server takes the level as a launch option, so the same code serves all three rungs of the ladder.

`.mcp.json` at the repo root launches the server through `uv`, so Claude Code and VS Code list the tools after `uv sync`.

Jayden updates this section when AGT-04 lands.

## Open questions before the freeze

- Frames per decision: does a 24-frame decision still land menu inputs reliably? OPS-02 checks 8, 16, 24 and 48.
- Number of environments: 12 or 16 on the training box? Set from the benchmark.
- Per-step sampling: is every fourth step enough for the reward decomposition chart? Decide before the freeze.
