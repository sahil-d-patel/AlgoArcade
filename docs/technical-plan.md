# AlgoArcade technical plan

Fine-tuned October 11, 2026. Interfaces are in [harness-spec.md](harness-spec.md). The week-by-week schedule is in [development-plan.md](development-plan.md).

## Overview

AlgoArcade is a reusable framework for training and comparing game-playing agents. We prove it works by teaching agents to play Pokémon FireRed. The framework is the deliverable and Pokémon is the demo.

We ordered it that way on purpose. If training stalls, we still ship a working harness, a reproducible experiment pipeline, a comparison method and documented results. A project that only succeeds if the agent beats Brock has no fallback.

### What makes this different

Most public Pokémon RL work targets Red on the Game Boy. We target FireRed on the GBA.

- The pokefirered decompilation gives us named memory symbols, so no address comes from a wiki.
- FireRed is a remake of Red, so published reward designs and milestones carry over.
- We found no public RL result that reaches Brock on FireRed or LeafGreen. The closest GBA work is on Emerald.
- The same harness serves a language model through MCP, so we can compare trained policies against an LLM agent on identical save states.
- A second, non-Pokémon game runs through the framework from week 2 (Super Mario Land), so "reusable" is tested continuously instead of claimed at the end.

### Success criteria

| Tier | What it means |
|---|---|
| Must | FireRed and Super Mario Land run through one backend interface with a conformance suite. A PPO baseline and three method arms, five seeds each (E1 to E4). The LLM agent evaluated on three segments across a tool ladder. Static dashboard. Writeup. |
| Should | Viridian City from a cold start. Pewter City from the post-parcel state. A reward hack catalogue with video evidence. The second game trained with an untouched PPO config. |
| Stretch | First badge (Brock) from the post-parcel state, then from a cold start. LLM trajectories used as a start-state curriculum for RL (E8). |

We are aiming at the stretch tier, and the schedule has the Brock runs in it. If they miss by November 22, the Must and Should tiers still stand, we report the miss as it is, and the runs continue after November 30.

## Team

Seven people. Sahil leads and Akshaj co-leads; both float to wherever help is needed. Five members hold lanes through week 5, each with one assigned issue a week. In weeks 6 and 7 everyone builds the visuals for the data their lane produced.

| Who | Lane | Weeks 1 to 5 | Weeks 6 and 7 |
|---|---|---|---|
| Sahil, lead PM | PM | Training box, throughput benchmark, logging and schema freeze, run queue, weekly video review, freeze decisions | Results freeze, writeup lead |
| Akshaj, Co-PM | PM | GitHub and review gate, PyBoy backend, start states, video renderer, statistics module, second game | Dashboard, showcase assets |
| Rishi, Emulation Engineer | Environment | mGBA backend, Gymnasium env, archive resets, observations | Run registry, video rendering, reproducibility check |
| Prince, Memory Engineer | Environment | Address table, StateReader, map data, battle state | Kanto map and heatmap, RE writeup |
| Isaac, RL Engineer | Training | Training runner, owned PPO trainer, PQN, E1, E4 | Analysis and comparison panels |
| Pri, Reward Engineer | Training | Reward components, exploit tests, baseline, RND, E3, E5 | Analysis, reward decomposition, hack catalogue |
| Jayden, Agent Engineer | Agent | MCP server, tools, LLM loop, E7 | Agent video grid, cost table, E8 |

Environment gets two people because everything else waits on it. The LLM arm is decoupled from training, so one engineer owns it and Akshaj backs them up.

The PMs each own a few decoupled items that nobody else's work waits on for more than a week. The rest of their time floats: any issue that is not in review by Thursday gets a PM pairing session.

## Architecture

The harness is built in layers. The lower layers are shared by everything above them.

1. The ROM sits at the bottom, pinned to one revision by its SHA-1.
2. GameBackend runs the emulator. It presses buttons, grabs frames, reads memory, and saves and loads states.
3. StateReader turns raw memory into a typed GameState: location, badges, party, progress flags.
4. Two consumers sit on top. AlgoArcadeEnv is a Gymnasium environment for RL. The MCP server exposes tools to a language model.
5. Both consumers write to one logging pipeline. Parquet files are the source of truth. Weights & Biases is the live view.
6. The dashboard reads the Parquet logs.

The two consumers run at very different speeds. An RL policy takes hundreds of steps per second and needs tens of millions. The LLM takes a step every few seconds and gets a few hundred.

There are three backends:

| Backend | Runs | Used for |
|---|---|---|
| FakeBackend | A toy grid game written in Python, no ROM | Unit tests and CI |
| PyBoyBackend | Super Mario Land (Game Boy) | Pipeline testing from week 2, and the second-game result |
| MgbaBackend | Pokémon FireRed (GBA) | The main experiments |

To add a game you write a StateReader, a reward config and a set of start states. Nothing downstream changes. The Super Mario Land port is how we measure that claim: we keep a diary of every file it touches.

## Simulation layer

Owned by the Environment pair.

### Emulation

FireRed runs on mGBA through the prebuilt `hanzi/libmgba-py` binding, release 0.2.0-2. A fetch script downloads the archive for the host OS and checks it against a SHA-256 stored in the repo. Nobody compiles mGBA.

| OS | Extra step |
|---|---|
| Windows x64 | None |
| macOS arm64 | `brew install mgba` |
| Ubuntu 24.04 or later | `sudo apt install libmgba0.10t64` |

None of this has been run on our machines yet. Proving it starts in week 1 (ENV-01) and finishes in week 2 (ENV-04). The binding's repo has had no push since December 2024, so we mirror the three archives we use. If the binding fails on someone's machine, the fallback is `libretro.py` with a prebuilt mGBA core.

We do not use PyGBA or the `mgba` package on PyPI. That package has no Windows or Apple Silicon wheel.

The ROM is not in the repo. Everyone supplies their own, and the harness checks its SHA-1 on load and refuses to run on a mismatch. A wrong revision would not crash. It would produce believable, wrong rewards.

### Throughput

Training runs on two machines: a Windows desktop, which we call the training box, and a Mac. Emulation is CPU-bound, and the networks are small, so the GPU is not the bottleneck. Both machines are also the easy cases for the emulator binding: Windows needs no system package and macOS needs one Homebrew install.

We have not measured throughput. No published number exists for mGBA driven from Python on a GBA game. Until the week 2 benchmark (OPS-02), every budget in this plan assumes 500 agent steps per second, about 43 million steps a day. For scale, Pleines et al. report about 392 agent steps per second for Pokémon Red on a Ryzen 7 2700X.

The benchmark measures frameskip 8, 16, 24 and 48 against 4, 8, 12 and 16 environments, with rendering on and off. Every compute number in this document gets replaced from that table.

We run one emulator per process. The vectorization library is not where the time goes: the PufferLib paper puts plain PufferLib, Gymnasium and Stable-Baselines3 within about 6% of each other on Pokémon Red. Async pooling was worth about 40% there, which we can test with Gymnasium's own async vector env.

### Decisions and determinism

One agent decision is 24 frames: hold the button for 8, release for 16. That is the convention in the Red literature, and one tile of walking takes 16 frames.

The action space is six buttons: up, down, left, right, A, B. Start and a no-op are config options. Start opens SAVE and OPTIONS, and nothing before Brock needs it.

FireRed has no real-time clock, and its RNG advances once per frame from a hardware timer. So a save state plus an input sequence fully determines what happens. That is good for replay and bad for exploration, so every reset adds a random number of no-op frames. A conformance test hashes RAM after a scripted input sequence, twice, and fails if the hashes differ.

### Start states and segments

Episodes start from save states, never from the title screen.

| State | Where | Used for |
|---|---|---|
| S0 | Player's bedroom, new game, intro replayed by script | The cold-start runs |
| S1 | Just after delivering Oak's Parcel, with Squirtle | Development, all multi-seed experiments, the first Brock runs |
| seg_01 to seg_05 | House, Pallet to Route 1, rival battle, Route 1 to Viridian, Viridian Forest | LLM against RL on identical tasks |

Both S0 and S1 have text speed set to fast and battle animations off, and the reader asserts that at reset. Pleines et al. report 33% fewer steps to Brock with those settings.

Every published Red agent starts after the parcel quest because the backtrack blocked learning. We do the same for development and treat the true cold start as the stretch. Squirtle is forced: in Red, Brock success was 0.99 with Squirtle and 0.69 with Charmander.

## Reverse engineering

Owned by the Memory Engineer. Every reward and every logged value is read from the game's live memory.

### Why we read memory instead of pixels

Reward signals are exact: a badge is a bit, not something guessed from the screen. Evaluation is objective: reaching Viridian City is a map id check. And a small state vector can feed a small network.

### Method

Step 1 is symbol lookup. The pret/pokefirered project builds the game from source in CI and publishes the linker's symbol files on a branch. We fetch those at a pinned commit. Nobody on the team builds the decompilation. We search the symbol file for the variables we need and keep a small, hand-curated address table in the repo, keyed by ROM SHA-1, with a note on where each address came from.

Step 2 is save-state diffing, for anything the symbols do not explain and as an onboarding exercise. Save state A, make exactly one change in the game, save state B, compare memory, repeat with a different change to remove coincidences, then confirm by writing a value and watching the game react. The Memory Engineer's diff script makes this a ten-minute exercise for anyone who wants to try it.

Step 3 is validation. Every field gets a test that loads a save state with a known situation and checks that we read exactly the expected value.

### Traps

- Save block pointers move. `gSaveBlock1Ptr` and `gSaveBlock2Ptr` point at blocks that relocate on every warp and every return from a battle or menu. We read the pointer on every access and never cache it. This is the most likely source of silently wrong data.
- Money is encrypted. It is XORed with a key that changes on every relocation.
- Party data is half encrypted. Level, HP, max HP and status are plain. Species and moves sit in a shuffled, encrypted block, and we validate its checksum before trusting it.
- Flags are not all progress. Temporary flags reset on every map load. We count only ranges that go up: the badge byte, trainer flags, world-map flags and a whitelist of story flags.
- Coordinates are per map. A position only means something as `(map_group, map_num, x, y)`.
- Revisions differ. RAM symbols match between FireRed rev 0 and rev 1, but ROM function addresses do not, so the table is keyed by hash and not by game title.

### Map data

The decompilation also holds every map's layout, collision bits and connections. We parse those into a world coordinate table and a walkability grid, generated locally and never committed. Three things use it: the symbolic observation, the LLM's pathfinding tool, and the exploration heatmap.

The decompilation has no license, and the tile art is Nintendo's. We commit our own code and the small address table. ROMs, save states, symbol files, map binaries and tile renders stay in the git-ignored `data/` folder. Public visuals use a schematic map drawn from collision data.

## Reward design

Owned by the Reward Engineer.

### The problem

Pokémon has sparse rewards and a very long horizon. The real reward (a badge) comes after tens of thousands of correct steps, and no algorithm we can run finds that by chance. So we add shaped rewards, and agents reliably find ways to exploit shaped rewards.

### Components

Every component has a name, a weight in config, and its own column in the logs.

| Component | Trigger | Starting weight | Default |
|---|---|---|---|
| events | Each newly set flag on the story whitelist | +1.0 | on |
| explore | First visit to a `(map_group, map_num, x, y)` this episode | +0.02 | on |
| level | Increase in party level sum, capped at +1.0 per step | +0.5 per level | on |
| badge | A new badge | +10.0 | on |
| milestone | A new entry on the ordered Pallet-to-Brock list | +2.0 | on |
| heal | Increase in party HP fraction | +0.3 times the increase | off |
| blackout | Party fainted and warped | -5.0 | off |
| anti_loop | Revisiting a tile within the last 30 steps | -0.01 | off |
| anti_spam | A press with no effect while a text box is open | -0.02 | off |
| time | Every step | -0.001 | off |

The default set is positive only, and the evidence for that is consistent. Pleines et al. had their heal reward farmed. The best pure-RL entry in the PokéAgent Challenge used positive-only milestone rewards. An open Emerald project reports that every penalty it tried backfired. The penalty components stay in the registry, and whether they help on FireRed is experiment E5.

Weights are starting points. The ratios matter most: badge above everything, events above level, and exploration over a full episode worth a few events.

### Exploits we expect

Anyone adding a component writes in the pull request how an agent would exploit it. Each known exploit is also a test: a scripted policy that tries it and must earn less than a stated bound.

- Tile oscillation: bouncing between two tiles to farm novelty. Novelty uses a permanent per-episode visited set keyed on coordinates, never screen hashes.
- Grass camping: leveling forever in the first tall grass. The per-step level reward is capped and events outweigh levels.
- Blackout farming: fainting to get a free heal. The heal component is off by default and never fires on a blackout transition.
- Menu spam: opening and closing menus because the screen changes. Start is not in the default action set.
- Reset farming: any reward paid on reset teaches the agent to die fast. Every component returns zero on the reset step.

### Catching what we did not predict

Per-component reward charts show when one component's share keeps growing while progress stalls. A divergence check flags runs where return rises and unique tiles do not. And every week one PM watches five minutes of agent footage and writes down what they see. Every exploit above is obvious in thirty seconds of video and can hide for a month in a reward curve.

## Training

Owned by the Training pair.

### Methods

| Role | Method | Why |
|---|---|---|
| Primary | PPO | The only method with a track record on Pokémon. This is the one we tune. |
| Value-based contrast | PQN | Q-learning built for parallel environments: no replay buffer, no target network. It consumes the same rollouts as PPO, so the comparison holds sampling constant. |
| Exploration | Count-based against RND | Counts are the proven baseline. Pleines et al. list RND as untested future work. |
| Frontier return | Archive resets | When any environment hits a new milestone, its save state enters an archive that later resets sample from. The pokemonred_puffer project uses a close cousin of this. |

What we do not use and why. DQN: a replay buffer is the wrong tool for sixteen parallel emulators, and PQN is the current answer to the same question. ICM: a third exploration arm costs about two box-days and tells us little that RND does not. A2C is a weaker PPO. SAC is for continuous actions. A third-party measurement puts DreamerV3 at about 21 environment frames per second on an RTX 3090. PufferLib's current version is a CUDA-only C trainer that cannot host a Python environment.

### Trainer

Weeks 1 to 3 run on Stable-Baselines3 PPO, because a new member can get a learning curve in an evening.

From week 4 we train on our own single-file PPO, derived from CleanRL with attribution. Stable-Baselines3 has one value head and episodic returns, and RND needs two heads and non-episodic intrinsic returns. PQN is not in it at all. Running every arm through one file also makes the comparison fair, and it is code we can explain line by line.

The switch has a gate. By November 8 our trainer must match Stable-Baselines3 on CartPole, Super Mario Land and one FireRed segment, five seeds each, with overlapping intervals. If it does not, we stay on Stable-Baselines3, use the weaker single-head RND and QR-DQN, and say so in the writeup.

### PPO starting point

| Parameter | Value | Note |
|---|---|---|
| Parallel environments | 16 | Set from the benchmark |
| Steps per environment per update | 2,048 | Pleines et al. found 2,048 beat 512 and 1,024 |
| Epochs per update | 3 | |
| Learning rate | 0.0003 | Constant, so runs can resume |
| Gamma | 0.997 | Never the 0.99 default |
| GAE lambda | 0.95 | |
| Entropy coefficient | 0.01 | |
| Clip range | 0.2 | |

**Gamma is the most likely silent failure in the project.** The effective horizon is about 1 / (1 - gamma). The library default of 0.99 gives 100 steps, which cannot cover a 2,000-step segment. An agent at the default will not throw an error. It will not learn, and someone will spend two weeks debugging the reward function. Published Red projects use 0.997 or 0.998.

We normalize observations. We do not normalize the summed reward: components are scaled by hand and logged raw, which keeps the per-component charts readable.

Episodes end only on a step budget. That is truncation, not termination, and the trainer bootstraps the value at the cut.

### Observations

`symbolic` is a scaled state vector plus a grid crop centered on the player with walkability and visited-tile channels. `pixels` is the screen at 72 by 80 greyscale, three frames stacked, plus the same vector. Comparing them with everything else fixed is experiment E2. The symbolic crop is only possible because we parse the game's own map data.

### Statistics

A single seed is noise. Every headline claim uses five seeds, and three-seed results are labeled exploratory. We report the interquartile mean with stratified bootstrap intervals, following Agarwal et al. (2021). The rliable library that paper shipped is archived, and its metrics are a few calls into scipy, so we implement them in the repo with tests.

Results are plotted against environment steps and against wall-clock hours. Those are different claims. Every run records its git commit, ROM hash, reward config hash and seed.

### Run queue

Sahil builds this in week 4. An experiment is a small Python file naming a base config and the factors to vary. The queue expands it to runs and executes them back to back on the training box. It keeps going after a crash, tags every run with its experiment id, posts to the team channel when something fails, and saves a checkpoint every 500,000 steps.

If one thing in this plan cannot slip, it is this. The experiment matrix costs machine time, and only a queue turns machine time into results without someone launching jobs by hand.

## Experiments

We change one thing at a time from a fixed baseline: PPO, symbolic observations, count-based exploration, archive off, starting from S1.

| ID | Question | Arms | Seeds | Owner |
|---|---|---|---|---|
| E1 | Policy gradient against value-based, same sampling | PPO, PQN | 5 | Isaac |
| E2 | Does a symbolic observation match pixels | symbolic, pixels | 5 | Rishi |
| E3 | Does learned novelty beat counts | none, count, RND | 5 | Pri |
| E4 | Does returning to the frontier help | archive off, on | 5 | Isaac |
| E5 | Do penalties help or hurt | positive-only, plus penalties | 3 | Pri |
| E6 | How far does the best config get | Brock from S1, then from S0 | 3, then 2 | Isaac, Sahil |
| E7 | An LLM agent across a tool ladder | L0, L1, L2 by model by segment | 10 attempts | Jayden |
| E8 | Do LLM trajectories help RL as a curriculum | curriculum, none | 3 | Jayden |
| E9 | Does the framework transfer | Super Mario Land, untouched PPO config | 8 to 10 | Akshaj |

E1 to E4 share one baseline arm, so they need 30 runs, not 45.

### Compute budget

These numbers are arithmetic on the 500 steps per second assumption. They get replaced after OPS-02.

| Block | Runs | Steps each | Box-days at 500 steps/s |
|---|---|---|---|
| E1 to E4 | 30 | 15M | 10.4 |
| E5 | 3 new | 15M | 1.0 |
| E6 from S1 | 3 | 100M | 6.9 |
| E6 from S0 | 2 | 100M or more | 4.6 or more |

That is about 23 box-days. The queue goes live November 8 and results freeze November 22, which leaves 14 days, or 28 machine-days across two machines. On paper that fits. In practice most arms only become ready during week 5, so it does not.

So the Mac is the second training machine. The training box runs E1, E3, E4 and the S1 Brock runs. The Mac runs E2, E5, E9 and the cold-start runs. The Brock runs are the first thing to spill past November 22: they keep running, and the writeup reports where they stood. If the benchmark comes in low, ablation runs drop from 15M to 10M steps before any arm loses seeds.

How many steps Brock takes on FireRed is not known. Our estimate is 50 to 150 million from S1 and 100 to 300 million from S0, built from Red results on faster emulators. Treat it as a guess.

## LLM arm

Owned by the Agent Engineer. We wrap the same harness as an MCP server, let a language model play through it, and compare it with the trained agents on the same segments.

### Why it is in scope

In June 2026 Anthropic reported that Claude Fable 5 finished FireRed with a vision-only harness, and that earlier models struggled even with helper tools. So the open question is no longer whether an LLM can play. It is how much the tools matter at each model tier. That is an experiment we can run for a few hundred dollars.

It also protects the demo. If RL underdelivers, we still have an agent that visibly plays Pokémon. And it is decoupled: a slip here blocks nothing else.

### Tools

| Tool | What it does |
|---|---|
| get_state | The GameState as structured JSON, with an optional 2x-upscaled screenshot |
| press | A short sequence of buttons |
| navigate_to | A* over the current map's collision grid, read live from memory. Stops on a battle, dialogue or warp and says why. |
| advance_dialogue | Presses A until the text box closes or a choice appears, and returns the text |

Every action tool returns the state after the action. The rule we follow: the more a decision costs, the bigger each action should be.

The experiment is a ladder. L0 is `press` and a screenshot. L1 adds the state JSON. L2 adds `navigate_to` and `advance_dialogue`. Published harnesses credit pathfinding and RAM-grounded state with most of their gains. Our ladder mirrors the scaffold tiers in the PokéAgent harness, moved to FireRed and scored against RL on the same states.

### Loop, cost and evaluation

The server uses the official MCP Python SDK over stdio. One server process owns one emulator, and every attempt gets a fresh process. A small loop of our own drives the model through the server, logs tokens on every request, and enforces a hard cap on actions and dollars per attempt.

The model is a config value. The plan is the full ladder on a small model and the top rung on a large one. Our estimate is $6 to $13 for a 300-action attempt on the large model with prompt caching, and $200 to $500 for the whole arm. That is arithmetic on list prices and assumed token counts. A 20-action pilot in week 2 replaces it with a measurement.

Each configuration gets at least ten isolated attempts per segment. We report success rate with an exact binomial interval, button presses to the goal (the unit shared with RL), and dollars per success. Success is graded from memory, never from the path the agent took.

### Connecting the two arms

E8 uses the LLM's successful runs as a curriculum for PPO. We save the emulator state every few actions along the run, start PPO episodes from those states, and move the start point backward as success rises. Starting from demonstration states is the method behind the single-demonstration Montezuma's Revenge result. We found no application of it to Pokémon.

## Visualization

All custom visualization happens in weeks 6 and 7 with everyone on it. Until then everyone stays on their lane.

Watching runs during training is not visualization work. Weights & Biases streams returns, per-component rewards, unique tiles and episode videos, and that is where we catch exploits. We need the free Academic plan: the pricing page caps the free tier at five seats and we are seven. OPS-01 confirms that when applying.

Building visuals last means anything we did not log is gone. Three rules protect us. At the schema freeze each lane confirms the logs hold every field its visual needs. We keep every per-step log. Every run saves a checkpoint every 500,000 steps, so we can render new video without retraining. The fake logger emits synthetic data in the exact final schema, so every panel is built on small data first.

| Visual | What it shows | Owner |
|---|---|---|
| Kanto exploration heatmap | Where the agent went, on a schematic map, animated across training, with a milestone timeline | Prince |
| Comparison panels | IQM with bootstrap bands for E1 to E4, against steps and against wall-clock | Isaac |
| Reward decomposition | Which component drove the return over training; the hack catalogue with clips | Pri |
| Agent video grid | RL arms and the LLM playing the same save states side by side; cost to first success | Jayden |
| Run registry | Every run's config joined to its results; the second-game panel | Rishi |
| Dashboard | All of it in one static site | Akshaj |

The dashboard is a Quarto site rendered to static HTML, with DuckDB reading Parquet at build time and Plotly for the interactive figures. Nothing depends on a live server during the showcase. If a custom panel is not ready, the matching Weights & Biases chart stands in.

## Working practices

- One sync a week, on Sunday, where each person demos that week's deliverable. Everything else is async.
- All work goes through pull requests, including solo work.
- Every lane has a second person who can pick it up. For the Agent lane that is Akshaj.
- By Thursday each person posts two sentences: what they did and what blocks them.
- Nobody debugs on the training box. Debug locally with one environment and 1,000 steps.
- In week 1 everyone ships one small starter task through a reviewed pull request. It takes about two hours, proves the setup and the review flow, and tells us early who is in.

The issues are sized at about five hours a week per member. That is more than the four hours a club project usually gets, and it is what aiming at the stretch tier costs.

## Timeline

| Week | Due | Gate |
|---|---|---|
| 1 | Oct 18 | Each person's starter task merged through a pull request |
| 2 | Oct 25 | Emulators, memory reads, training runner and MCP server run on every machine |
| 3 | Nov 1 | Logging schema frozen. First PPO curve on FireRed |
| 4 | Nov 8 | Baseline clears the early segments across seeds. Run queue owns the box. Trainer parity decided |
| 5 | Nov 15 | Feature freeze. Every experiment and the Brock runs queued. LLM arm evaluated. Second game trained |
| 6 | Nov 22 | Everyone on analysis and visualization. Results frozen |
| 7 | Nov 29 | Dashboard, writeup, reproducibility check, showcase assets. Tag v1.0 on Nov 30 |

Four dates cannot move.

- Schema freeze, November 1. Runs logged before it are thrown away. If the StateReader is not validated on real memory by then, the freeze moves to November 8 instead of freezing a schema nobody has tested.
- Run queue, November 8. Experiments do not fit without it.
- Feature freeze, November 15. After it, only fixes and visualization merge.
- Results freeze, November 22. Thanksgiving is the 26th, so week 7 is writing only.

Week 4's rule is to reproduce a known result with no new ideas. It will be frustrating. It gives us a baseline that every later experiment varies one factor from, instead of five experiments that cannot be compared.

## Risks

| Risk | Likelihood | Impact | Plan |
|---|---|---|---|
| The prebuilt mGBA binding fails on a member's machine | Medium | High | It is the first issue of week 1, and tested on three OSes in week 2. Archives are mirrored. libretro.py is the fallback. Super Mario Land on PyBoy keeps the pipeline moving meanwhile. |
| Throughput is well under 500 steps per second | Medium | High | Measured in week 2. Below 250 we cut E2 and E5 first and shorten ablation runs before touching seed counts. |
| A pointer or address bug gives silently wrong data | Medium | High | Pointers are never cached. Every field has a validation test. The ROM hash is checked on load. |
| Reward hacking eats weeks | High | Medium | Positive-only defaults, exploit tests, per-component charts, weekly video review. |
| The owned trainer misses the parity gate | Medium | Medium | Decided on November 8 either way. The fallback on Stable-Baselines3 is defined. |
| The run queue slips | Low | High | It has one owner, one week and nothing else scheduled against it. |
| Brock is not reached by November 22 | High | Low | It is the stretch tier. The result is reported as it landed and the runs continue. |
| A member leaves | Medium | High | Each lane has a second person. If Jayden leaves, Akshaj keeps E7 and E8 is dropped. |
| A needed field was never logged | Medium | High | Field check at the schema freeze. Per-step logs and checkpoints are kept. |
| LLM arm cost runs over | Medium | Low | Hard caps per attempt in the runner. The pilot sets the budget before the evaluation starts. |
| Game data ends up in git | Low | High | A pre-commit hook rejects ROMs, save states and anything under `data/`. |
| Scope creep toward beating the game | High | High | Everything is scoped to Brock. Nothing past the first badge is planned. |

## Sources

Read by a researcher and re-checked by a second reader, unless marked "read once".

- Pleines et al., "Pokémon Red via Reinforcement Learning": https://arxiv.org/abs/2502.19920
- Whidden, PokemonRedExperiments (v2 baseline): https://github.com/PWhiddy/PokemonRedExperiments
- The pokemonred_puffer writeup: https://drubinstein.github.io/pokerl/
- Mudireddy and Patibandla (Texas A&M), "PokeRL": https://arxiv.org/abs/2604.10812
- The PokéAgent Challenge: https://arxiv.org/html/2603.15563
- Continual Harness: https://arxiv.org/html/2605.09998 (read once)
- RL-emerald, an open Emerald PPO project: https://github.com/rkhexed/RL-emerald (read once)
- pret/pokefirered and its symbols branch: https://github.com/pret/pokefirered/tree/symbols
- hanzi/libmgba-py: https://github.com/hanzi/libmgba-py
- pokebot-gen3, which ships the same binding to end users on three OSes: https://github.com/40Cakes/pokebot-gen3
- PyBoy: https://github.com/Baekalfen/PyBoy
- The PufferLib paper: https://arxiv.org/html/2406.12905v1
- Agarwal et al., "Deep RL at the Edge of the Statistical Precipice": https://arxiv.org/abs/2108.13264
- Burda et al., "Exploration by Random Network Distillation": https://arxiv.org/abs/1810.12894 (read once)
- Gallici et al., "Simplifying Deep Temporal Difference Learning" (PQN): https://arxiv.org/abs/2407.04811 (read once)
- Ecoffet et al., "First return, then explore" (Go-Explore): https://arxiv.org/abs/2004.12919 (read once)
- Salimans and Chen, "Learning Montezuma's Revenge from a Single Demonstration": https://arxiv.org/abs/1812.03381
- CleanRL single-file implementations: https://github.com/vwxyzjn/cleanrl (read once)
- Anthropic, Claude Fable 5 announcement: https://www.anthropic.com/news/claude-fable-5-mythos-5
- MCP Python SDK: https://py.sdk.modelcontextprotocol.io/ (read once)
