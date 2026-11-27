# Development plan: October 11 to November 30

Seven weeks, seven Sunday deliverables, 48 issues. Each person owns one issue a week: a two-hour starter in week 1, then about five hours a week. The issue ids below match GitHub; each issue carries its own description and acceptance criteria.

What we are building and why is in [technical-plan.md](technical-plan.md). This file is who does what, by when.

## The team

| Who | Role | Lane | What they own across the project |
|---|---|---|---|
| Sahil | Lead PM | PM, floating | Training machines, logging schema, run queue, video review, the four freeze dates |
| Akshaj | Co-PM | PM, floating | GitHub and reviews, PyBoy backend, start states, evaluation tooling, the second game, dashboard |
| Rishi | Emulation Engineer | Environment | The mGBA backend, the Gymnasium env, archive resets, observations |
| Prince | Memory Engineer | Environment | Addresses, StateReader, map data, battle state, the Kanto map |
| Isaac | RL Engineer | Training | Training runner, our PPO trainer, PQN, the analysis of E1, E2 and E4 |
| Pri | Reward Engineer | Training | Reward components, exploit tests, the baseline, RND, the analysis of E3 and E5, the hack catalogue |
| Jayden | Agent Engineer | Agent | MCP server, tools, the LLM loop, LLM evaluation, the bridge experiment |

Everyone has exactly one assigned issue each week. Anyone who finishes early picks up from the floating list: stretch work, a PM pairing session, or a lane partner's blocked issue. Sahil and Akshaj float to wherever the Thursday updates point.

## Team size

| Subteam | People | Who |
|---|---|---|
| PM | 2 | Sahil (lead), Akshaj |
| Environment | 2 | Rishi, Prince |
| Training | 2 | Isaac, Pri |
| Agent | 1 | Jayden, backed by Akshaj |

## Before October 15

Sahil pushes `main`, makes the repository public, and runs `python3 tools/github/bootstrap.py` so all 48 issues and seven milestones exist when assignments go out on Thursday. Only Sahil and Akshaj have write access; members fork the repository and open pull requests from their forks.

## How a week works

- Sunday: sync. Everyone demos what they shipped. The deliverable is due before the sync, merged or in review.
- Monday: pick up next week's issue. Read it, ask questions the same day.
- Thursday: post two sentences in the channel: what you did and what blocks you. Anything not in review by Thursday gets a PM pairing session.
- From week 4, one PM watches five minutes of agent footage each week and writes what they saw in `docs/video-review.md`.

A deliverable counts when its acceptance criteria are checked and the evidence is in the pull request. Nothing counts because a date passed.

## Week 1, Oct 15 to 18: first commits

Assignments go out Thursday. Sahil and each member get one small task, about two hours, that ends in a reviewed pull request merged into `main`. Akshaj floats this week, reviewing the first pull requests. The point is a working setup and one trip through the review flow, not progress.

| Who | Issue | Deliverable |
|---|---|---|
| Sahil | OPS-01 | Both training machines pass `uv run poe check`; remote access works |
| Rishi | ENV-01 | Fetch script for the mGBA binding, importing on your machine |
| Prince | ENV-02 | Symbol files fetched and the first nine FireRed addresses recorded |
| Isaac | TRN-01 | CartPole trained from a committed script on three seeds |
| Pri | TRN-02 | The `explore` reward component with its tests |
| Jayden | AGT-01 | MCP server with a `get_state` tool on the fake backend |

Every starter has the same two last criteria: `uv run poe check` passes on your machine, and the pull request is merged.

## Week 2, due October 25: everything boots

By Sunday both emulators, the first memory reads, the training runner and the MCP server run on every member's machine.

| Who | Issue | Deliverable |
|---|---|---|
| Sahil | OPS-02 | Throughput table for both machines; compute budget recomputed |
| Akshaj | ENV-03 | PyBoy backend and a Super Mario Land reader |
| Rishi | ENV-04 | mGBA backend passing the conformance suite on macOS, Windows and Linux |
| Prince | ENV-05 | StateReader v0: position, party and badges from real memory |
| Isaac | TRN-03 | One command trains PPO and writes a reproducible run directory |
| Pri | TRN-04 | Reward registry with five components and isolation tests |
| Jayden | AGT-02 | `press` tool, `.mcp.json`, and a clean stdout test |

Watch for: the mGBA binding on Windows. If it is not working by Thursday, a PM pairs on it and the fallback (libretro.py) gets a one-hour spike.

## Week 3, due November 1: schema freeze

By Sunday an agent trains on the real environment, and the logging schema is frozen.

| Who | Issue | Deliverable |
|---|---|---|
| Sahil | OPS-03 | Schema module, Parquet writer, fake logger, sign-off from each lane, tag `schema-v1` |
| Akshaj | ENV-06 | Start states S0 and S1, five segments, the gym states, a hashed manifest |
| Rishi | ENV-07 | AlgoArcadeEnv passing Gymnasium's checker on all three backends |
| Prince | ENV-08 | The full FireRed GameState, one validation test per field |
| Isaac | TRN-05 | First PPO runs on Super Mario Land and two FireRed segments, curves posted |
| Pri | TRN-06 | Reward v1 and a scripted-exploit test for each known hack |
| Jayden | AGT-03 | MCP server on FireRed, the LLM loop with hard caps, a measured cost per action |

This is the heaviest build week. ENV-07 and OPS-03 both have to land before TRN-05 can produce a curve, so Rishi and Sahil aim for Wednesday.

Watch for: freezing a schema nobody has tested. If the StateReader is not validated on real memory by November 1, the freeze moves to November 8. It did: the freeze is November 8 (#20).

## Week 4, due November 8: reproduce and automate

By Sunday the baseline clears the early segments on five seeds, the queue runs the training box, and we know which trainer we are on.

| Who | Issue | Deliverable |
|---|---|---|
| Sahil | OPS-04 | The run queue: crash-tolerant, tagged, notifying, nightly |
| Akshaj | OPS-05 | Checkpoint to mp4 with overlays; the statistics module; the first video review entry |
| Rishi | ENV-09 | Archive resets, with evaluation always from the fixed state |
| Prince | ENV-10 | World coordinates and walkability grids from the game's own map data |
| Isaac | TRN-07 | Our PPO trainer, and the parity result against Stable-Baselines3 |
| Pri | TRN-08 | seg_01 to seg_03 cleared at 80% or better over five seeds; baseline config named |
| Jayden | AGT-04 | `navigate_to`, `advance_dialogue`, and the three tool levels |

The rule this week is no new ideas. TRN-08 reproduces what the Red literature already shows, on our harness. If the parity gate fails, we stay on Stable-Baselines3 and record that in `docs/decisions.md`.

## Week 5, due November 15: feature freeze

By Sunday every experiment and the Brock runs are in the queue, the LLM arm is evaluated, and the second game is trained. After Sunday no new features merge.

| Who | Issue | Deliverable |
|---|---|---|
| Sahil | OPS-06 | E1 to E5 and the Brock runs queued; freeze checklist passed |
| Akshaj | ENV-11 | Super Mario Land trained with the untouched config (E9); the port diary |
| Rishi | ENV-12 | Symbolic and pixel observations; E2 queued |
| Prince | ENV-13 | Battle state in GameState; the ordered milestone list from Pallet to Brock |
| Isaac | TRN-09 | PQN, and E1 queued |
| Pri | TRN-10 | RND with its unit tests; E3 and E5 queued |
| Jayden | AGT-05 | E7: ten attempts per model, tool level and segment, with cost per success |

This week holds all of the experiment setup, so the arms land in the order their code is ready: baseline and archive first, then PQN, RND and pixels by Wednesday.

## Week 6, due November 22: analysis and dashboard

Everyone is on analysis and visualization. Each panel is built on fake-logger data first, then pointed at real runs. Runs keep going in the background.

| Who | Issue | Deliverable |
|---|---|---|
| Sahil | OPS-07 | Final runs finished or cut off; results frozen under tag `results-v1` |
| Akshaj | VIZ-01 | Dashboard building by Tuesday on fake data; all panels assembled by Sunday |
| Rishi | VIZ-02 | Run registry, second-game panel, batch video rendering |
| Prince | VIZ-03 | Schematic Kanto map and the animated exploration heatmap |
| Isaac | VIZ-04 | E1, E2 and E4 analysed; comparison panels with bootstrap bands |
| Pri | VIZ-05 | E3 and E5 analysed; reward decomposition; the hack catalogue |
| Jayden | VIZ-06 | Agent video grid; LLM against RL cost table |
| Jayden | AGT-06 | Optional extra: LLM trajectories as an RL curriculum (E8), only after VIZ-06 merges |

If a panel is not ready by Sunday, the matching Weights & Biases chart goes in its place and the issue says so. The Brock runs report where they stood on November 22 and keep running.

## Week 7, due November 29: ship

Thanksgiving is Thursday the 26th. This week is writing and polish. No training, no features.

| Who | Issue | Deliverable |
|---|---|---|
| Sahil | DOC-01 | The technical writeup, assembled from the lane sections |
| Akshaj | DOC-02 | Dashboard published; slide deck and a two-minute demo video |
| Rishi | DOC-03 | A fresh clone reproduces one result; docs fixed where it stumbled |
| Prince | DOC-04 | Reverse engineering and harness section; the adding-a-game guide |
| Isaac and Pri | DOC-05 | Methods, results and the final hack catalogue |
| Jayden | DOC-06 | The LLM arm and what the comparison shows |

Monday November 30: tag v1.0 and hold a short retro.

## Dates that cannot move

| Date | What | Why |
|---|---|---|
| Nov 8 | Schema freeze (moved from Nov 1, #20) | Every run logged before it is thrown away |
| Nov 8 | Run queue live; trainer parity decided | The experiment matrix does not fit without the queue |
| Nov 15 | Feature freeze | Final runs need a week of machine time behind them |
| Nov 22 | Results freeze | Week 7 has Thanksgiving in it and is for writing |

## How PMs float

Each PM owns one decoupled issue a week. None of them sits on another lane's critical path for more than a week, except OPS-03 and OPS-04, which are Sahil's and have nothing else scheduled against them.

The rest of PM time goes where the Thursday updates point. The order of priority is fixed: an issue marked critical-path first, then anything blocking another lane, then stretch work.

Sahil also owns the two training machines. The Windows desktop takes E1, E3, E4 and the S1 Brock runs. The Mac takes E2, E5, E9 and the cold-start runs.

## If we fall behind

Cut in this order, and write each cut in `docs/decisions.md`:

1. E8, the LLM-to-RL bridge (AGT-06).
2. Cold-start Brock runs from S0. They continue after November 30.
3. E5, the penalties ablation.
4. E2, pixels against symbolic.
5. Shorten ablation runs from 15M steps before reducing seed counts. Five seeds stays.

Do not cut: the conformance suite, the schema freeze, the run queue, the baseline reproduction in week 4, or the second game.

If a member leaves, their lane partner takes the critical-path issues and a PM takes the rest. If Jayden leaves, Akshaj keeps E7 running and E8 is dropped.

## After November 30

Unfinished stretch work continues on the same queue: cold-start Brock runs, a Metroid: Zero Mission port on the mGBA backend, and an ICM arm for E3.
