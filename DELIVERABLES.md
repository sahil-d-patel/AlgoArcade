# Deliverables & Timeline

> **How to read this file.** This is the PMs' current plan for what AlgoArcade ships and when. It is a living plan, not a contract. The authoritative picture lives in **GitHub Issues and the Project board**. The week-by-week assignment is in [`docs/development-plan.md`](docs/development-plan.md).

Kickoff was October 11, 2026, and assignments start Thursday October 15. A deliverable is due every Sunday through November 29, and v1.0 is tagged on November 30.

## Milestones

| # | Deliverable | Description | Owner (role) | Target | Issues |
|---|-------------|-------------|--------------|--------|--------|
| 1 | First commits | One two-hour starter task each for Sahil and the five members, each merged through a reviewed pull request. Proves every setup and the review flow before real work starts. | Everyone | Oct 18 | OPS-01, ENV-01, ENV-02, TRN-01, TRN-02, AGT-01 |
| 2 | Three backends, one interface | mGBA (FireRed) and PyBoy (Super Mario Land) pass the same conformance suite as the fake backend. | Rishi, Akshaj | Oct 25 | ENV-04, ENV-03 |
| 3 | Throughput benchmark | Measured steps per second on both machines. Every compute budget is recomputed from it. | Sahil | Oct 25 | OPS-02 |
| 4 | FireRed StateReader | Position, party and badges first, then the full GameState, then battle state and milestones. Staged so no week waits on the hardest fields. | Prince | Oct 25, Nov 1, Nov 15 | ENV-05, ENV-08, ENV-13 |
| 5 | Training runner | One command trains PPO and writes a reproducible run directory. | Isaac | Oct 25 | TRN-03 |
| 6 | Reward registry and exploit tests | Named, separately logged components. Each known hack is a scripted test. | Pri | Oct 25, Nov 1 | TRN-04, TRN-06 |
| 7 | MCP server and LLM loop | The same harness exposed to a language model, with hard cost caps. | Jayden | Oct 25, Nov 1 | AGT-02, AGT-03 |
| 8 | Gymnasium environment and start states | AlgoArcadeEnv on all three backends. Start states S0 and S1, five segments. | Rishi, Akshaj | Nov 1 | ENV-07, ENV-06 |
| 9 | Logging pipeline and schema freeze | Parquet writer, fake logger, schema frozen. Runs logged before this are thrown away. Moved from Nov 1 because ENV-08 was not validated by then (#20). | Sahil | Nov 8 | OPS-03 |
| 10 | First learning curves | PPO on Super Mario Land and two FireRed segments. | Isaac | Nov 1 | TRN-05 |
| 11 | Run queue | The training box runs experiments unattended. The matrix does not fit without it. | Sahil | Nov 8 | OPS-04 |
| 12 | Baseline reproduced, trainer decided | Early segments cleared over five seeds. Our PPO trainer passes or fails its parity gate. | Isaac, Pri | Nov 8 | TRN-07, TRN-08 |
| 13 | Map data and archive resets | World coordinates and walkability from the game's own data. Return-to-frontier resets. | Rishi, Prince | Nov 8 | ENV-09, ENV-10 |
| 14 | Evaluation tooling | Checkpoint to video, and the statistics module. | Akshaj | Nov 8, Nov 15 | OPS-05 |
| 15 | LLM tool ladder (E7) | Three tool levels, ten attempts per configuration, cost per success. | Jayden | Nov 8, Nov 15 | AGT-04, AGT-05 |
| 16 | Experiments queued (E1 to E5) and feature freeze | PPO against PQN, symbolic against pixels, count against RND, archive resets, penalties. Five seeds each for E1 to E4. No new features after this. | Isaac, Pri, Rishi, Sahil | Nov 15 | TRN-09, TRN-10, ENV-12, OPS-06 |
| 17 | Second game (E9) | Super Mario Land trained with the untouched PPO config, and a diary of what the port touched. | Akshaj | Nov 15 | ENV-11 |
| 18 | Brock runs (E6), stretch | The baseline config with archive resets, from the post-parcel state and then from a cold start. Reported where they stand on Nov 22; they keep running. | Sahil | Nov 15 to Nov 29 | OPS-06, OPS-07 |
| 19 | Analysis and dashboard | E1 to E5 analysed with intervals; Kanto heatmap, comparison panels, reward decomposition, agent video grid, run registry, in one static site. Built in week 6 by everyone (PM decision: no visualization work before then). | Everyone | Nov 22 | VIZ-01 to VIZ-06 |
| 20 | LLM-to-RL curriculum (E8), stretch | LLM trajectories as start states for PPO. First to slip. | Jayden | Nov 22 | AGT-06 |
| 21 | Results freeze | Final runs finished or cut off, numbers recomputed from Parquet. | Sahil | Nov 22 | OPS-07 |
| 22 | Writeup and showcase | Methods, results, the hack catalogue, deck and demo video. | Everyone | Nov 29 | DOC-01, DOC-02, DOC-04, DOC-05, DOC-06 |
| 23 | Reproducibility check, v1.0 | A fresh clone reproduces one result. Tag and retro. | Rishi, Sahil, Akshaj | Nov 30 | DOC-03 |

## Timeline

```
Week ending:   Oct18  Oct25  Nov1   Nov8   Nov15  Nov22  Nov29
               |------|------|------|------|------|------|
Starters       ██████
Backends              ██████
StateReader           ██████ ██████        ██████
Env, rewards          ██████ ██████
Schema freeze                ▲
Run queue                           ██████
Baseline                            ██████
Experiments                                ██████ ██████
LLM arm        ██████ ██████ ██████ ██████ ██████
Second game           ██████               ██████
Feature freeze                             ▲
Brock runs                                 ██████ ██████ ██████
Dashboard                                         ██████
Results freeze                                    ▲
Writeup                                                  ██████
```

## Working agreements

- **Each deliverable maps to one or more GitHub Issues.** The board is the source of truth; this file is the summary.
- **Four dates do not move:** schema freeze (Nov 8), run queue (Nov 8), feature freeze (Nov 15), results freeze (Nov 22). The schema freeze moved once, from Nov 1, by its own rule (#20). Everything else is an estimate.
- **"Done" is defined per issue** via acceptance criteria, not by a date passing.
- **Reprioritize openly.** If a deliverable is cut or moved, a PM writes why in the issue and in [`docs/decisions.md`](docs/decisions.md).
