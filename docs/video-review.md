# Video review

Each week from week 4, one PM watches at least five minutes of agent footage and writes an entry here (docs/development-plan.md). An entry gives the run ids and checkpoints, the minutes watched, what we saw, and marks each exploit signature seen or not seen.

Render a clip with:

```
uv run python -m algoarcade.viz.render --checkpoint <run dir>/checkpoints/step_<n>.zip --seed 0 --out data/videos/<name>.mp4
```

The policy plays deterministically from the run's start state, one video frame per decision at 15 frames per second, so a clip runs six times faster than the game. Clips stay in `data/videos/`.

The six signatures:

| Signature | What it looks like |
|---|---|
| oscillation | Stepping back and forth between two tiles. |
| menu spam | Opening menus or pressing through text boxes with no progress. |
| faint loop | Letting the party faint again and again, then healing. |
| reset farming | Reward that comes back after a reset or a continue. |
| grass camping | Staying in tall grass to fight wild battles for level reward. |
| warp flicker | Stepping on and off a door or stairs so the map changes back and forth. |

## Week of Nov 8 (written Nov 13)

**Minutes watched by a person: none yet.** A PM still has to watch at least five minutes of the clips below, put their name and minutes here, and correct the table. Until then this entry rests on two checks the agent that wrote it could run:

- A per-step trace of each episode: every decision's action, location, reward components and, for FireRed, battle, text box and party HP. The trace replays the same deterministic episode as the clip; return and final location matched for all nine.
- 35 frames sampled from six of the clips and looked at.

Clips, 27 min 18 s in total, all seed 0 from each run's final checkpoint:

| Run id | Checkpoint | Clip length | Return | Final location |
|---|---|---|---|---|
| firered-seg_01_house-seed0-20261104-124733 | step_1015808 | 2:17 | 0.04 | (4, 1, 6, 8) |
| firered-seg_01_house-seed1-20261104-131018 | step_1015808 | 2:17 | 0.04 | (4, 1, 6, 8) |
| firered-seg_01_house-seed2-20261104-133157 | step_1015808 | 2:17 | 0.06 | (4, 1, 3, 6) |
| firered-seg_02_pallet-seed0-20261104-161742 | step_1015808 | 4:33 | 0.20 | (3, 0, 6, 14) |
| firered-seg_02_pallet-seed1-20261104-164226 | step_1015808 | 4:33 | 0.10 | (3, 0, 21, 14) |
| firered-seg_02_pallet-seed2-20261104-170645 | step_1015808 | 4:33 | 0.20 | (4, 3, 6, 4) |
| super_mario_land-level_1_1-seed0-20261104-210903 | step_2031616 | 2:17 | 1.46 | (1, 1, 28, 16) |
| super_mario_land-level_1_1-seed1-20261104-211407 | step_2031616 | 2:17 | 2.98 | (1, 1, 52, 16) |
| super_mario_land-level_1_1-seed2-20261104-211912 | step_2031616 | 2:17 | 1.52 | (1, 1, 35, 16) |

The FireRed returns equal the last row of each run's `eval.csv`.

What we saw: every final policy settles on one action and stands still for most of the episode. None of the six signatures explains the flat returns; the policies stop moving.

- FireRed seg_01_house: each policy presses one direction on all 2,048 steps, Down for seeds 0 and 1 and Left for seed 2. The player walks 2 or 3 tiles to the edge of the bedroom and stands there. The last reward comes on step 3 or 4.
- FireRed seg_02_pallet: the same, with Left, Right and Up on all 4,096 steps. Seed 0 walks into the sign by the flowers in Pallet Town and seed 1 walks 5 tiles right. Seed 2 walks into Oak's Lab on step 4, up 8 tiles to an NPC, and holds Up for the remaining 4,083 steps. The last reward comes on step 6 to 13. No battle, no text box and no faint in any of the six episodes.
- Super Mario Land: all three reach x_tile 71 to 92 within the first 270 steps and never pass it. Seeds 0 and 2 lose all three lives to the timer, at steps 676, 1,353 and about 2,040; a life lasts about 675 decisions. Seeds 0 and 2 jump in place at x_tile 71 for the rest of the first life, about 600 steps, earning nothing. A tap of A lands within the 24-frame decision, so the location's y reads 16 throughout. After the first life seed 0 walks back and forth between x_tile 28 and 32, seed 1 presses Right into a pipe at x_tile 52 for about 1,000 steps, and seed 2 jumps in place at x_tile 48.

| Signature | FireRed | Super Mario Land | Evidence |
|---|---|---|---|
| oscillation | not seen | seen, pays nothing | FireRed: no step returns to the tile of two steps before. SML seed 0: x alternates on 431 steps, all after step 677, and steps 677 to 2,048 pay 0.06 in total. |
| menu spam | not seen | not seen | No action opens a menu; Start is not in the action set. FireRed: 0 steps with a text box open. |
| faint loop | not seen | not seen | FireRed: 0 battle steps, party HP never below full in seg_02. SML has no party. |
| reset farming | not seen | not seen | FireRed: an evaluation episode has no reset after its start. SML: one continue per episode; it paid 0 over the next 20 steps for seeds 0 and 2, and 0.02 for one new tile for seed 1. |
| grass camping | not seen | not seen | FireRed: 0 battle steps. SML has no grass. |
| warp flicker | not seen | not seen | FireRed: one map change in six episodes (seg_02 seed 2 into Oak's Lab), never back. SML has one map per episode. |

Why the policies collapse onto one action is a guess, not checked: with a reward that stops after a handful of steps, the deterministic mode of the policy is one action whatever the observation.
