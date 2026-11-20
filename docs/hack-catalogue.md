# Hack catalogue

Every way we know for an agent to earn reward without making progress: what it does, the term that pays, the fix and its test, and whether a run has shown it. As of Nov 20 no E3 or E5 run exists and no clip is under `data/videos/hacks/`, so every clip cell says "no clip yet". Sources: `tests/test_exploits.py`, `docs/harness-spec.md` (Reward registry), `docs/video-review.md` and `docs/decisions.md`.

An entry is **observed** when a run shows it, with the run id and a clip, or **not observed**, with the test that bounds it. To add a clip, render the checkpoint into `data/videos/hacks/<exploit>.mp4`, named by the first column, and put the path in the row; the dashboard's Hacks page plays every clip in that folder:

```
uv run python -m algoarcade.viz.render --checkpoint <run dir>/checkpoints/step_<n>.zip --seed 0 --out data/videos/hacks/<exploit>.mp4
```

## The scripted exploits

Each has a policy in `tests/test_exploits.py` that tries it and must stay under the bound in its assert. `uv run pytest -k exploit` runs exactly these seven.

| Exploit | What the agent does | Pays | Fix | Status | Run | Clip |
|---|---|---|---|---|---|---|
| oscillation | Steps back and forth between two tiles | explore | The visited set is permanent for the episode and keyed on the full location, so the pair pays two first visits. `test_exploit_two_tile_oscillation` | Observed, and the fix held: on Super Mario Land seed 0 walked between x_tile 31 and 32 on 431 steps, and steps 677 to 2,048 paid 0.06 in all (video review, week of Nov 8) | `super_mario_land-level_1_1-seed0-20261104-210903`, `step_2031616` | no clip yet |
| menu spam | Opens a text box and mashes buttons that do nothing | Nothing by default: no term pays a screen change. `anti_spam` charges it when on | Start is not in the action set, and novelty keys on coordinates, never the screen. `test_exploit_menu_spam` (bound 0) | Not observed: no text box opened in the six FireRed review episodes | | no clip yet |
| faint loop | Loses a battle so the whiteout restores full HP | heal | heal never pays a rise from a fainted party, a party size change or one bad read, and is off by default. `test_exploit_faint_loop` (heal pays 0) | Not observed | | no clip yet |
| reset farming | Collects a reward paid at the start of an episode, or after a continue | Any term that read the start state as a gain | Every term returns 0 on the reset step, and a progress best is set only by two agreeing reads. `test_exploit_instant_reset` (100 starts pay 0), `test_zero_on_reset_and_on_a_no_op_step`. A Super Mario Land continue pays nothing: `test_the_preset_plays_through_a_game_over_without_paying_for_it` | Not observed: the one judgeable continue in the Nov 8 review was followed by 20 steps that paid 0.02, one new tile | | no clip yet |
| grass camping | Stays in the first tall grass fighting wild battles for levels | level | level pays at most 1.0 a step and nothing past a party level sum of 20 (`LEVEL_CAP`). `test_exploit_grass_camping` (level bound 7.5) | Not observed | | no clip yet |
| warp flicker | Steps on and off a warp; while the save blocks move, a count can read one high for a frame | events, level, badge, milestone | A progress term pays a rise only once two consecutive reads agree (#73, decisions 2026-11-04). `test_exploit_warp_flicker` (progress terms pay 0) | Not observed in a run. Review found it on a read sequence: badge reads 0, 0, 0, 1, 0, 0, 1 paid the spike and not the real badge (#22) | | no clip yet |
| archive farming | Resets into an archived cell again and again to collect its progress or its path | progress terms, explore | The cell carries each term's best and its visited set. `test_exploit_archive_farming` (bound 0) | Not observed: no archive-on run yet (E4) | | no clip yet |

## Other exploits with a test

| Exploit | What the agent does | Pays | Fix | Status | Run | Clip |
|---|---|---|---|---|---|---|
| jumping (Super Mario Land) | Jumps in place, or while running, to reach new y tiles | explore | Bounded, not prevented: the location is in 8-pixel tiles (#58), and each y tile in a column pays once, at most 32 a column. `test_jumping_in_place_pays_explore_once_per_y_tile` (a 48-pixel arc pays 6) | Not observed as pay: in the Nov 8 review seeds 0 and 2 pressed A in place for hundreds of steps, and seed 0's location read y 16 on all of them | | no clip yet |
| title demo (Super Mario Land) | Loses three lives so the title menu plays its demo, whose tiles are new | explore | The backend reloads the start state as soon as the menu flag (0xFF9F) is set (decisions 2026-11-04). `test_continues_loads_the_start_state_only_when_the_menu_is_up` | Not observed | | no clip yet |

## Predicted, not fixed

No test bounds these. E5's plus-penalties arm turns heal and the four penalties on, so its runs are where they would show; RND is E3's.

| Exploit | What the agent does | Pays or saves | Why it stands |
|---|---|---|---|
| grass camping for heal | Camps in grass with heal on: a level-up raises a damaged Pokémon's HP fraction | heal | The level cap does not limit heal |
| battle dodging | Flees or avoids every battle so the party never whites out, as in RL-emerald | blackout (-5.0) | Nothing stops it; it is why blackout is off |
| standing still | Stops moving, or walks loops longer than 30 steps | anti_loop (-0.01) | Standing still is free |
| NPC avoidance | Never talks to an NPC or reads a sign | anti_spam (-0.02) | Talking risks the charge |
| level grinding for novelty | Grinds levels: each level is a new `party_level_sum`, with no cap | RND (#86, not merged) | Decays once the predictor has seen the value; RND is outside the registry and schema-v1 |
| faint warping for novelty | Faints to be warped to a Pokémon Center, a new tile early in training; blackout is off in E3 | RND (#86, not merged) | Same |
