"""E8, do LLM trajectories help RL as a curriculum (AGT-06, #37). docs/technical-plan.md,
Experiments.

The factor is where training episodes on seg_05_forest start: the fixed state (none), or the
states a successful LLM seg_05 attempt saved, latest first, moving one state earlier at 80% goal
rate over 100 episodes (curriculum). Both arms evaluate from the fixed seg_05 state. Three seeds
per arm, so the result is exploratory.

CURRICULUM is that attempt's run name, its folder under data/states/curriculum/. No LLM attempt
has reached the seg_05 goal yet, so make one before submitting, with the loop's capture on:

    uv run python -m algoarcade.agent.loop --name e8-seg05 --segment seg_05_forest --level 2 \
        --max-actions 2000 --max-dollars <cap> --capture-every 10

The base is the FIRERED preset at 15M steps, a guess matching the E1 to E4 runs. An episode is
8,192 steps whatever its start, so moving back one state takes at least 7 rounds of 16 episodes,
917,504 steps: a 15M-step run moves back at most 16 times.
"""

import dataclasses

from algoarcade.training.config import FIRERED

EXPERIMENT = "E8"
CURRICULUM = "e8-seg05"
BASE = dataclasses.replace(
    FIRERED, start_state="seg_05_forest", total_timesteps=15_000_000, track=True
)
ARMS = {"none": {}, "curriculum": {"curriculum": CURRICULUM}}
SEEDS = [0, 1, 2]
