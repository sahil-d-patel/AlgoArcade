"""The shared baseline arm of E1 to E4 (OPS-06, #34): PPO, symbolic observations, the count
bonus, archive off, from S1. docs/technical-plan.md, Experiments.

Every E1 to E5 file uses this BASE and these seeds, so each compares against these five runs.
algoarcade.viz.comparison finds them by the experiment id "baseline"; E3's table counts them as
its count arm.
"""

import dataclasses

from algoarcade.training.config import BASELINE_V1

EXPERIMENT = "baseline"
BASE = dataclasses.replace(BASELINE_V1, start_state="s1", track=True)
ARMS = {"v1": {}}
SEEDS = [0, 1, 2, 3, 4]
