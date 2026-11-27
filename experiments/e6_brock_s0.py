"""E6 from S0, the cold start (OPS-06, #34). docs/technical-plan.md, Experiments.

The same config as experiments/e6_brock_s1.py from the player's bedroom, two seeds. 100M steps
is the low end of the technical plan's budget for S0; its guess is 100M to 300M. Queue it after
e6_brock_s1.py.
"""

import dataclasses

from algoarcade.training.config import BASELINE_V1

EXPERIMENT = "E6"
BASE = dataclasses.replace(
    BASELINE_V1,
    start_state="s0",
    track=True,
    archive=True,
    total_timesteps=100_000_000,
    eval_every=10_000_000,
)
ARMS = {"s0": {}}
SEEDS = [0, 1]
