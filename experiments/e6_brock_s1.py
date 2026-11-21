"""E6 from S1, how far does the best config get (OPS-06, #34). docs/technical-plan.md, Experiments.

The baseline with the archive on, because no E1 to E4 result was in when this was queued; the
writeup says so. Success is goal_reached in the evaluation episodes: the badge flag
FLAG_BADGE01_GET (0x820), with the archive off, as every evaluation runs. 100M steps is the
technical plan's budget for a Brock run from S1, inside its 50M to 150M guess. eval_every gives
ten evaluations a run, of BASELINE_V1's 5 episodes each, so the writeup can say when a run first
got the badge. Submit this file before e6_brock_s0.py, so the S1 runs train first.
"""

import dataclasses

from algoarcade.training.config import BASELINE_V1

EXPERIMENT = "E6"
BASE = dataclasses.replace(
    BASELINE_V1,
    start_state="s1",
    track=True,
    archive=True,
    total_timesteps=100_000_000,
    eval_every=10_000_000,
)
ARMS = {"s1": {}}
SEEDS = [0, 1, 2]
