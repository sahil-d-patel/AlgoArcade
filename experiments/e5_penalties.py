"""E5, do penalties help or hurt (TRN-10, #36). docs/technical-plan.md, Experiments.

The factor is the reward set, on BASELINE_V1 from S1. Three seeds per arm, so the result is
exploratory. Both arms are queued here, 6 runs, as OPS-06 counts them.
"""

import dataclasses

from algoarcade.rewards.registry import WEIGHTS
from algoarcade.training.config import BASELINE_V1

EXPERIMENT = "E5"
BASE = dataclasses.replace(BASELINE_V1, start_state="s1", track=True)
# The components on in each arm, at their starting weights.
POSITIVE = ("events", "explore", "level", "badge", "milestone")
OFF_BY_DEFAULT = ("heal", "blackout", "anti_loop", "anti_spam", "time")
ARMS = {
    "positive-only": {"reward_weights": {name: WEIGHTS[name] for name in POSITIVE}},
    "plus-penalties": {
        "reward_weights": {name: WEIGHTS[name] for name in POSITIVE + OFF_BY_DEFAULT}
    },
}
SEEDS = [0, 1, 2]
