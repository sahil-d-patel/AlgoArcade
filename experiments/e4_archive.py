"""E4, does returning to the frontier help (OPS-06, #34). docs/technical-plan.md, Experiments.

The factor is archive resets (ENV-09) for training episodes, at archive_p 0.5, a guess. The
archive-off arm is the shared baseline arm in experiments/baseline.py. Evaluation never uses the
archive.
"""

import dataclasses

from algoarcade.training.config import BASELINE_V1

EXPERIMENT = "E4"
BASE = dataclasses.replace(BASELINE_V1, start_state="s1", track=True)
ARMS = {"on": {"archive": True}}
SEEDS = [0, 1, 2, 3, 4]
