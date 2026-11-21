"""TRN-08 on the training box: BASELINE_V1 on the three early segments, five seeds each (#29).

Each arm is a segment, not a factor: the baseline is the same in all three. The runs keep the
evaluation BASELINE_V1 had when they were queued, 20 episodes every 1.5M steps; BASELINE_V1 now
evaluates less (docs/decisions.md, 2026-11-20).
"""

import dataclasses

from algoarcade.training.config import BASELINE_V1

EXPERIMENT = "TRN-08"
BASE = dataclasses.replace(BASELINE_V1, track=True, eval_every=1_500_000, eval_episodes=20)
ARMS = {
    segment: {"start_state": segment}
    for segment in ("seg_01_house", "seg_02_pallet", "seg_03_rival")
}
SEEDS = [0, 1, 2, 3, 4]
