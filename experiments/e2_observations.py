"""E2, does a symbolic observation match pixels (ENV-12, #32). docs/technical-plan.md, Experiments.

The factor is the observation mode, on BASELINE_V1 from S1 with the baseline's five seeds. The
symbolic arm is the shared baseline arm E1 to E4 compare against. OPS-06 queues it, and it
compares only if it runs this BASE on these seeds, so this file queues only pixels.

Pixels need a CNN, which E2 cannot remove. The encoder follows from obs_mode: SB3's
MultiInputPolicy gives the uint8 frames to NatureCNN (256 features) and flattens the rest. On
FireRed's 7 features and 6 actions the symbolic policy has 44,359 parameters and the pixels
policy 610,279, of which the CNN is 567,712. Both arms train on the SB3 runner.
"""

import dataclasses

from algoarcade.training.config import BASELINE_V1

EXPERIMENT = "E2"
BASE = dataclasses.replace(BASELINE_V1, start_state="s1", track=True)
ARMS = {"pixels": {"obs_mode": "pixels"}}
SEEDS = [0, 1, 2, 3, 4]
