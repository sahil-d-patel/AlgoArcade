"""E9: does the framework transfer? Super Mario Land 1-1 on PyBoy with FireRed's PPO config (#31).

Only the game changes: the PPO values, environments and normalization are FireRed's, and the run
is the FireRed baseline's 15M steps. Eight seeds, the ticket's minimum.
"""

import dataclasses

from algoarcade.training.config import BASELINE, SUPER_MARIO_LAND

EXPERIMENT = "E9"
BASE = dataclasses.replace(SUPER_MARIO_LAND, total_timesteps=BASELINE.total_timesteps, track=True)
ARMS: dict[str, dict[str, object]] = {"untouched": {}}
SEEDS = list(range(8))
