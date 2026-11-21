"""E9: does the framework transfer? Super Mario Land 1-1 on PyBoy with FireRed's PPO config (#31).

Only the game changes: the PPO values, environments and normalization are FireRed's, and the run
is the FireRed baseline's 15M steps. The arms differ only in how long each decision holds its
button: hold16 is the Super Mario Land default, hold8 FireRed's, under which every
TRN-05 seed stopped at the pipe at x_tile 103. Eight seeds each, the ticket's minimum.
"""

import dataclasses

from algoarcade.training.config import BASELINE, SUPER_MARIO_LAND

EXPERIMENT = "E9"
BASE = dataclasses.replace(SUPER_MARIO_LAND, total_timesteps=BASELINE.total_timesteps, track=True)
ARMS = {"hold16": {"hold": 16}, "hold8": {"hold": 8}}
SEEDS = list(range(8))
