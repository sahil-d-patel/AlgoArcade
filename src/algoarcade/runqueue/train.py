"""What `run` starts for each attempt: python -m algoarcade.runqueue.train <config.json>.

A config with the second value head, which RND's int_coef needs, trains on the owned trainer;
every other config on the SB3 runner (docs/decisions.md, 2026-11-14).
"""

import sys
from pathlib import Path

from algoarcade.runqueue.queue import STOP
from algoarcade.training import ppo, runner
from algoarcade.training.config import load_config

# SubprocVecEnv and AsyncVectorEnv with spawn re-import this module in every worker.
if __name__ == "__main__":
    path = Path(sys.argv[1])
    config = load_config(path)
    trainer = ppo.train if config.second_value_head else runner.train
    # The config is <queue>/configs/<run>.json.
    print(trainer(config, stop=path.parent.parent / STOP))
