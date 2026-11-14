"""What `run` starts for each attempt: python -m algoarcade.runqueue.train <config.json>. A config
with a pqn block trains PQN; any other trains SB3 PPO through the runner."""

import sys
from pathlib import Path

from algoarcade.training import pqn, runner
from algoarcade.training.config import load_config

# SubprocVecEnv and AsyncVectorEnv with spawn re-import this module in every worker.
if __name__ == "__main__":
    config = load_config(Path(sys.argv[1]))
    print((runner.train if config.pqn is None else pqn.train)(config))
