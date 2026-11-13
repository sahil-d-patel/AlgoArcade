"""What `run` starts for each attempt: python -m algoarcade.runqueue.train <config.json>."""

import sys
from pathlib import Path

from algoarcade.training.config import load_config
from algoarcade.training.runner import train

# SubprocVecEnv with spawn re-imports this module in every worker.
if __name__ == "__main__":
    print(train(load_config(Path(sys.argv[1]))))
