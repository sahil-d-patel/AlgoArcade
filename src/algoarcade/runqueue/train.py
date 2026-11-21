"""What `run` starts for each attempt: python -m algoarcade.runqueue.train <config.json>.

A config with a pqn block trains PQN; one with the second value head, which RND's int_coef needs,
trains on the owned PPO trainer; every other config on the SB3 runner (docs/decisions.md,
2026-11-14).
"""

import sys
from pathlib import Path

from algoarcade.training import ppo, pqn, runner
from algoarcade.training.config import load_config

# SubprocVecEnv and AsyncVectorEnv with spawn re-import this module in every worker.
if __name__ == "__main__":
    config = load_config(Path(sys.argv[1]))
    if config.pqn is not None:
        print(pqn.train(config))
    else:
        print((ppo.train if config.second_value_head else runner.train)(config))
