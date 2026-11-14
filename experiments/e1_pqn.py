"""E1's PQN arm (TRN-09, #35): policy gradient against value-based learning, same sampling.

The PPO arm is the shared E1 to E4 baseline that OPS-06 queues: BASELINE_V1 from S1, seeds 0 to 4.
This arm changes one thing, the learner. It keeps the baseline's 16 envs, step budget,
observations, reward and gamma, and takes CleanRL's PQN values for the rest. No retuning after
the curves are in: a PQN that loses is the result.

The metric, fixed before any run: the mean milestone_count over the evaluation episodes of each
run's last checkpoint, one score per seed. scripts/e1.py computes it and the comparison.
"""

import dataclasses

from algoarcade.training.config import BASELINE_V1, PQN_CLEANRL

EXPERIMENT = "E1"
BASE = dataclasses.replace(BASELINE_V1, start_state="s1", track=True)
ARMS = {"pqn": {"pqn": dataclasses.replace(PQN_CLEANRL, gamma=BASE.ppo.gamma)}}
SEEDS = [0, 1, 2, 3, 4]
METRIC = "milestone_count"
