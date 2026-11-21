"""E3, does learned novelty beat counts (TRN-10, #36). docs/technical-plan.md, Experiments.

The factor is the exploration bonus, on BASELINE_V1 from S1 with the baseline's five seeds. The
count arm is the shared baseline arm E1 to E4 compare against. OPS-06 queues it, and it compares
only if it runs this BASE on these seeds, so this file queues the other two:

- none: the baseline's reward without explore.
- rnd: RND in place of explore. It trains on the owned trainer and the other two arms on SB3;
  the TRN-07 parity gate found no detectable difference between them (docs/decisions.md,
  2026-11-14).

int_coef 0.5 is the RND paper's 1:2 intrinsic-to-extrinsic ratio. It and RND's input
(algoarcade.training.ppo.rnd_fields) are guesses, written here before any run.

S1 has no step budget yet, so these runs stop at startup. Submit them once OPS-06 sets one.
"""

import dataclasses

from algoarcade.training.config import BASELINE_V1

EXPERIMENT = "E3"
BASE = dataclasses.replace(BASELINE_V1, start_state="s1", track=True)
NO_COUNT = {name: weight for name, weight in BASE.reward_weights.items() if name != "explore"}
ARMS = {
    "none": {"reward_weights": NO_COUNT},
    "rnd": {
        "reward_weights": NO_COUNT,
        "second_value_head": True,
        "int_coef": 0.5,
        "int_gamma": 0.99,
    },
}
SEEDS = [0, 1, 2, 3, 4]
