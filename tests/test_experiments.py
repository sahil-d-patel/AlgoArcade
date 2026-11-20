"""The E3 and E5 experiment files: each arm differs from the baseline only in its factor."""

import dataclasses
from pathlib import Path

from algoarcade.rewards.registry import DEFAULT_WEIGHTS, WEIGHTS
from algoarcade.runqueue.queue import load_experiment
from algoarcade.training.config import BASELINE_V1

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
FROM_S1 = dataclasses.replace(BASELINE_V1, start_state="s1", track=True)


def test_e3_queues_none_and_rnd_on_the_baseline_seeds() -> None:
    configs = load_experiment(EXPERIMENTS / "e3_rnd.py")
    assert sorted((c.arm, c.seed) for c in configs) == [
        (arm, seed) for arm in ("none", "rnd") for seed in range(5)
    ]
    no_count = {name: w for name, w in DEFAULT_WEIGHTS.items() if name != "explore"}
    for c in configs:
        rnd = c.arm == "rnd"
        assert c == dataclasses.replace(
            FROM_S1,
            reward_weights=no_count,
            second_value_head=rnd,
            int_coef=0.5 if rnd else 0.0,
            int_gamma=0.99,
            experiment_id="E3",
            arm=c.arm,
            seed=c.seed,
        )


def test_e5_lists_the_components_on_in_each_arm_with_three_seeds() -> None:
    configs = load_experiment(EXPERIMENTS / "e5_penalties.py")
    assert sorted((c.arm, c.seed) for c in configs) == [
        (arm, seed) for arm in ("plus-penalties", "positive-only") for seed in range(3)
    ]
    for c in configs:
        weights = DEFAULT_WEIGHTS if c.arm == "positive-only" else WEIGHTS
        assert c == dataclasses.replace(
            FROM_S1, reward_weights=weights, experiment_id="E5", arm=c.arm, seed=c.seed
        )
