"""E2's experiment file: five pixel runs that differ from the baseline arm only in obs_mode."""

import dataclasses
from pathlib import Path

from algoarcade.runqueue.queue import load_experiment
from algoarcade.training.config import BASELINE_V1

E2 = Path(__file__).resolve().parents[1] / "experiments" / "e2_observations.py"


def test_e2_queues_five_pixel_runs_that_differ_from_the_baseline_only_in_obs_mode() -> None:
    configs = load_experiment(E2)
    assert [(c.arm, c.seed) for c in configs] == [("pixels", seed) for seed in range(5)]
    baseline = dataclasses.replace(BASELINE_V1, start_state="s1", track=True)
    for c in configs:
        assert c == dataclasses.replace(
            baseline, obs_mode="pixels", experiment_id="E2", arm="pixels", seed=c.seed
        )
