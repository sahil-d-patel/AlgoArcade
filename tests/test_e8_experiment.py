"""The E8 experiment file: curriculum against none on seg_05, three seeds each."""

import dataclasses
from pathlib import Path

from algoarcade.runqueue.queue import load_experiment
from algoarcade.training.config import BASELINE_V1

E8 = Path(__file__).resolve().parents[1] / "experiments" / "e8_curriculum.py"


def test_e8_queues_three_seeds_per_arm_differing_only_in_the_curriculum() -> None:
    configs = load_experiment(E8)
    assert sorted((c.arm, c.seed) for c in configs) == [
        (arm, seed) for arm in ("curriculum", "none") for seed in range(3)
    ]
    base = dataclasses.replace(BASELINE_V1, start_state="seg_05_forest", track=True)
    for c in configs:
        curriculum = "e8-seg05" if c.arm == "curriculum" else ""
        assert c == dataclasses.replace(
            base, curriculum=curriculum, experiment_id="E8", arm=c.arm, seed=c.seed
        )
        assert c.ppo.gamma == 0.997
        assert not c.archive
        assert (c.total_timesteps, c.eval_every, c.eval_episodes) == (15_000_000, 7_500_000, 5)
        assert not c.eval_deterministic
