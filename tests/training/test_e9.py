"""E9 (#31): Super Mario Land trains with FireRed's PPO config."""

import dataclasses
from pathlib import Path

from algoarcade.runqueue.queue import load_experiment
from algoarcade.training.config import FIRERED

E9 = Path(__file__).resolve().parents[2] / "experiments" / "e9_super_mario_land.py"


def test_e9_ppo_config_equals_the_firered_baseline_field_by_field() -> None:
    configs = load_experiment(E9)
    assert {c.arm: c.hold for c in configs} == {"hold16": 16, "hold8": 8}
    for arm in ("hold16", "hold8"):
        assert len({c.seed for c in configs if c.arm == arm}) >= 8
    for config in configs:
        assert config.env_id == "super_mario_land"
        for field in dataclasses.fields(FIRERED.ppo):
            assert getattr(config.ppo, field.name) == getattr(FIRERED.ppo, field.name), field.name
        assert (config.n_envs, config.norm_obs, config.norm_reward) == (
            FIRERED.n_envs,
            FIRERED.norm_obs,
            FIRERED.norm_reward,
        )
