import dataclasses
import json
from pathlib import Path

import pytest
import tyro

from algoarcade.rewards.registry import DEFAULT_WEIGHTS
from algoarcade.training.config import BASELINE_V1, PRESETS, load_config, save_config


@pytest.mark.parametrize("name", ["cartpole", "acrobot", "firered"])
def test_saved_config_reloads_equal(name: str, tmp_path: Path) -> None:
    config = dataclasses.replace(
        PRESETS[name][1], seed=3, runs_dir=tmp_path / "runs", start_state="seg_01_house"
    )
    path = tmp_path / "config.json"
    save_config(config, path)
    assert load_config(path) == config


@pytest.mark.parametrize("name", ["firered", "super_mario_land"])
def test_game_presets_have_gamma_0997_and_no_reward_normalisation(name: str) -> None:
    config = PRESETS[name][1]
    assert config.ppo.gamma == 0.997
    assert config.norm_reward is False
    assert config.norm_obs is True


def test_firered_preset_pays_the_default_reward() -> None:
    assert PRESETS["firered"][1].reward_weights == DEFAULT_WEIGHTS


def test_super_mario_land_trains_with_the_firered_ppo_values() -> None:
    assert PRESETS["super_mario_land"][1].ppo == PRESETS["firered"][1].ppo


def test_baseline_v1_pins_every_field() -> None:
    # Every experiment changes one field of the baseline, so changing it must change this test.
    assert PRESETS["baseline_v1"][1] is BASELINE_V1
    assert dataclasses.asdict(BASELINE_V1) == {
        "env_id": "firered",
        "total_timesteps": 15_000_000,
        "n_envs": 16,
        "checkpoint_every": 500_000,
        "eval_every": 1_500_000,
        "ppo": {
            "learning_rate": 0.0003,
            "n_steps": 2048,
            "batch_size": 4096,
            "n_epochs": 3,
            "gamma": 0.997,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
        },
        "norm_obs": True,
        "norm_reward": False,
        "eval_deterministic": False,
        "seed": 0,
        "eval_episodes": 20,
        "runs_dir": Path("data/runs"),
        "track": False,
        "start_state": tyro.MISSING,
        "reward_weights": {
            "events": 1.0,
            "explore": 0.02,
            "level": 0.5,
            "badge": 10.0,
            "milestone": 2.0,
        },
        "step_log_every": 4,
        "archive": False,
        "archive_p": 0.5,
        "experiment_id": "",
        "arm": "",
        "second_value_head": False,
    }


@pytest.mark.parametrize("name", [n for n in PRESETS if n != "baseline_v1"])
def test_other_presets_evaluate_deterministically_at_every_checkpoint(name: str) -> None:
    config = PRESETS[name][1]
    assert config.eval_deterministic is True
    assert config.eval_every == config.checkpoint_every


@pytest.mark.parametrize("eval_every", [0, 750_000, -500_000])
def test_eval_every_must_be_a_positive_multiple_of_checkpoint_every(eval_every: int) -> None:
    with pytest.raises(ValueError, match="eval_every"):
        dataclasses.replace(BASELINE_V1, eval_every=eval_every)


def test_a_config_saved_before_eval_every_loads_as_it_ran(tmp_path: Path) -> None:
    # Runs before 2026-11-14 evaluated every checkpoint with the argmax, and wrote neither field.
    config = dataclasses.replace(
        PRESETS["firered"][1],
        start_state="seg_01_house",
        checkpoint_every=250_000,
        eval_every=250_000,
    )
    path = tmp_path / "config.json"
    save_config(config, path)
    raw = json.loads(path.read_text())
    del raw["eval_every"], raw["eval_deterministic"]
    path.write_text(json.dumps(raw))
    assert load_config(path) == config
