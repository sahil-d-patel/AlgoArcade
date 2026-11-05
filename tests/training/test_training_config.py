import dataclasses
from pathlib import Path

import pytest

from algoarcade.rewards.registry import DEFAULT_WEIGHTS
from algoarcade.training.config import PRESETS, load_config, save_config


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
