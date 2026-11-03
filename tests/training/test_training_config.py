import dataclasses
from pathlib import Path

import pytest

from algoarcade.training.config import PRESETS, load_config, save_config


@pytest.mark.parametrize("name", ["cartpole", "acrobot"])
def test_saved_config_reloads_equal(name: str, tmp_path: Path) -> None:
    config = dataclasses.replace(PRESETS[name][1], seed=3, runs_dir=tmp_path / "runs")
    path = tmp_path / "config.json"
    save_config(config, path)
    assert load_config(path) == config
