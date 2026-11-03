"""Skipped without the train group. CI installs it, so these run on all three OSes."""

import csv
import dataclasses
import json
from pathlib import Path

import pytest

pytest.importorskip("stable_baselines3")

from stable_baselines3.common.env_util import make_vec_env

from algoarcade.training.config import PRESETS, PPOConfig, load_config
from algoarcade.training.runner import build_model, train

CARTPOLE = PRESETS["cartpole"][1]


def test_model_takes_every_ppo_value_from_config() -> None:
    ppo = PPOConfig(
        learning_rate=0.0007,
        n_steps=64,
        batch_size=32,
        n_epochs=2,
        gamma=0.977,
        gae_lambda=0.9,
        clip_range=0.15,
        ent_coef=0.02,
    )
    config = dataclasses.replace(CARTPOLE, n_envs=1, ppo=ppo)
    model = build_model(config, make_vec_env(config.env_id, n_envs=1))
    assert model.gamma == ppo.gamma
    assert model.learning_rate == ppo.learning_rate
    assert model.n_steps == ppo.n_steps
    assert model.batch_size == ppo.batch_size
    assert model.n_epochs == ppo.n_epochs
    assert model.gae_lambda == ppo.gae_lambda
    clip_range = model.clip_range  # SB3 wraps the float in a schedule at setup
    assert callable(clip_range)
    assert clip_range(1.0) == ppo.clip_range
    assert model.ent_coef == ppo.ent_coef


def test_smoke_run_writes_all_four_artifacts(tmp_path: Path) -> None:
    config = dataclasses.replace(
        CARTPOLE,
        total_timesteps=2_000,
        n_envs=2,
        checkpoint_every=1_000,
        eval_episodes=2,
        runs_dir=tmp_path,
        ppo=dataclasses.replace(CARTPOLE.ppo, n_steps=250, batch_size=100),
    )
    run_dir = train(config)

    assert list(tmp_path.iterdir()) == [run_dir]
    assert load_config(run_dir / "config.json") == config

    git = json.loads((run_dir / "git.json").read_text())
    assert len(git["commit"]) == 40
    assert isinstance(git["dirty"], bool)

    assert len(list((run_dir / "checkpoints").glob("*.zip"))) >= 2

    with (run_dir / "eval.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert [int(row["step"]) for row in rows] == [1_000, 2_000]
