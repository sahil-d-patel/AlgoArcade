"""The runner on the fake game: Parquet logs, normalisation statistics and checkpoint replay."""

import dataclasses
from functools import partial
from pathlib import Path

import pyarrow.parquet as pq
import pytest

pytest.importorskip("stable_baselines3")

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from algoarcade.logs.schema import TABLES, validate
from algoarcade.training.config import PRESETS
from algoarcade.training.games import FAKE_STEP_BUDGET, fake_env
from algoarcade.training.runner import train

FIRERED = PRESETS["firered"][1]
# 2 envs x 64 steps per rollout, 4 rollouts: 4 episodes per env, checkpoints at 256 and 512.
FAKE = dataclasses.replace(
    FIRERED,
    env_id="fake",
    start_state="",
    total_timesteps=512,
    n_envs=2,
    checkpoint_every=256,
    eval_episodes=1,
    reward_weights={"events": 1.0, "explore": 0.1},
    ppo=dataclasses.replace(FIRERED.ppo, n_steps=FAKE_STEP_BUDGET, batch_size=32),
)


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return train(dataclasses.replace(FAKE, runs_dir=tmp_path_factory.mktemp("runs")))


def test_every_table_validates(run_dir: Path) -> None:
    for name in TABLES:
        validate(pq.read_table(run_dir / name), name)


def test_tables_agree_on_the_step_count(run_dir: Path) -> None:
    runs = pq.read_table(run_dir / "runs").to_pylist()
    assert [r["status"] for r in runs] == ["running", "finished"]
    assert runs[-1]["env_steps"] == FAKE.total_timesteps
    assert sum(pq.read_table(run_dir / "visits")["visits"].to_pylist()) == FAKE.total_timesteps
    steps = pq.read_table(run_dir / "steps")
    assert steps.num_rows == FAKE.total_timesteps // FAKE.step_log_every
    assert max(steps["env_steps"].to_pylist()) == FAKE.total_timesteps


def test_episodes_split_training_and_evaluation(run_dir: Path) -> None:
    episodes = pq.read_table(run_dir / "episodes").to_pylist()
    train_rows = [e for e in episodes if not e["is_eval"]]
    eval_rows = [e for e in episodes if e["is_eval"]]
    assert len(train_rows) == FAKE.total_timesteps // FAKE_STEP_BUDGET
    assert all(e["episode_length"] == FAKE_STEP_BUDGET for e in episodes)
    assert all(e["checkpoint_step"] is None for e in train_rows)
    assert [e["checkpoint_step"] for e in eval_rows] == [256, 512]
    assert [e["env_steps"] for e in eval_rows] == [256, 512]
    assert all(e["reward_level"] is None for e in episodes)
    assert all(e["reward_explore"] is not None for e in episodes)


def test_checkpoint_loads_with_its_statistics_and_plays_an_episode(run_dir: Path) -> None:
    checkpoints = run_dir / "checkpoints"
    assert sorted(p.name for p in checkpoints.iterdir()) == [
        "step_256.pkl",
        "step_256.zip",
        "step_512.pkl",
        "step_512.zip",
    ]
    model = PPO.load(checkpoints / "step_512.zip")
    assert model.gamma == 0.997
    env = VecNormalize.load(
        str(checkpoints / "step_512.pkl"),
        DummyVecEnv([partial(fake_env, FAKE.reward_weights)]),
    )
    assert env.norm_obs
    assert not env.norm_reward
    env.training = False
    _, lengths = evaluate_policy(model, env, n_eval_episodes=1, return_episode_rewards=True)
    assert lengths == [FAKE_STEP_BUDGET]
