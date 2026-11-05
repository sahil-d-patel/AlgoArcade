"""The runner on the fake game: Parquet logs, normalisation statistics and checkpoint replay."""

import dataclasses
import time
from functools import partial
from pathlib import Path

import gymnasium as gym
import numpy as np
import pyarrow.parquet as pq
import pytest

pytest.importorskip("stable_baselines3")

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from algoarcade.backends import fake
from algoarcade.envs.env import AlgoArcadeEnv
from algoarcade.logs.schema import TABLES, validate
from algoarcade.training import runner
from algoarcade.training.config import PRESETS
from algoarcade.training.games import FAKE_STEP_BUDGET, FIRERED_ROM, env_factory, fake_env
from algoarcade.training.parquet import ParquetLog
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


def test_the_start_row_is_readable_before_the_run_ends(run_dir: Path) -> None:
    # An open Parquet file cannot be read, so a killed run shows `running` only if the start
    # row's file was closed at launch.
    first = pq.read_table(run_dir / "runs" / "train-00000.parquet").to_pylist()
    assert [r["status"] for r in first] == ["running"]


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
    # The fake's RAM starts with a level sum of 5 and nothing in the game changes it.
    assert {(e["party_level_sum"], e["badge_count"], e["milestone_count"]) for e in episodes} == {
        (5, 0, 0)
    }


def test_a_count_changes_only_once_two_reads_agree() -> None:
    env = fake_env({"explore": 0.1})
    env.reset(seed=0)
    unwrapped = env.unwrapped
    assert isinstance(unwrapped, AlgoArcadeEnv)
    backend = unwrapped.backend
    counts = []
    for badges in (0, 1, 0, 1, 1, 1):
        ram = bytearray(backend.save_state())
        ram[fake.BADGES] = badges
        backend.load_state(bytes(ram))
        counts.append(env.step(5)[4]["badge_count"])
    assert counts == [0, 0, 0, 0, 1, 1]


@pytest.mark.rom
def test_firered_logs_the_readers_values(tmp_path: Path) -> None:
    if not FIRERED_ROM.is_file():
        pytest.skip("Needs the local FireRed ROM")
    config = dataclasses.replace(FIRERED, start_state="seg_02_pallet", n_envs=1)
    env_fn, _, _ = env_factory("firered", config.start_state, config.reward_weights)
    # 64-step episodes keep the test short; the manifest's budget is 4,096.
    venv = DummyVecEnv([lambda: gym.wrappers.TimeLimit(env_fn(), 64)])
    log = ParquetLog(venv, config, tmp_path / "run_id=t", time.perf_counter())
    log.reset()
    for step in range(128):
        log.step(np.array([2 + step % 2]))  # left, right
    log.close()

    steps = pq.read_table(tmp_path / "run_id=t" / "steps")
    episodes = pq.read_table(tmp_path / "run_id=t" / "episodes")
    validate(steps, "steps")
    validate(episodes, "episodes")
    # seg_02_pallet's party is one level 6 Pokemon, with three whitelist flags and two
    # milestones set; walking left and right in Pallet Town changes none of them.
    assert [
        (e["party_level_sum"], e["story_flags"], e["badge_count"], e["milestone_count"])
        for e in episodes.to_pylist()
    ] == [(6, 3, 0, 2)] * 2
    assert set(steps["milestone_count"].to_pylist()) == {2}
    assert set(steps["in_battle"].to_pylist()) == {False}
    assert all(
        e[f"reward_{name}"] is not None
        for e in episodes.to_pylist()
        for name in config.reward_weights
    )


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


class CrashOnSecondWorker(gym.Wrapper):
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rank = seed - FAKE.seed
        self.steps = 0
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        self.steps += 1
        if getattr(self, "rank", 0) == 1 and self.steps == 10:
            raise RuntimeError("worker 1 fails")
        return self.env.step(action)


def crashing_env(weights: dict[str, float]) -> gym.Env:
    return CrashOnSecondWorker(fake_env(weights))


def test_a_dead_worker_raises_and_marks_the_run_crashed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def factory(_env_id: str, _start: str, weights: dict[str, float]) -> tuple:
        return partial(crashing_env, dict(weights)), "fake", ""

    monkeypatch.setattr(runner, "env_factory", factory)
    with pytest.raises(RuntimeError, match="worker died"):
        train(dataclasses.replace(FAKE, runs_dir=tmp_path))
    (run_dir,) = tmp_path.iterdir()
    assert [r["status"] for r in pq.read_table(run_dir / "runs").to_pylist()] == [
        "running",
        "crashed",
    ]
