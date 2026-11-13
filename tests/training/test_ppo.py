"""The owned PPO trainer: truncation, autoreset, value heads, logs and checkpoints."""

import csv
import dataclasses
from functools import partial
from pathlib import Path

import gymnasium as gym
import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

pytest.importorskip("stable_baselines3")

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from algoarcade.logs.schema import TABLES, validate
from algoarcade.training import ppo, runner
from algoarcade.training.config import PRESETS
from algoarcade.training.games import FAKE_STEP_BUDGET, fake_env

CARTPOLE = PRESETS["cartpole"][1]
FIRERED = PRESETS["firered"][1]
# As in test_runner: four 500-step rollouts, checkpoints at 1,000 and 2,000.
SMOKE = dataclasses.replace(
    CARTPOLE,
    total_timesteps=2_000,
    n_envs=2,
    checkpoint_every=1_000,
    eval_episodes=2,
    ppo=dataclasses.replace(CARTPOLE.ppo, n_steps=250, batch_size=100),
)
# As in test_game_runner: 2 envs x 64 steps, 4 rollouts, checkpoints at 256 and 512.
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
GAMMA = 0.9
LAMBDA = 0.95


class Clock(gym.Env):
    """Pays 1 per step and observes the step count. Ends after three steps, by termination or
    truncation."""

    observation_space = gym.spaces.Box(0.0, 10.0, (1,), np.float32)
    action_space = gym.spaces.Discrete(2)

    def __init__(self, terminate: bool) -> None:
        self.terminate = terminate
        self.t = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return np.array([0.0], np.float32), {}

    def step(self, action):
        self.t += 1
        end = self.t == 3
        obs = np.array([self.t], np.float32)
        return obs, 1.0, end and self.terminate, end and not self.terminate, {}


class FixedValue(ppo.Agent):
    """V(obs) = 10 * obs."""

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return 10 * x[:, :1]


def returns_at_the_end(terminate: bool) -> float:
    envs = ppo.make_envs(partial(Clock, terminate), 1)
    try:
        obs = ppo.observe(envs.reset(seed=0)[0], envs.single_observation_space, None)
        agent = FixedValue(1, 2, second_value_head=False)
        rollout, last = ppo.collect(envs, agent, None, obs, 4)
    finally:
        envs.close()
    last_value = agent.get_value(torch.as_tensor(last))[:, 0]
    advantages = ppo.gae(rollout, last_value, GAMMA, LAMBDA)
    assert rollout.dones[:, 0].tolist() == [False, False, True, False]
    return float((advantages + rollout.values)[2, 0])


def test_truncation_bootstrap() -> None:
    # The step budget cut the episode at obs 3, and the reset obs 0 came back in its place.
    assert returns_at_the_end(terminate=False) == pytest.approx(1 + GAMMA * 10 * 3)


def test_termination_does_not_bootstrap() -> None:
    assert returns_at_the_end(terminate=True) == pytest.approx(1.0)


def test_no_cross_episode_transition() -> None:
    weights = {"explore": 0.1}
    envs = ppo.make_envs(partial(fake_env, weights), 1)
    space = envs.single_observation_space
    try:
        obs = ppo.observe(envs.reset(seed=0)[0], space, None)
        agent = ppo.Agent(obs.shape[1], 6, second_value_head=False)
        rollout, _ = ppo.collect(envs, agent, None, obs, FAKE_STEP_BUDGET + 2)
    finally:
        envs.close()
    reset_obs, _ = fake_env(weights).reset(seed=0)
    reset = torch.as_tensor(ppo.observe({k: v[None] for k, v in reset_obs.items()}, space, None))
    dones = rollout.dones[:, 0].tolist()
    assert dones == [False] * (FAKE_STEP_BUDGET - 1) + [True, False, False]
    torch.testing.assert_close(rollout.obs[0], reset)
    assert not torch.equal(rollout.obs[FAKE_STEP_BUDGET - 1], reset)
    # Under next-step autoreset this would be the final obs, and the step after it a reset
    # whose action is ignored.
    torch.testing.assert_close(rollout.obs[FAKE_STEP_BUDGET], reset)


@pytest.mark.parametrize(("second", "outputs"), [(False, 1), (True, 2)])
def test_second_value_head(second: bool, outputs: int) -> None:
    agent = ppo.Agent(5, 3, second_value_head=second)
    x = torch.zeros(4, 5)
    assert agent.get_value(x).shape == (4, outputs)
    assert agent.get_action_and_value(x)[3].shape == (4, outputs)


def test_sb3_checkpoint_acts_like_the_agent() -> None:
    env = make_vec_env(partial(fake_env, {"explore": 0.1}), n_envs=1)
    model = runner.build_model(FAKE, env)
    agent = ppo.Agent(139, 6, second_value_head=True)
    model.policy.load_state_dict(ppo.sb3_state_dict(agent))
    x = torch.randn(8, 139)
    obs = {"vector": x[:, :4], "visited": x[:, 4:].reshape(8, 1, 9, 15)}
    logits = agent.actor(x)
    distribution = model.policy.get_distribution(obs).distribution
    assert isinstance(distribution, torch.distributions.Categorical)
    torch.testing.assert_close(distribution.logits, torch.log_softmax(logits, dim=1))
    torch.testing.assert_close(model.policy.predict_values(obs), agent.get_value(x)[:, :1])


def test_sb3_has_the_settings_the_owned_trainer_fixes() -> None:
    model = runner.build_model(FAKE, make_vec_env(partial(fake_env, {}), n_envs=1))
    assert model.normalize_advantage
    assert model.clip_range_vf is None
    assert model.target_kl is None
    assert model.policy.optimizer.defaults["eps"] == ppo.ADAM_EPS
    assert model.policy.ortho_init
    assert isinstance(model.policy.activation_fn(), torch.nn.Tanh)
    assert model.policy.net_arch == {"pi": [64, 64], "vf": [64, 64]}


def test_smoke_run_writes_the_runner_artifacts(tmp_path: Path) -> None:
    run_dir = ppo.train(dataclasses.replace(SMOKE, runs_dir=tmp_path))
    assert "-owned-seed0-" in run_dir.name
    with (run_dir / "eval.csv").open() as f:
        assert [int(row["step"]) for row in csv.DictReader(f)] == [1_000, 2_000]
    with (run_dir / "progress.csv").open() as f:
        progress = list(csv.DictReader(f))
    # 2,000 steps in 500-step rollouts: four updates.
    assert [int(row["time/total_timesteps"]) for row in progress] == [500, 1_000, 1_500, 2_000]
    assert sorted(p.name for p in (run_dir / "checkpoints").iterdir()) == [
        "step_1000.zip",
        "step_2000.zip",
    ]


@pytest.fixture(scope="module")
def fake_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return ppo.train(dataclasses.replace(FAKE, runs_dir=tmp_path_factory.mktemp("runs")))


def test_owned_rows_pass_the_schema(fake_run: Path) -> None:
    for name in TABLES:
        validate(pq.read_table(fake_run / name), name)
    runs = pq.read_table(fake_run / "runs").to_pylist()
    assert [(r["status"], r["trainer"]) for r in runs] == [
        ("running", "owned"),
        ("finished", "owned"),
    ]
    assert runs[-1]["env_steps"] == FAKE.total_timesteps
    assert sum(pq.read_table(fake_run / "visits")["visits"].to_pylist()) == FAKE.total_timesteps
    episodes = pq.read_table(fake_run / "episodes").to_pylist()
    train_rows = [e for e in episodes if not e["is_eval"]]
    assert len(train_rows) == FAKE.total_timesteps // FAKE_STEP_BUDGET
    assert all(e["episode_length"] == FAKE_STEP_BUDGET for e in episodes)
    assert [e["checkpoint_step"] for e in episodes if e["is_eval"]] == [256, 512]


def test_the_trn05_evaluation_path_loads_the_checkpoint(fake_run: Path) -> None:
    checkpoints = fake_run / "checkpoints"
    model = PPO.load(checkpoints / "step_512.zip")
    assert model.gamma == FAKE.ppo.gamma
    env = VecNormalize.load(
        str(checkpoints / "step_512.pkl"),
        DummyVecEnv([partial(fake_env, FAKE.reward_weights)]),
    )
    assert env.norm_obs
    assert not env.norm_reward
    # The training statistics: every training observation, the resets included.
    assert isinstance(env.obs_rms, dict)
    assert env.obs_rms["vector"].count > FAKE.total_timesteps
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
    with pytest.raises(RuntimeError, match="worker 1 fails"):
        ppo.train(dataclasses.replace(FAKE, runs_dir=tmp_path))
    (run_dir,) = tmp_path.iterdir()
    assert [r["status"] for r in pq.read_table(run_dir / "runs").to_pylist()] == [
        "running",
        "crashed",
    ]
