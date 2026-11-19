"""PQN: Q(lambda) returns, truncation through the shared rollout, exploration, logs and
checkpoints."""

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

from algoarcade.logs.schema import TABLES, validate
from algoarcade.training import ppo, pqn
from algoarcade.training.config import PQN_CLEANRL, PRESETS
from algoarcade.training.games import FAKE_STEP_BUDGET

CARTPOLE = PRESETS["pqn_cartpole"][1]
FIRERED = PRESETS["firered"][1]
SMOKE = dataclasses.replace(
    CARTPOLE,
    total_timesteps=2_000,
    n_envs=2,
    checkpoint_every=1_000,
    eval_every=1_000,
    eval_episodes=2,
    pqn=dataclasses.replace(PQN_CLEANRL, n_steps=250),
)
# 2 envs x 64 steps, 4 rollouts, checkpoints at 256 and 512, as test_ppo's FAKE.
FAKE = dataclasses.replace(
    FIRERED,
    env_id="fake",
    start_state="",
    total_timesteps=512,
    n_envs=2,
    checkpoint_every=256,
    eval_every=256,
    eval_episodes=1,
    reward_weights={"events": 1.0, "explore": 0.1},
    pqn=dataclasses.replace(PQN_CLEANRL, n_steps=FAKE_STEP_BUDGET, gamma=FIRERED.ppo.gamma),
)
GAMMA = 0.9
LAMBDA = 0.5


def test_q_lambda_returns_by_hand() -> None:
    # Three envs over three steps, rewards 1, 2, 3 and max Q 10, 20, 30 in each. Env 0
    # terminates at step 3, env 1 is truncated there with max Q(final_obs) 40, env 2 runs on to
    # a next input with max Q 50.
    f64 = torch.float64
    no, yes = False, True
    rollout = ppo.Rollout(
        obs=torch.zeros(3, 3, 1, dtype=f64),
        actions=torch.zeros(3, 3, dtype=torch.long),
        logprobs=torch.zeros(3, 3, dtype=f64),
        rewards=torch.tensor([[1.0] * 3, [2.0] * 3, [3.0] * 3], dtype=f64),
        values=torch.tensor([[10.0] * 3, [20.0] * 3, [30.0] * 3], dtype=f64),
        dones=torch.tensor([[no, no, no], [no, no, no], [yes, yes, no]]),
        final_values=torch.tensor([[0.0] * 3, [0.0] * 3, [0.0, 40.0, 0.0]], dtype=f64),
    )
    returns = pqn.q_lambda_returns(rollout, torch.full((3,), 50.0, dtype=f64), GAMMA, LAMBDA)
    # R_2 = r_2 + g * next; R_t = r_t + g * (l * R_{t+1} + (1 - l) * maxQ_{t+1}).
    # Terminated: R_2 = 3, R_1 = 2 + 0.9 * (0.5 * 3 + 0.5 * 30) = 16.85,
    #   R_0 = 1 + 0.9 * (0.5 * 16.85 + 0.5 * 20) = 17.5825.
    # Truncated: R_2 = 3 + 0.9 * 40 = 39, R_1 = 2 + 0.9 * (19.5 + 15) = 33.05,
    #   R_0 = 1 + 0.9 * (16.525 + 10) = 24.8725.
    # Running: R_2 = 3 + 0.9 * 50 = 48, R_1 = 2 + 0.9 * (24 + 15) = 37.1,
    #   R_0 = 1 + 0.9 * (18.55 + 10) = 26.695.
    expected = torch.tensor(
        [[17.5825, 24.8725, 26.695], [16.85, 33.05, 37.1], [3.0, 39.0, 48.0]], dtype=f64
    )
    torch.testing.assert_close(returns, expected, rtol=0, atol=1e-6)


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


class FixedQ(pqn.QNetwork):
    """Q(obs) = (10 * obs, 0)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([10 * x[:, :1], torch.zeros_like(x[:, :1])], dim=1)


def return_at_the_end(terminate: bool) -> float:
    envs = ppo.make_envs(partial(Clock, terminate), 1)
    try:
        obs = ppo.observe(envs.reset(seed=0)[0], envs.single_observation_space, None)
        q_network = FixedQ(1, 2)
        rollout, last = ppo.collect(envs, q_network, None, obs, 4)
    finally:
        envs.close()
    last_value = q_network.get_value(torch.as_tensor(last))[:, 0]
    returns = pqn.q_lambda_returns(rollout, last_value, GAMMA, LAMBDA)
    assert rollout.dones[:, 0].tolist() == [False, False, True, False]
    return float(returns[2, 0])


def test_truncation_bootstraps_from_the_final_observation() -> None:
    # The step budget cut the episode at obs 3; the reset obs 0 came back in its place.
    assert return_at_the_end(terminate=False) == pytest.approx(1 + GAMMA * 10 * 3)


def test_termination_does_not_bootstrap() -> None:
    assert return_at_the_end(terminate=True) == pytest.approx(1.0)


def test_epsilon_falls_over_the_exploration_fraction() -> None:
    config = PQN_CLEANRL
    assert pqn.epsilon(config, 1_000, 0) == 1.0
    assert pqn.epsilon(config, 1_000, 250) == pytest.approx((1.0 + 0.05) / 2)
    assert pqn.epsilon(config, 1_000, 500) == pytest.approx(0.05)
    assert pqn.epsilon(config, 1_000, 900) == 0.05


@pytest.mark.parametrize(("epsilon", "greedy_share"), [(0.0, 1.0), (1.0, 0.5)])
def test_actions_are_epsilon_greedy(epsilon: float, greedy_share: float) -> None:
    torch.manual_seed(0)
    q_network = FixedQ(1, 2)
    q_network.epsilon = epsilon
    action, _, _, value = q_network.get_action_and_value(torch.ones(10_000, 1))
    assert float((action == 0).float().mean()) == pytest.approx(greedy_share, abs=0.02)
    assert torch.equal(value, torch.full((10_000, 1), 10.0))


def test_the_ppo_trainer_refuses_a_pqn_config(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="trains PQN"):
        ppo.train(dataclasses.replace(SMOKE, runs_dir=tmp_path))


def test_smoke_run_writes_the_runner_artifacts(tmp_path: Path) -> None:
    run_dir = pqn.train(dataclasses.replace(SMOKE, runs_dir=tmp_path))
    assert "-pqn-seed0-" in run_dir.name
    with (run_dir / "eval.csv").open() as f:
        assert [int(row["step"]) for row in csv.DictReader(f)] == [1_000, 2_000]
    with (run_dir / "progress.csv").open() as f:
        progress = list(csv.DictReader(f))
    assert [int(row["time/total_timesteps"]) for row in progress] == [500, 1_000, 1_500, 2_000]
    assert float(progress[-1]["train/learning_rate"]) == pytest.approx(
        0.25 * PQN_CLEANRL.learning_rate
    )
    assert sorted(p.name for p in (run_dir / "checkpoints").iterdir()) == [
        "step_1000.pt",
        "step_2000.pt",
    ]
    q_network = pqn.QNetwork(4, 2)
    q_network.load_state_dict(torch.load(run_dir / "checkpoints" / "step_2000.pt"))


def test_evaluation_waits_for_eval_every(tmp_path: Path) -> None:
    config = dataclasses.replace(SMOKE, checkpoint_every=500, runs_dir=tmp_path)
    run_dir = pqn.train(config)
    with (run_dir / "eval.csv").open() as f:
        assert [int(row["step"]) for row in csv.DictReader(f)] == [1_000, 2_000]
    assert len(list((run_dir / "checkpoints").iterdir())) == 4


def test_fake_run_passes_the_schema(tmp_path: Path) -> None:
    run_dir = pqn.train(dataclasses.replace(FAKE, runs_dir=tmp_path))
    for name in TABLES:
        validate(pq.read_table(run_dir / name), name)
    runs = pq.read_table(run_dir / "runs").to_pylist()
    assert [(r["status"], r["algorithm"], r["trainer"]) for r in runs] == [
        ("running", "pqn", "owned"),
        ("finished", "pqn", "owned"),
    ]
    assert runs[-1]["env_steps"] == FAKE.total_timesteps
    episodes = pq.read_table(run_dir / "episodes").to_pylist()
    assert [e["checkpoint_step"] for e in episodes if e["is_eval"]] == [256, 512]
    assert sorted(p.name for p in (run_dir / "checkpoints").iterdir()) == [
        "step_256.pkl",
        "step_256.pt",
        "step_512.pkl",
        "step_512.pt",
    ]
