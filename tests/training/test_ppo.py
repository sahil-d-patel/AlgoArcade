"""The owned PPO trainer: truncation, autoreset, value heads, logs, checkpoints and RND."""

import csv
import dataclasses
import os
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
from algoarcade.training.config import PRESETS, PPOConfig
from algoarcade.training.games import FAKE_STEP_BUDGET, fake_env

CARTPOLE = PRESETS["cartpole"][1]
FIRERED = PRESETS["firered"][1]
# As in test_runner: four 500-step rollouts, checkpoints at 1,000 and 2,000.
SMOKE = dataclasses.replace(
    CARTPOLE,
    total_timesteps=2_000,
    n_envs=2,
    checkpoint_every=1_000,
    eval_every=1_000,
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
    eval_every=256,
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
    agent = ppo.Agent(274, 6, second_value_head=True)
    model.policy.load_state_dict(ppo.sb3_state_dict(agent))
    x = torch.randn(8, 274)
    # CombinedExtractor joins the keys in the space's sorted order: grid, then vector.
    obs = {"grid": x[:, :270].reshape(8, 2, 9, 15), "vector": x[:, 270:]}
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


def test_every_checkpoint_is_saved_and_every_eval_every_steps_evaluated(tmp_path: Path) -> None:
    # As the SB3 runner's test: checkpoints at 128, 256, 384 and the final 512. Evaluations at
    # 256, the one multiple of eval_every, and at the final checkpoint, which always evaluates.
    config = dataclasses.replace(FAKE, runs_dir=tmp_path, checkpoint_every=128, eval_every=256)
    run_dir = ppo.train(config)
    assert sorted(p.name for p in (run_dir / "checkpoints").glob("*.zip")) == [
        "step_128.zip",
        "step_256.zip",
        "step_384.zip",
        "step_512.zip",
    ]
    with (run_dir / "eval.csv").open() as f:
        assert [int(row["step"]) for row in csv.DictReader(f)] == [256, 512]
    episodes = pq.read_table(run_dir / "episodes").to_pylist()
    assert [e["checkpoint_step"] for e in episodes if e["is_eval"]] == [256, 512]


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


def test_the_same_seed_trains_the_same_run(fake_run: Path, tmp_path: Path) -> None:
    again = ppo.train(dataclasses.replace(FAKE, runs_dir=tmp_path))
    for step in (256, 512):
        first = PPO.load(fake_run / "checkpoints" / f"step_{step}.zip").policy.state_dict()
        second = PPO.load(again / "checkpoints" / f"step_{step}.zip").policy.state_dict()
        for name, value in first.items():
            torch.testing.assert_close(second[name], value, rtol=0, atol=0, msg=name)
        stats = [
            VecNormalize.load(str(run / "checkpoints" / f"step_{step}.pkl"), DummyVecEnv([env]))
            for run in (fake_run, again)
            for env in [partial(fake_env, FAKE.reward_weights)]
        ]
        assert isinstance(stats[0].obs_rms, dict)
        assert isinstance(stats[1].obs_rms, dict)
        for key, rms in stats[0].obs_rms.items():
            np.testing.assert_array_equal(stats[1].obs_rms[key].mean, rms.mean)
    columns = ["env_idx", "env_steps", "action", "x", "y", "reward_total"]
    assert (
        pq.read_table(again / "steps", columns=columns)
        .sort_by([("env_steps", "ascending"), ("env_idx", "ascending")])
        .equals(
            pq.read_table(fake_run / "steps", columns=columns).sort_by(
                [("env_steps", "ascending"), ("env_idx", "ascending")]
            )
        )
    )
    returns = []
    for run in (fake_run, again):
        with (run / "eval.csv").open() as f:
            returns.append([(row["step"], row["mean_return"]) for row in csv.DictReader(f)])
    assert returns[0] == returns[1]


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


class ExitOnSecondWorker(CrashOnSecondWorker):
    def step(self, action):
        self.steps += 1
        if getattr(self, "rank", 0) == 1 and self.steps == 10:
            os._exit(1)
        return self.env.step(action)


def exiting_env(weights: dict[str, float]) -> gym.Env:
    return ExitOnSecondWorker(fake_env(weights))


def test_a_worker_that_exits_raises_and_marks_the_run_crashed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def factory(_env_id: str, _start: str, weights: dict[str, float], _obs_mode: str) -> tuple:
        return partial(exiting_env, dict(weights)), "fake", ""

    monkeypatch.setattr(runner, "env_factory", factory)
    with pytest.raises(RuntimeError, match="An env worker died"):
        ppo.train(dataclasses.replace(FAKE, runs_dir=tmp_path))
    (run_dir,) = tmp_path.iterdir()
    assert [r["status"] for r in pq.read_table(run_dir / "runs").to_pylist()] == [
        "running",
        "crashed",
    ]


def test_pixels_are_refused_because_the_agent_has_no_cnn(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="SB3 runner"):
        ppo.train(dataclasses.replace(FAKE, obs_mode="pixels", runs_dir=tmp_path))
    assert not any(tmp_path.iterdir())


def test_a_dead_worker_raises_and_marks_the_run_crashed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def factory(_env_id: str, _start: str, weights: dict[str, float], _obs_mode: str) -> tuple:
        return partial(crashing_env, dict(weights)), "fake", ""

    monkeypatch.setattr(runner, "env_factory", factory)
    with pytest.raises(RuntimeError, match="worker 1 fails"):
        ppo.train(dataclasses.replace(FAKE, runs_dir=tmp_path))
    (run_dir,) = tmp_path.iterdir()
    assert [r["status"] for r in pq.read_table(run_dir / "runs").to_pylist()] == [
        "running",
        "crashed",
    ]


UPDATE = PPOConfig(
    learning_rate=0.0003,
    n_steps=4,
    batch_size=4,
    n_epochs=2,
    gamma=0.997,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
)
# What TRN-07's trainer (d8d8a45) returned and left behind for this rollout, from the same calls.
TRN07_LOGS = {
    "train/policy_gradient_loss": -0.0014129341579973698,
    "train/value_loss": 1.6768923252820969,
    "train/entropy_loss": -0.6931222379207611,
    "train/approx_kl": 2.318248152732849e-05,
    "train/clip_fraction": 0.0,
    "train/explained_variance": -0.780814528465271,
}
TRN07_LOGPROBS = [-0.6720183491706848, -0.7097486853599548]
TRN07_VALUES = [
    [-0.18840552866458893, 0.1986779123544693],
    [-0.1453297883272171, 0.1587374210357666],
]


def hand_rollout(agent: ppo.Agent) -> ppo.Rollout:
    """4 steps x 2 envs of a 3-input, 2-action agent, with a done in env 1 at step 1."""
    obs = torch.linspace(-1.0, 1.0, 24).reshape(4, 2, 3)
    actions = torch.tensor([[0, 1], [1, 0], [1, 1], [0, 0]])
    with torch.no_grad():
        _, logprobs, _, value = agent.get_action_and_value(obs.flatten(0, 1), actions.flatten())
    return ppo.Rollout(
        obs=obs,
        actions=actions,
        logprobs=logprobs.reshape(4, 2),
        rewards=torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [1.0, 1.0]]),
        values=value[:, 0].reshape(4, 2),
        dones=torch.tensor([[False, False], [False, True], [False, False], [False, False]]),
        final_values=torch.tensor([[0.0, 0.0], [0.0, 0.5], [0.0, 0.0], [0.0, 0.0]]),
        int_values=value[:, 1].reshape(4, 2),
    )


def test_int_coef_zero_updates_as_the_trn07_trainer_did() -> None:
    # The parity gate (TRN-07) passed on that trainer. With int_coef 0 no RND code runs, so the
    # same update must give the same numbers.
    torch.manual_seed(0)
    agent = ppo.Agent(3, 2, second_value_head=True)
    optimizer = torch.optim.Adam(agent.parameters(), lr=UPDATE.learning_rate, eps=ppo.ADAM_EPS)
    rollout = hand_rollout(agent)
    advantages = ppo.gae(rollout, torch.zeros(2), UPDATE.gamma, UPDATE.gae_lambda)
    logs = ppo.update(agent, optimizer, rollout, advantages, UPDATE, np.random.default_rng(0))
    assert logs == pytest.approx(TRN07_LOGS, rel=1e-3)
    with torch.no_grad():
        _, logprobs, _, values = agent.get_action_and_value(rollout.obs[0], rollout.actions[0])
    np.testing.assert_allclose(logprobs, TRN07_LOGPROBS, rtol=1e-3)
    np.testing.assert_allclose(values, TRN07_VALUES, rtol=1e-3)


def test_the_network_returns_two_values_and_each_head_learns_its_return() -> None:
    # Extrinsic returns of 0 and intrinsic returns of 1 at every step. Swapping the heads or the
    # returns fails.
    torch.manual_seed(0)
    agent = ppo.Agent(3, 2, second_value_head=True)
    rnd = ppo.RND(np.zeros((1, 3)), gamma=0.99)
    optimizer = torch.optim.Adam([*agent.parameters(), *rnd.predictor.parameters()], lr=0.01)
    rollout = hand_rollout(agent)
    intrinsic = ppo.Intrinsic(
        rnd, 0.5, 1 - rollout.int_values, torch.zeros(4, 2, 3), torch.zeros(4, 2)
    )
    for _ in range(100):
        ppo.update(
            agent, optimizer, rollout, -rollout.values, UPDATE, np.random.default_rng(0), intrinsic
        )
    with torch.no_grad():
        values = agent.get_value(rollout.obs.flatten(0, 1))
    assert values.shape == (8, 2)
    torch.testing.assert_close(values, torch.tensor([[0.0, 1.0]]).expand(8, 2), atol=0.05, rtol=0)


def test_intrinsic_returns_ignore_a_done() -> None:
    # One env with a done at step 2. Intrinsic rewards 1 to 4, the second head's value 0.5 at
    # every step and after the last, gamma 0.99 and lambda 1, so each return is the discounted
    # sum bootstrapped from 0.5. Masking the done would make step 2's return 3.0.
    rollout = ppo.Rollout(
        obs=torch.zeros(4, 1, 1),
        actions=torch.zeros(4, 1, dtype=torch.long),
        logprobs=torch.zeros(4, 1),
        rewards=torch.zeros(4, 1),
        values=torch.zeros(4, 1),
        dones=torch.tensor([[False], [False], [True], [False]]),
        final_values=torch.zeros(4, 1),
        int_values=torch.full((4, 1), 0.5),
    )
    rewards = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    advantages = ppo.intrinsic_gae(rollout, rewards, torch.tensor([0.5]), 0.99, 1.0)
    r3 = 4 + 0.99 * 0.5
    r2 = 3 + 0.99 * r3
    r1 = 2 + 0.99 * r2
    r0 = 1 + 0.99 * r1
    torch.testing.assert_close(
        (advantages + rollout.int_values)[:, 0], torch.tensor([r0, r1, r2, r3])
    )


def test_whitened_input_stays_within_5_and_a_constant_column_gives_0() -> None:
    rng = np.random.default_rng(0)
    warm = np.column_stack(
        [rng.normal(10.0, 3.0, 1000), np.full(1000, 7.0), rng.integers(0, 2, 1000)]
    )
    rnd = ppo.RND(warm, gamma=0.99)
    # Far outside the warmup's range, and the constant column at a value it never had.
    white = ppo.whiten(np.vstack([warm, [[1e6, 7.0, 1.0], [-1e6, 8.0, 0.0]]]), rnd.input_rms)
    assert np.isfinite(white).all()
    assert white.min() >= -5.0
    assert white.max() <= 5.0
    assert white[-2:, 0].tolist() == [5.0, -5.0]
    assert (white[:-1, 1] == 0.0).all()
    assert white[-1, 1] == 5.0


def test_intrinsic_reward_is_divided_by_the_running_std_of_intrinsic_returns() -> None:
    rnd = ppo.RND(np.zeros((1, 3)), gamma=0.5)
    novelty = np.array([[1.0, 2.0], [3.0, 4.0]])
    # Two steps of two envs. Each env's return over time: [1, 2], then
    # [0.5 * 1 + 3, 0.5 * 2 + 4] = [3.5, 5]. Those four have variance 2.296875; the statistics'
    # prior of 1e-4 samples moves the std by under 1e-4.
    np.testing.assert_allclose(rnd.scale(novelty), novelty / np.sqrt(2.296875), rtol=1e-4)
    # The filter carries into the next rollout, and nothing resets it.
    rnd.scale(np.array([[1.0, 1.0]]))
    np.testing.assert_array_equal(rnd.returns, [0.5 * 3.5 + 1, 0.5 * 5 + 1])


INFO = {"location": (3, 0, 5, 6), "badge_flags": 0b101, "milestone_count": 3, "party_level_sum": 12}


def test_the_rnd_input_is_the_seven_fields() -> None:
    assert list(ppo.rnd_fields(INFO)) == [
        "map_group",
        "map_num",
        "x",
        "y",
        "badge_bits",
        "milestone_bits",
        "party_level_sum",
    ]
    # Badges 1 and 3, the first three milestones. A counter, the money or anything else in the
    # info never reaches the input.
    expected = [3, 0, 5, 6, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 12]
    noisy = INFO | {"frame": 123_456, "money": 3000, "story_flags": 2}
    np.testing.assert_array_equal(ppo.rnd_input([INFO, noisy]), [expected, expected])
    # The fake game's (room, x, y) gives its room as map_num, as the logs do.
    assert ppo.rnd_fields(INFO | {"location": (1, 5, 6)})["map_num"] == [1]


def test_an_rnd_run_trains_on_the_fake_game(tmp_path: Path) -> None:
    config = dataclasses.replace(FAKE, runs_dir=tmp_path, int_coef=0.5, second_value_head=True)
    run_dir = ppo.train(config)
    with (run_dir / "progress.csv").open() as f:
        progress = list(csv.DictReader(f))
    assert len(progress) == 4
    assert all(float(row["train/rnd_loss"]) > 0 for row in progress)
    assert all(float(row["train/intrinsic_value_loss"]) >= 0 for row in progress)
    assert all(float(row["rollout/intrinsic_reward_mean"]) > 0 for row in progress)
    assert all(float(row["rollout/intrinsic_return_std"]) > 0 for row in progress)
    for name in TABLES:
        validate(pq.read_table(run_dir / name), name)
    # The warmup's random-policy steps are not training steps.
    assert pq.read_table(run_dir / "runs").to_pylist()[-1]["env_steps"] == config.total_timesteps
    assert sum(pq.read_table(run_dir / "visits")["visits"].to_pylist()) == config.total_timesteps
