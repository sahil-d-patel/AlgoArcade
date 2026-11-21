"""PQN, derived from CleanRL's pqn.py (TRN-09): Q-learning on parallel envs with a LayerNorm
network, Q(lambda) returns and no replay buffer or target network.

Source: https://github.com/vwxyzjn/cleanrl/blob/1ed80620842b4cdeb1edc07e12825dff18091da9/cleanrl/pqn.py
That commit carries the done-masking fix to the return (CleanRL PR 494).

    MIT License

    Copyright (c) 2019 CleanRL developers

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

What we changed, and why:

- The rollout, the env setup, the Parquet and progress logging, checkpoints and evaluation are
  the owned PPO file's (algoarcade.training.ppo), so both arms of E1 sample the same way.
- Truncation bootstraps from max_a Q(info["final_obs"], a); only termination ends the return.
  CleanRL merges the two, and every episode here ends on a step budget.
- Epsilon is set after each env step from the steps taken so far, as CleanRL's is.
- Training stops at the first whole rollout at or past total_timesteps, as the PPO file does.
- A checkpoint is the network's state dict, step_<n>.pt, beside the VecNormalize pickle.
  Evaluation plays argmax Q, or epsilon-greedy at end_e when the config samples.

    uv run python -m algoarcade.training.pqn pqn_cartpole --seed 0
"""

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import tyro
from gymnasium import spaces
from stable_baselines3.common.type_aliases import PolicyPredictor
from stable_baselines3.common.vec_env import VecEnv
from torch import nn
from torch.nn import functional as F

from algoarcade.training import ppo
from algoarcade.training.config import PRESETS, PQNConfig, TrainConfig
from algoarcade.training.parquet import RunLog


class QNetwork(nn.Module):
    """CleanRL's network: 120 and 84 units, each followed by LayerNorm and ReLU. As a ppo.Policy
    it acts epsilon-greedy and its value is max_a Q."""

    def __init__(self, obs_dim: int, n_actions: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            ppo.layer_init(nn.Linear(obs_dim, 120)),
            nn.LayerNorm(120),
            nn.ReLU(),
            ppo.layer_init(nn.Linear(120, 84)),
            nn.LayerNorm(84),
            nn.ReLU(),
            ppo.layer_init(nn.Linear(84, n_actions)),
        )
        self.epsilon = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self(x).max(dim=1, keepdim=True).values

    def get_action_and_value(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self(x)
        greedy = q.argmax(dim=1)
        uniform = torch.randint(0, q.shape[1], greedy.shape, device=x.device)
        explore = torch.rand(greedy.shape, device=x.device) < self.epsilon
        action = torch.where(explore, uniform, greedy)
        zeros = torch.zeros_like(greedy, dtype=q.dtype)
        return action, zeros, zeros, q.max(dim=1, keepdim=True).values


def epsilon(pqn: PQNConfig, total_timesteps: int, step: int) -> float:
    duration = pqn.exploration_fraction * total_timesteps
    return max(pqn.start_e + (pqn.end_e - pqn.start_e) * step / duration, pqn.end_e)


def q_lambda_returns(
    rollout: ppo.Rollout, last_value: torch.Tensor, gamma: float, q_lambda: float
) -> torch.Tensor:
    """Q(lambda) targets. `values` holds max_a Q(obs_t, a) and last_value max_a Q of the next
    input. After a done the next value is final_values: max_a Q(final_obs, a) at a truncation, 0
    at a termination, and no return flows back across it."""
    returns = torch.zeros_like(rollout.rewards)
    n_steps = len(rollout.rewards)
    for t in reversed(range(n_steps)):
        if t == n_steps - 1:
            next_return = last_value
        else:
            next_return = q_lambda * returns[t + 1] + (1 - q_lambda) * rollout.values[t + 1]
        next_return = torch.where(rollout.dones[t], rollout.final_values[t], next_return)
        returns[t] = rollout.rewards[t] + gamma * next_return
    return returns


def update(  # noqa: PLR0913 - CleanRL's update loop, with its state passed in
    q_network: QNetwork,
    optimizer: torch.optim.Optimizer,
    rollout: ppo.Rollout,
    returns: torch.Tensor,
    pqn: PQNConfig,
    rng: np.random.Generator,
) -> dict[str, float]:
    """pqn.n_epochs passes over the rollout in pqn.n_minibatches shuffled minibatches, regressing
    Q(obs, action) on the return."""
    b_obs = rollout.obs.flatten(0, 1)
    b_actions = rollout.actions.flatten()
    b_returns = returns.flatten()
    size = len(b_obs) // pqn.n_minibatches
    losses, q_values = [], []
    for _ in range(pqn.n_epochs):
        order = torch.as_tensor(rng.permutation(len(b_obs)), device=b_obs.device)
        for start in range(0, len(b_obs), size):
            mb = order[start : start + size]
            q = q_network(b_obs[mb]).gather(1, b_actions[mb, None])[:, 0]
            loss = F.mse_loss(q, b_returns[mb])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q_network.parameters(), pqn.max_grad_norm)
            optimizer.step()
            losses.append(loss.item())
            q_values.append(q.mean().item())
    return {"train/td_loss": float(np.mean(losses)), "train/q_values": float(np.mean(q_values))}


def learn(
    config: TrainConfig,
    envs: gym.vector.VectorEnv,
    checkpoint: ppo.Checkpoint,
    log: RunLog | None,
    logger: Any,
) -> None:
    pqn = config.pqn
    assert pqn is not None
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_rms, obs = ppo.begin(config, envs, log)
    action_space = envs.single_action_space
    assert isinstance(action_space, spaces.Discrete)
    q_network = QNetwork(obs.shape[1], int(action_space.n)).to(device)
    optimizer = torch.optim.RAdam(q_network.parameters(), lr=pqn.learning_rate)
    steps = ppo.Steps(config, log, lambda step, due: checkpoint(q_network, obs_rms, step, due))

    def after_step(*args: Any) -> None:
        # Afterwards: a checkpoint inside steps() evaluates with the evaluation epsilon.
        steps(*args)
        q_network.epsilon = epsilon(pqn, config.total_timesteps, steps.n)

    while not steps.done:
        learning_rate = (1 - steps.n / config.total_timesteps) * pqn.learning_rate
        optimizer.param_groups[0]["lr"] = learning_rate
        rollout, obs = ppo.collect(envs, q_network, obs_rms, obs, pqn.n_steps, after_step)
        with torch.no_grad():
            last_value = q_network.get_value(torch.as_tensor(obs, device=device))[:, 0]
            returns = q_lambda_returns(rollout, last_value, pqn.gamma, pqn.q_lambda)
        for key, value in update(q_network, optimizer, rollout, returns, pqn, rng).items():
            logger.record(key, value)
        logger.record("train/epsilon", q_network.epsilon)
        logger.record("train/learning_rate", learning_rate)
        steps.dump(logger)


class Player:
    """What evaluate() plays: argmax Q, or epsilon-greedy at end_e when sampled."""

    def __init__(self, q_network: QNetwork, space: gym.Space, end_e: float) -> None:
        self.q_network = q_network
        self.space = space
        self.end_e = end_e

    def predict(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        state: tuple[np.ndarray, ...] | None = None,  # noqa: ARG002 - SB3's PolicyPredictor
        episode_start: np.ndarray | None = None,  # noqa: ARG002 - SB3's PolicyPredictor
        deterministic: bool = False,
    ) -> tuple[np.ndarray, None]:
        device = next(self.q_network.parameters()).device
        x = torch.as_tensor(ppo.observe(observation, self.space, None), device=device)
        self.q_network.epsilon = 0.0 if deterministic else self.end_e
        with torch.no_grad():
            action = self.q_network.get_action_and_value(x)[0]
        return action.cpu().numpy(), None


def saver(config: TrainConfig, eval_env: VecEnv) -> ppo.Save:
    assert config.pqn is not None
    end_e = config.pqn.end_e

    def save(q_network: nn.Module, path: Path, step: int) -> PolicyPredictor:  # noqa: ARG001
        assert isinstance(q_network, QNetwork)
        torch.save(q_network.state_dict(), path.with_suffix(".pt"))
        return Player(q_network, eval_env.observation_space, end_e)

    return save


def train(config: TrainConfig) -> Path:
    if config.pqn is None:
        raise ValueError("A PQN run needs config.pqn: try the pqn_cartpole preset")
    if config.second_value_head:
        raise ValueError("PQN has no second value head; RND trains on algoarcade.training.ppo")
    return ppo.run(config, "pqn", "pqn", learn, saver)


def main() -> None:
    print(train(tyro.extras.overridable_config_cli(PRESETS)))  # noqa: T201


# AsyncVectorEnv with spawn re-imports this module in every worker.
if __name__ == "__main__":
    main()
