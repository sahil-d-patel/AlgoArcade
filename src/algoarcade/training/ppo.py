"""PPO we own, derived from CleanRL's ppo.py (TRN-07).

Source: https://github.com/vwxyzjn/cleanrl/blob/fe8d8a03c41a7ef5b523e2e354bd01c363e786bb/cleanrl/ppo.py

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

- Gymnasium's AsyncVectorEnv with spawn and same-step autoreset. CleanRL's default next-step
  autoreset stores a transition from the last observation to the reset one after every episode.
- Truncation bootstraps from V(info["final_obs"]); only termination ends the return. CleanRL
  merges the two, and every episode here ends on a step budget.
- Hyperparameters come from TrainConfig, the presets the SB3 runner trains. The update is
  Stable-Baselines3's PPO, so the parity gate compares like with like: constant learning rate,
  minibatches of ppo.batch_size, no value clipping, and a value loss of vf_coef times the mean
  squared error, without CleanRL's extra factor of 0.5.
- Observations are normalised with VecNormalize's arithmetic. Checkpoints are SB3 PPO zips with a
  VecNormalize pickle beside them, evaluation runs through the runner's evaluate(), and the
  Parquet rows through RunLog, so a run directory looks the same from either trainer.
- An optional second value head (config.second_value_head), which RND trains on the intrinsic
  return.

RND (TRN-10) follows CleanRL's ppo_rnd_envpool.py at the same commit, under the same license:
https://github.com/vwxyzjn/cleanrl/blob/fe8d8a03c41a7ef5b523e2e354bd01c363e786bb/cleanrl/ppo_rnd_envpool.py
It runs only when config.int_coef is not 0; at 0 the trainer is TRN-07's. What we changed:

- The input is a slice of RAM from each step's info (rnd_fields), not the screen, so the
  networks are MLPs 64 wide in place of the Atari convolutions.
- A column that has never varied whitens to 0: the input statistics start from no samples and
  the divisor carries VecNormalize's epsilon. CleanRL divides by the bare standard deviation.
- The forward filter discounts each env's novelty over time, as OpenAI's RND does (rff_int over
  buf_rews_int.T in its ppo_agent.py). CleanRL's buffer is steps by envs, so its .T runs the
  filter across envs instead.
- A rollout's inputs join the statistics before its novelty is measured, and the predictor
  trains on those same whitened inputs, of the states the actions reached. CleanRL measures with
  the previous statistics and trains on the observations the policy acted from.
- The predictor trains on every sample. CleanRL's update proportion of 0.25 is for 128 envs.
- The extrinsic advantage keeps weight 1 (CleanRL's ext_coef is 2), so int_coef is the ratio.
  Both value losses drop CleanRL's factor of 0.5, as the extrinsic one does above.
- The random-policy warmup plays one rollout (CleanRL plays 50) and comes before the seeded
  reset that starts training. CleanRL trains on from where the warmup left the envs, from a
  stale observation.

    uv run python -m algoarcade.training.ppo cartpole --seed 0
    uv run python -m algoarcade.training.ppo firered --start-state seg_02_pallet --seed 0
"""

import time
import warnings
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import tyro
from gymnasium import spaces
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.vec_env import VecNormalize
from torch import nn
from torch.distributions.categorical import Categorical
from torch.nn import functional as F

from algoarcade.games.firered.milestones import MILESTONES
from algoarcade.training import runner
from algoarcade.training.config import PRESETS, PPOConfig, TrainConfig
from algoarcade.training.parquet import RunLog

# SB3's ActorCriticPolicy builds Adam with this eps, as CleanRL does.
ADAM_EPS = 1e-5
# VecNormalize's defaults.
CLIP_OBS = 10.0
NORM_EPSILON = 1e-8
# RND's input clip is CleanRL's. The width is a guess.
RND_CLIP = 5.0
RND_WIDTH = 64
# NUM_BADGES, include/constants/flags.h in pret/pokefirered.
BADGES = 8
# SB3 averages episode returns over its last 100 episodes for rollout/ep_rew_mean.
EPISODE_WINDOW = 100
# Our parameter names to SB3's ActorCriticPolicy's. SB3 has one value head: our first.
SB3_NAMES = {
    "actor.0": "mlp_extractor.policy_net.0",
    "actor.2": "mlp_extractor.policy_net.2",
    "actor.4": "action_net",
    "critic.0": "mlp_extractor.value_net.0",
    "critic.2": "mlp_extractor.value_net.2",
    "value": "value_net",
}


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """CleanRL's network, which is also SB3's default policy: separate 64-64 tanh trunks for the
    actor and the critic. The critic has one value output, or two with the second head."""

    def __init__(self, obs_dim: int, n_actions: int, second_value_head: bool) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, n_actions), std=0.01),
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )
        self.value = layer_init(nn.Linear(64, 2 if second_value_head else 1), std=1.0)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.value(self.critic(x))

    def get_action_and_value(
        self, x: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        probs = Categorical(logits=self.actor(x))
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.get_value(x)


def sb3_state_dict(agent: Agent) -> dict[str, torch.Tensor]:
    state = {}
    for name, tensor in agent.state_dict().items():
        module, param = name.rsplit(".", 1)
        state[f"{SB3_NAMES[module]}.{param}"] = tensor[:1] if module == "value" else tensor
    return state


@dataclass
class Rollout:
    """n_steps x n_envs. `obs` is what the policy saw. `final_values` is V(final_obs) where the
    step budget cut an episode, and 0 everywhere else. `int_values` is the second head's value,
    or 0 without it."""

    obs: torch.Tensor
    actions: torch.Tensor
    logprobs: torch.Tensor
    rewards: torch.Tensor
    values: torch.Tensor
    dones: torch.Tensor
    final_values: torch.Tensor
    int_values: torch.Tensor


def make_envs(make: Callable[[], gym.Env], n_envs: int) -> gym.vector.AsyncVectorEnv:
    return gym.vector.AsyncVectorEnv(
        [make] * n_envs, context="spawn", autoreset_mode=gym.vector.AutoresetMode.SAME_STEP
    )


def by_key(obs: Any, space: gym.Space) -> dict[str, Any]:
    """An observation as {key: array} in the space's key order. A Box has the one key ""."""
    return {k: obs[k] for k in space.spaces} if isinstance(space, spaces.Dict) else {"": obs}


def observe(
    obs: Any, space: gym.Space, obs_rms: dict[str, RunningMeanStd] | None, update: bool = False
) -> np.ndarray:
    """A batch of observations as the policy's input. With obs_rms, each key is normalised as
    VecNormalize does, after adding the batch to the statistics if `update`. Then the keys are
    flattened and joined in order, as SB3's CombinedExtractor does."""
    parts = []
    for key, x in by_key(obs, space).items():
        if obs_rms is not None:
            if update:
                obs_rms[key].update(x)
            x = np.clip(
                (x - obs_rms[key].mean) / np.sqrt(obs_rms[key].var + NORM_EPSILON),
                -CLIP_OBS,
                CLIP_OBS,
            )
        parts.append(x.reshape(len(x), -1))
    return np.concatenate(parts, axis=1, dtype=np.float32)


def collect(  # noqa: PLR0913 - CleanRL's rollout loop, with its state passed in
    envs: gym.vector.VectorEnv,
    agent: Agent,
    obs_rms: dict[str, RunningMeanStd] | None,
    obs: np.ndarray,
    n_steps: int,
    after_step: Callable[[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]], None] | None = None,
) -> tuple[Rollout, np.ndarray]:
    """Step every env n_steps times from `obs`, the policy's input. Returns the rollout and the
    next input. `after_step(actions, rewards, dones, infos)` runs after each step."""
    space = envs.single_observation_space
    device = next(agent.parameters()).device
    shape = (n_steps, envs.num_envs)
    rollout = Rollout(
        obs=torch.zeros((*shape, obs.shape[1]), device=device),
        actions=torch.zeros(shape, dtype=torch.long, device=device),
        logprobs=torch.zeros(shape, device=device),
        rewards=torch.zeros(shape, device=device),
        values=torch.zeros(shape, device=device),
        dones=torch.zeros(shape, dtype=torch.bool, device=device),
        final_values=torch.zeros(shape, device=device),
        int_values=torch.zeros(shape, device=device),
    )
    for t in range(n_steps):
        x = torch.as_tensor(obs, device=device)
        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(x)
        actions = action.cpu().numpy()
        raw, reward, terminated, truncated, infos = envs.step(actions)
        obs = observe(raw, space, obs_rms, update=True)
        dones = terminated | truncated
        rollout.obs[t] = x
        rollout.actions[t] = action
        rollout.logprobs[t] = logprob
        rollout.rewards[t] = torch.as_tensor(reward, device=device)
        rollout.values[t] = value[:, 0]
        if value.shape[1] > 1:
            rollout.int_values[t] = value[:, 1]
        rollout.dones[t] = torch.as_tensor(dones, device=device)
        # Under same-step autoreset `obs` is already the reset observation for these envs; the
        # observation the step budget cut at is in final_obs. A step both terminated and
        # truncated is a termination.
        cut = np.flatnonzero(truncated & ~terminated)
        if len(cut):
            finals = [infos["final_obs"][i] for i in cut]
            batch = (
                {k: np.stack([f[k] for f in finals]) for k in space.spaces}
                if isinstance(space, spaces.Dict)
                else np.stack(finals)
            )
            final = observe(batch, space, obs_rms)
            with torch.no_grad():
                final_values = agent.get_value(torch.as_tensor(final, device=device))
            rollout.final_values[t, cut] = final_values[:, 0]
        if after_step is not None:
            after_step(actions, reward, dones, infos)
    return rollout, obs


def gae(
    rollout: Rollout, last_value: torch.Tensor, gamma: float, gae_lambda: float
) -> torch.Tensor:
    """Generalised advantage estimates. After a done, the next value is final_values: V(final_obs)
    at a truncation, 0 at a termination, and nothing flows back across it."""
    advantages = torch.zeros_like(rollout.rewards)
    carry = torch.zeros_like(last_value)
    for t in reversed(range(len(rollout.rewards))):
        next_value = last_value if t == len(rollout.rewards) - 1 else rollout.values[t + 1]
        next_value = torch.where(rollout.dones[t], rollout.final_values[t], next_value)
        delta = rollout.rewards[t] + gamma * next_value - rollout.values[t]
        carry = delta + gamma * gae_lambda * ~rollout.dones[t] * carry
        advantages[t] = carry
    return advantages


def intrinsic_gae(
    rollout: Rollout,
    rewards: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> torch.Tensor:
    """GAE on the intrinsic return, which ignores episode ends: novelty found after a reset still
    counts toward the steps before it, and the value after a done is the second head's value of
    the reset observation."""
    stream = replace(
        rollout, rewards=rewards, values=rollout.int_values, dones=torch.zeros_like(rollout.dones)
    )
    return gae(stream, last_value, gamma, gae_lambda)


def rnd_fields(info: Mapping[str, Any]) -> dict[str, list[int]]:
    """RND's input from a step's info, by field: the tile, a bit per badge, a bit per milestone
    reached this episode and the party level sum. The badge byte and the level sum are the values
    two consecutive reads agreed on, and milestone_count is the env's episode best, so a torn read
    while a save block moves is not novel. Never a counter, play time or RNG: the target network
    is frozen, so an input that never repeats stays novel forever."""
    map_group, map_num, x, y = (0,) * (4 - len(info["location"])) + tuple(info["location"])
    return {
        "map_group": [map_group],
        "map_num": [map_num],
        "x": [x],
        "y": [y],
        "badge_bits": [info["badge_flags"] >> bit & 1 for bit in range(BADGES)],
        "milestone_bits": [int(info["milestone_count"] > i) for i in range(len(MILESTONES))],
        "party_level_sum": [info["party_level_sum"]],
    }


def rnd_input(infos: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """One row per env: rnd_fields' columns in order."""
    return np.array(
        [[v for column in rnd_fields(info).values() for v in column] for info in infos],
        dtype=np.float32,
    )


def whiten(x: np.ndarray, rms: RunningMeanStd) -> np.ndarray:
    """RND's input standardised column by column, clipped to +-RND_CLIP."""
    standard = (x - rms.mean) / np.sqrt(rms.var + NORM_EPSILON)
    return np.clip(standard, -RND_CLIP, RND_CLIP).astype(np.float32)


class RND(nn.Module):
    """Random network distillation: an input's novelty is the error of a predictor trained to
    match a frozen, randomly initialised target on it. Also holds the input statistics, started
    from `warm`, the intrinsic return's forward filter, both carried across rollouts, and the
    current rollout's inputs."""

    def __init__(self, warm: np.ndarray, gamma: float) -> None:
        super().__init__()
        n_inputs = warm.shape[1]
        self.target = nn.Sequential(
            layer_init(nn.Linear(n_inputs, RND_WIDTH)),
            nn.LeakyReLU(),
            layer_init(nn.Linear(RND_WIDTH, RND_WIDTH)),
            nn.LeakyReLU(),
            layer_init(nn.Linear(RND_WIDTH, RND_WIDTH)),
        )
        # As CleanRL's: the target's layers and two more, so it can fit the target.
        self.predictor = nn.Sequential(
            layer_init(nn.Linear(n_inputs, RND_WIDTH)),
            nn.LeakyReLU(),
            layer_init(nn.Linear(RND_WIDTH, RND_WIDTH)),
            nn.LeakyReLU(),
            layer_init(nn.Linear(RND_WIDTH, RND_WIDTH)),
            nn.ReLU(),
            layer_init(nn.Linear(RND_WIDTH, RND_WIDTH)),
            nn.ReLU(),
            layer_init(nn.Linear(RND_WIDTH, RND_WIDTH)),
        )
        self.target.requires_grad_(False)
        # No prior sample, so a column that has never varied has variance 0 and whitens to 0.
        self.input_rms = RunningMeanStd(epsilon=0.0, shape=(n_inputs,))
        self.input_rms.update(warm)
        self.return_rms = RunningMeanStd()
        self.gamma = gamma
        self.returns: np.ndarray | None = None
        self.inputs: list[np.ndarray] = []

    def novelty(self, x: torch.Tensor) -> torch.Tensor:
        """The squared error per input, averaged over the target's outputs."""
        return (self.predictor(x) - self.target(x)).pow(2).mean(-1)

    def scale(self, novelty: np.ndarray) -> np.ndarray:
        """A rollout's novelty, n_steps x n_envs, divided by the running std of the intrinsic
        return: each env's novelty discounted by gamma over time, never reset by a done."""
        returns = np.empty_like(novelty)
        for t, row in enumerate(novelty):
            self.returns = row if self.returns is None else self.returns * self.gamma + row
            returns[t] = self.returns
        self.return_rms.update(returns.ravel())
        return novelty / np.sqrt(self.return_rms.var)


@dataclass
class Intrinsic:
    """RND's part of an update, n_steps x n_envs: the intrinsic advantages and their weight, and
    the whitened inputs the predictor trains on."""

    rnd: RND
    coef: float
    advantages: torch.Tensor
    inputs: torch.Tensor


def intrinsic_terms(
    rnd: RND, rollout: Rollout, last_value: torch.Tensor, coef: float, gae_lambda: float
) -> Intrinsic:
    """Takes the rollout's inputs from rnd.inputs. They join the statistics before they are
    whitened, so the predictor trains on what the reward measured."""
    inputs = np.stack(rnd.inputs)
    rnd.inputs.clear()
    rnd.input_rms.update(inputs.reshape(-1, inputs.shape[-1]))
    x = torch.as_tensor(whiten(inputs, rnd.input_rms), device=last_value.device)
    with torch.no_grad():
        novelty = rnd.novelty(x).cpu().numpy()
    rewards = torch.as_tensor(rnd.scale(novelty), dtype=torch.float32, device=last_value.device)
    advantages = intrinsic_gae(rollout, rewards, last_value, rnd.gamma, gae_lambda)
    return Intrinsic(rnd, coef, advantages, x)


def step_infos(
    infos: dict[str, Any], current: list[dict[str, Any]], dones: np.ndarray
) -> list[dict[str, Any]]:
    """Each env's info for the step it took. For an env that finished, `infos` holds its reset
    info, unbatched as `current`, and final_info its last step's."""
    final = unbatch(infos["final_info"], len(current)) if "final_info" in infos else current
    return [final[i] if done else current[i] for i, done in enumerate(dones)]


def warm_up(
    envs: gym.vector.VectorEnv, seed: int, n_steps: int, rng: np.random.Generator
) -> np.ndarray:
    """RND's input on n_steps random-policy steps per env from a seeded reset, to start its
    statistics as CleanRL does. Training resets the envs after this."""
    envs.reset(seed=seed)
    n = envs.num_envs
    action_space = envs.single_action_space
    assert isinstance(action_space, spaces.Discrete)
    inputs = []
    for _ in range(n_steps):
        _, _, terminated, truncated, infos = envs.step(rng.integers(int(action_space.n), size=n))
        inputs.append(rnd_input(step_infos(infos, unbatch(infos, n), terminated | truncated)))
    return np.concatenate(inputs)


def update(  # noqa: PLR0913 - CleanRL's update loop, with its state passed in
    agent: Agent,
    optimizer: torch.optim.Optimizer,
    rollout: Rollout,
    advantages: torch.Tensor,
    ppo: PPOConfig,
    rng: np.random.Generator,
    intrinsic: Intrinsic | None = None,
) -> dict[str, float]:
    """ppo.n_epochs passes over the rollout in shuffled minibatches, as SB3's PPO.train does.
    Returns the scalars SB3 logs, under its names. With RND the policy follows the advantages
    plus intrinsic.coef times the intrinsic ones, the second head learns the intrinsic return and
    the predictor learns the target on the rollout's inputs, all in one loss."""
    returns = (advantages + rollout.values).flatten()
    b_obs = rollout.obs.flatten(0, 1)
    b_actions = rollout.actions.flatten()
    b_logprobs = rollout.logprobs.flatten()
    b_advantages = advantages.flatten()
    parameters = list(agent.parameters())
    logs: dict[str, list[float]] = {
        "policy_gradient_loss": [],
        "value_loss": [],
        "entropy_loss": [],
        "approx_kl": [],
        "clip_fraction": [],
    }
    if intrinsic is not None:
        b_advantages = b_advantages + intrinsic.coef * intrinsic.advantages.flatten()
        parameters += intrinsic.rnd.predictor.parameters()
        logs |= {"intrinsic_value_loss": [], "rnd_loss": []}
    for _ in range(ppo.n_epochs):
        order = torch.as_tensor(rng.permutation(len(b_obs)), device=b_obs.device)
        for start in range(0, len(b_obs), ppo.batch_size):
            mb = order[start : start + ppo.batch_size]
            _, logprob, entropy, value = agent.get_action_and_value(b_obs[mb], b_actions[mb])
            logratio = logprob - b_logprobs[mb]
            ratio = logratio.exp()
            mb_advantages = b_advantages[mb]
            if len(mb) > 1:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                    mb_advantages.std() + 1e-8
                )
            pg_loss = torch.max(
                -mb_advantages * ratio,
                -mb_advantages * torch.clamp(ratio, 1 - ppo.clip_range, 1 + ppo.clip_range),
            ).mean()
            v_loss = F.mse_loss(value[:, 0], returns[mb])
            entropy_loss = -entropy.mean()
            loss = pg_loss + ppo.ent_coef * entropy_loss + ppo.vf_coef * v_loss
            if intrinsic is not None:
                int_returns = (intrinsic.advantages + rollout.int_values).flatten()[mb]
                int_v_loss = F.mse_loss(value[:, 1], int_returns)
                rnd_loss = intrinsic.rnd.novelty(intrinsic.inputs.flatten(0, 1)[mb]).mean()
                loss = loss + ppo.vf_coef * int_v_loss + rnd_loss
                logs["intrinsic_value_loss"].append(int_v_loss.item())
                logs["rnd_loss"].append(rnd_loss.item())

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, ppo.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                logs["approx_kl"].append(((ratio - 1) - logratio).mean().item())
                clipped = (ratio - 1).abs() > ppo.clip_range
                logs["clip_fraction"].append(clipped.float().mean().item())
            logs["policy_gradient_loss"].append(pg_loss.item())
            logs["value_loss"].append(v_loss.item())
            logs["entropy_loss"].append(entropy_loss.item())

    y_true = returns.cpu().numpy()
    var_y = np.var(y_true)
    y_pred = rollout.values.flatten().cpu().numpy()
    explained = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
    return {f"train/{k}": float(np.mean(v)) for k, v in logs.items()} | {
        "train/explained_variance": float(explained)
    }


def unbatch(infos: dict[str, Any], n_envs: int) -> list[dict[str, Any]]:
    """Gymnasium's batched info as one dict per env, without final_obs and final_info."""
    return [
        {
            k: v[i].item() if isinstance(v[i], np.generic) else v[i]
            for k, v in infos.items()
            if not k.startswith("_") and k not in ("final_obs", "final_info") and infos[f"_{k}"][i]
        }
        for i in range(n_envs)
    ]


class Progress:
    """The training episodes as SB3 reports them in progress.csv: mean return and length over
    the last 100 episodes, and Super Mario Land's continues summed over the envs."""

    def __init__(self, n_envs: int) -> None:
        self.returns = np.zeros(n_envs)
        self.lengths = np.zeros(n_envs, dtype=int)
        self.episodes: deque[tuple[float, int]] = deque(maxlen=EPISODE_WINDOW)
        self.continues: dict[int, int] = {}

    def step(self, rewards: np.ndarray, dones: np.ndarray, infos: list[dict[str, Any]]) -> None:
        self.returns += rewards
        self.lengths += 1
        for i in np.flatnonzero(dones):
            self.episodes.append((float(self.returns[i]), int(self.lengths[i])))
            self.returns[i], self.lengths[i] = 0.0, 0
        for i, info in enumerate(infos):
            if "continues" in info:
                self.continues[i] = info["continues"]

    def record(self, logger: Any) -> None:
        if self.episodes:
            returns, lengths = zip(*self.episodes, strict=True)
            logger.record("rollout/ep_rew_mean", float(np.mean(returns)))
            logger.record("rollout/ep_len_mean", float(np.mean(lengths)))
        if self.continues:
            logger.record("rollout/continues", sum(self.continues.values()))


def learn(
    config: TrainConfig,
    envs: gym.vector.VectorEnv,
    checkpoint: Callable[[Agent, dict[str, RunningMeanStd] | None, int, bool], None],
    log: RunLog | None,
    logger: Any,
) -> None:
    """Train until config.total_timesteps, rounded up to a whole rollout as SB3 does. A
    checkpoint falls every checkpoint_every steps inside a rollout, before its update, as the SB3
    runner's does, and once more after the last update. It is evaluated when its scheduled step
    is a multiple of eval_every, and always at the end."""
    ppo = config.ppo
    n = config.n_envs
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    space = envs.single_observation_space
    obs_rms = (
        {k: RunningMeanStd(shape=s.shape) for k, s in by_key(space, space).items()}
        if config.norm_obs
        else None
    )
    # One rollout of random-policy steps, a guess; CleanRL plays 50 at 128 envs.
    warm = warm_up(envs, config.seed, ppo.n_steps, rng) if config.int_coef else None
    raw, infos = envs.reset(seed=config.seed)
    obs = observe(raw, space, obs_rms, update=True)
    if log is not None:
        log.begin(range(n), unbatch(infos, n))
    action_space = envs.single_action_space
    assert isinstance(action_space, spaces.Discrete)
    agent = Agent(obs.shape[1], int(action_space.n), config.second_value_head).to(device)
    # After the agent, so the agent starts from the weights a plain run with this seed has.
    rnd = None if warm is None else RND(warm, config.int_gamma).to(device)
    optimizer = torch.optim.Adam(
        [*agent.parameters(), *(() if rnd is None else rnd.predictor.parameters())],
        lr=ppo.learning_rate,
        eps=ADAM_EPS,
    )

    steps = 0
    next_checkpoint = config.checkpoint_every
    progress = Progress(n)

    def after_step(
        actions: np.ndarray, rewards: np.ndarray, dones: np.ndarray, infos: dict[str, Any]
    ) -> None:
        nonlocal steps, next_checkpoint
        steps += n
        current = unbatch(infos, n)
        taken = step_infos(infos, current, dones)
        if log is not None:
            log.step(actions, rewards, dones, taken)
            log.begin(np.flatnonzero(dones), current)
        progress.step(rewards, dones, taken)
        if rnd is not None:
            rnd.inputs.append(rnd_input(taken))
        if next_checkpoint <= steps < config.total_timesteps:
            # The scheduled step, not `steps`, which can overshoot it by up to n_envs - 1.
            due = next_checkpoint % config.eval_every == 0
            next_checkpoint += config.checkpoint_every
            checkpoint(agent, obs_rms, steps, due)

    start = time.time()
    while steps < config.total_timesteps:
        rollout, obs = collect(envs, agent, obs_rms, obs, ppo.n_steps, after_step)
        with torch.no_grad():
            last_value = agent.get_value(torch.as_tensor(obs, device=device))
        advantages = gae(rollout, last_value[:, 0], ppo.gamma, ppo.gae_lambda)
        intrinsic = (
            None
            if rnd is None
            else intrinsic_terms(rnd, rollout, last_value[:, 1], config.int_coef, ppo.gae_lambda)
        )
        for key, value in update(
            agent, optimizer, rollout, advantages, ppo, rng, intrinsic
        ).items():
            logger.record(key, value)
        if steps >= config.total_timesteps:
            checkpoint(agent, obs_rms, steps, True)
        progress.record(logger)
        logger.record("time/fps", int(steps / (time.time() - start)))
        logger.record("time/total_timesteps", steps)
        logger.dump(steps)


def train(config: TrainConfig) -> Path:
    if config.norm_reward:
        raise ValueError("The owned trainer does not normalise rewards; the plan never does")
    if config.archive:
        raise ValueError("Archive resets (ENV-09) run on the SB3 runner only")
    run_dir, env_fn, backend, rom_sha1, commit = runner.prepare(config, "owned")
    game = config.env_id in runner.GAMES
    # Wall clock, as the schema means it: perf_counter stops while macOS sleeps.
    start = time.time()
    envs = make_envs(env_fn if game else partial(gym.make, config.env_id), config.n_envs)
    log = RunLog(config.n_envs, config, run_dir, start) if game else None
    eval_env = runner.make_eval_env(config, env_fn, run_dir, start, log)
    # The SB3 model only carries our weights to disk and through evaluation, so its one-env
    # rollout buffer, smaller than a minibatch, is never used. It seeds the global RNGs, so it
    # is built before learn() seeds ours.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="You have specified a mini-batch size")
        model = runner.build_model(config, eval_env)
    logger = runner.make_logger(config, run_dir)

    def checkpoint(
        agent: Agent, obs_rms: dict[str, RunningMeanStd] | None, step: int, evaluate_it: bool
    ) -> None:
        path = run_dir / "checkpoints" / f"step_{step}"
        model.policy.load_state_dict(sb3_state_dict(agent))
        model.num_timesteps = step
        model.save(path.with_suffix(".zip"))
        if obs_rms is not None:
            assert isinstance(eval_env, VecNormalize)
            dict_obs = isinstance(eval_env.observation_space, spaces.Dict)
            eval_env.obs_rms = deepcopy(obs_rms if dict_obs else obs_rms[""])
            eval_env.save(str(path.with_suffix(".pkl")))
        if evaluate_it:
            mean = runner.evaluate(model, eval_env, config, run_dir, step, start)
            logger.record("eval/mean_return", mean)

    fixed = runner.run_row(config, run_dir.name.removeprefix("run_id="), backend, rom_sha1, commit)
    fixed["trainer"] = "owned"
    if log is not None:
        runner.write_run_row(
            run_dir, fixed | {"status": "running", "env_steps": 0, "wall_clock_s": 0.0}
        )
    status = "crashed"
    try:
        learn(config, envs, checkpoint, log, logger)
        status = "finished"
    except EOFError as error:
        runner.stop_workers(eval_env)
        raise RuntimeError("The eval worker died; its traceback is above") from error
    finally:
        if log is not None:
            elapsed = time.time() - start
            end = {"status": status, "env_steps": log.env_steps, "wall_clock_s": elapsed}
            runner.write_run_row(run_dir, fixed | end)
        logger.close()
        envs.close()
        if log is not None:
            log.close()
        eval_env.close()
    return run_dir


def main() -> None:
    print(train(tyro.extras.overridable_config_cli(PRESETS)))  # noqa: T201


# AsyncVectorEnv with spawn re-imports this module in every worker.
if __name__ == "__main__":
    main()
