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
- An optional second value head (config.second_value_head). Nothing trains it until TRN-10 gives
  it an intrinsic return.

    uv run python -m algoarcade.training.ppo cartpole --seed 0
    uv run python -m algoarcade.training.ppo firered --start-state seg_02_pallet --seed 0
"""

import time
import warnings
from collections import deque
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
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

from algoarcade.training import runner
from algoarcade.training.config import PRESETS, PPOConfig, TrainConfig
from algoarcade.training.parquet import RunLog

# SB3's ActorCriticPolicy builds Adam with this eps, as CleanRL does.
ADAM_EPS = 1e-5
# VecNormalize's defaults.
CLIP_OBS = 10.0
NORM_EPSILON = 1e-8
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
    step budget cut an episode, and 0 everywhere else."""

    obs: torch.Tensor
    actions: torch.Tensor
    logprobs: torch.Tensor
    rewards: torch.Tensor
    values: torch.Tensor
    dones: torch.Tensor
    final_values: torch.Tensor


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


def update(  # noqa: PLR0913 - CleanRL's update loop, with its state passed in
    agent: Agent,
    optimizer: torch.optim.Optimizer,
    rollout: Rollout,
    advantages: torch.Tensor,
    ppo: PPOConfig,
    rng: np.random.Generator,
) -> dict[str, float]:
    """ppo.n_epochs passes over the rollout in shuffled minibatches, as SB3's PPO.train does.
    Returns the scalars SB3 logs, under its names."""
    returns = (advantages + rollout.values).flatten()
    b_obs = rollout.obs.flatten(0, 1)
    b_actions = rollout.actions.flatten()
    b_logprobs = rollout.logprobs.flatten()
    b_advantages = advantages.flatten()
    logs: dict[str, list[float]] = {
        "policy_gradient_loss": [],
        "value_loss": [],
        "entropy_loss": [],
        "approx_kl": [],
        "clip_fraction": [],
    }
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

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), ppo.max_grad_norm)
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
    checkpoint: Callable[[Agent, dict[str, RunningMeanStd] | None, int], None],
    log: RunLog | None,
    logger: Any,
) -> None:
    """Train until config.total_timesteps, rounded up to a whole rollout as SB3 does. A checkpoint falls every checkpoint_every steps inside a rollout, before
    its update, as the SB3 runner's does, and once more after the last update."""
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
    raw, infos = envs.reset(seed=config.seed)
    obs = observe(raw, space, obs_rms, update=True)
    if log is not None:
        log.begin(range(n), unbatch(infos, n))
    action_space = envs.single_action_space
    assert isinstance(action_space, spaces.Discrete)
    agent = Agent(obs.shape[1], int(action_space.n), config.second_value_head).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=ppo.learning_rate, eps=ADAM_EPS)

    steps = 0
    next_checkpoint = config.checkpoint_every
    progress = Progress(n)

    def after_step(
        actions: np.ndarray, rewards: np.ndarray, dones: np.ndarray, infos: dict[str, Any]
    ) -> None:
        nonlocal steps, next_checkpoint
        steps += n
        current = unbatch(infos, n)
        # For an env that finished, `infos` holds its reset info and final_info its last step's.
        final = unbatch(infos["final_info"], n) if "final_info" in infos else current
        step_infos = [final[i] if done else current[i] for i, done in enumerate(dones)]
        if log is not None:
            log.step(actions, rewards, dones, step_infos)
            log.begin(np.flatnonzero(dones), current)
        progress.step(rewards, dones, step_infos)
        if next_checkpoint <= steps < config.total_timesteps:
            next_checkpoint += config.checkpoint_every
            checkpoint(agent, obs_rms, steps)

    start = time.time()
    while steps < config.total_timesteps:
        rollout, obs = collect(envs, agent, obs_rms, obs, ppo.n_steps, after_step)
        with torch.no_grad():
            last_value = agent.get_value(torch.as_tensor(obs, device=device))[:, 0]
        advantages = gae(rollout, last_value, ppo.gamma, ppo.gae_lambda)
        for key, value in update(agent, optimizer, rollout, advantages, ppo, rng).items():
            logger.record(key, value)
        if steps >= config.total_timesteps:
            checkpoint(agent, obs_rms, steps)
        progress.record(logger)
        logger.record("time/fps", int(steps / (time.time() - start)))
        logger.record("time/total_timesteps", steps)
        logger.dump(steps)


def train(config: TrainConfig) -> Path:
    if config.norm_reward:
        raise ValueError("The owned trainer does not normalise rewards; the plan never does")
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

    def checkpoint(agent: Agent, obs_rms: dict[str, RunningMeanStd] | None, step: int) -> None:
        path = run_dir / "checkpoints" / f"step_{step}"
        model.policy.load_state_dict(sb3_state_dict(agent))
        model.num_timesteps = step
        model.save(path.with_suffix(".zip"))
        if obs_rms is not None:
            assert isinstance(eval_env, VecNormalize)
            dict_obs = isinstance(eval_env.observation_space, spaces.Dict)
            eval_env.obs_rms = deepcopy(obs_rms if dict_obs else obs_rms[""])
            eval_env.save(str(path.with_suffix(".pkl")))
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
