"""Training run configs. Contract: docs/technical-plan.md, Training.

No hyperparameter has a default, so neither a library default nor a forgotten field can slip in.
"""

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

import tyro


@dataclass(frozen=True)
class PPOConfig:
    """Keyword arguments for Stable-Baselines3 PPO, passed through by name."""

    learning_rate: float
    n_steps: int
    """Steps per environment per update."""
    batch_size: int
    n_epochs: int
    gamma: float
    gae_lambda: float
    clip_range: float
    ent_coef: float


@dataclass(frozen=True)
class TrainConfig:
    env_id: str
    """Any registered Gymnasium env id."""
    total_timesteps: int
    """Total env steps across all environments. PPO rounds up to a whole rollout."""
    n_envs: int
    checkpoint_every: int
    """Env steps between checkpoints. Each checkpoint is evaluated."""
    ppo: PPOConfig
    seed: int = 0
    eval_episodes: int = 10
    runs_dir: Path = Path("data/runs")
    track: bool = False
    """Send SB3's scalars to Weights & Biases."""


def save_config(config: TrainConfig, path: Path) -> None:
    path.write_text(json.dumps(dataclasses.asdict(config), indent=2, default=str) + "\n")


def load_config(path: Path) -> TrainConfig:
    raw = json.loads(path.read_text())
    raw["ppo"] = PPOConfig(**raw["ppo"])
    raw["runs_dir"] = Path(raw["runs_dir"])
    return TrainConfig(**raw)


# The project baseline from docs/technical-plan.md, "PPO starting point". batch_size is not in
# that table; 4,096 (8 minibatches per 32,768-step rollout) is a guess for TRN-05 to tune.
BASELINE = TrainConfig(
    env_id=tyro.MISSING,
    total_timesteps=15_000_000,
    n_envs=16,
    checkpoint_every=500_000,
    ppo=PPOConfig(
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=4096,
        n_epochs=3,
        gamma=0.997,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
    ),
)

# The control tasks pay every step and end at 500 steps, so their returns, not their horizons,
# set gamma. At 0.997 the value target for CartPole is about 1 / (1 - gamma) = 333, which the
# value head fits slowly. These values start from rl-baselines3-zoo's tuned PPO settings for
# each env, with constant rates in place of the zoo's linear schedules.
CARTPOLE = TrainConfig(
    env_id="CartPole-v1",
    total_timesteps=100_000,
    n_envs=8,
    checkpoint_every=10_000,
    ppo=PPOConfig(
        learning_rate=0.001,
        n_steps=32,
        batch_size=256,
        n_epochs=20,
        # A 50-step horizon: a pole that starts to fall is lost within a few dozen steps.
        gamma=0.98,
        gae_lambda=0.8,
        clip_range=0.2,
        ent_coef=0.0,
    ),
)

ACROBOT = TrainConfig(
    env_id="Acrobot-v1",
    total_timesteps=500_000,
    n_envs=16,
    checkpoint_every=50_000,
    ppo=PPOConfig(
        learning_rate=0.0003,
        n_steps=256,
        batch_size=64,
        n_epochs=4,
        # A 100-step horizon: the -100 threshold means swinging up in under 100 steps.
        gamma=0.99,
        gae_lambda=0.94,
        clip_range=0.2,
        ent_coef=0.0,
    ),
)

PRESETS = {
    "baseline": ("The project PPO baseline. Pass --env-id.", BASELINE),
    "cartpole": ("CartPole-v1, tuned for the control task.", CARTPOLE),
    "acrobot": ("Acrobot-v1, tuned for the control task.", ACROBOT),
}
