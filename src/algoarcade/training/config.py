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
    norm_obs: bool
    norm_reward: bool
    """VecNormalize's switches. The plan normalizes observations and never the reward."""
    seed: int = 0
    eval_episodes: int = 10
    runs_dir: Path = Path("data/runs")
    track: bool = False
    """Send SB3's scalars to Weights & Biases."""
    start_state: str = ""
    """With env_id firered: a manifest entry with a step budget, such as seg_01_house."""
    reward_weights: dict[str, float] = dataclasses.field(default_factory=dict)
    """With env_id fake or firered: component name to weight."""
    step_log_every: int = 4
    """With env_id fake or firered: log every Nth decision of each episode to the steps table."""


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
    norm_obs=True,
    norm_reward=False,
)

# TRN-05's first FireRed runs. Explore is the only reward: the other default components read
# story_flags, party_level_sum, badge_count and milestone_count, which FireRedState gains in
# ENV-08 and ENV-13. 1M steps and 3 eval episodes are guesses; a deterministic policy from an
# indoor state replays near-identical episodes, so more eval episodes add little.
FIRERED = dataclasses.replace(
    BASELINE,
    env_id="firered",
    start_state=tyro.MISSING,
    total_timesteps=1_000_000,
    eval_episodes=3,
    reward_weights={"explore": 0.02},
)

# The control tasks end within 500 steps, so they do not need the baseline's long horizon. These
# are rl-baselines3-zoo's PPO values (hyperparams/ppo.yml at eec15dc), with constant rates in
# place of CartPole's linear schedules. For Acrobot the zoo also normalizes observations and
# trains for 1M steps; we do neither. Both step budgets are guesses checked on seed 0.
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
    norm_obs=False,
    norm_reward=False,
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
    norm_obs=False,
    norm_reward=False,
)

PRESETS = {
    "baseline": ("The project PPO baseline. Pass --env-id.", BASELINE),
    "firered": ("FireRed on mGBA, explore reward. Pass --start-state.", FIRERED),
    "cartpole": ("CartPole-v1 with rl-baselines3-zoo values.", CARTPOLE),
    "acrobot": ("Acrobot-v1 with rl-baselines3-zoo values.", ACROBOT),
}
