"""Play one episode of a checkpoint and write it as mp4, for the weekly video review.

    uv run python -m algoarcade.viz.render --checkpoint <run dir>/checkpoints/step_<n>.zip \\
        --seed 0 --out data/videos/<name>.mp4

`--grid 4` plays seeds `seed` to `seed + 3` side by side in one video. The game, start state and
reward weights come from the run's config.json, two folders above the checkpoint, and the frozen
normalisation statistics from step_<n>.pkl beside it. The policy acts deterministically on the
CPU from the manifest start state, reset with `seed`, so a checkpoint and seed always play the
same episode. Each decision is one video frame, the screen after it, with the step, the return,
the location, the badge count and one bar per reward component's weighted episode total.

`--eval-episode k` replays instead the k-th episode the runner evaluated at this checkpoint: the
eval seed, the run's eval_deterministic and the runner's eval_rng, with episodes 0 to k-1 played
first and not recorded. A sampled episode replays exactly only on the device the run trained on,
so this mode loads the model as training did (CUDA when present).
"""

import contextlib
from pathlib import Path
from typing import Any

import imageio_ffmpeg
import numpy as np
import tyro
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from algoarcade.training.config import load_config
from algoarcade.training.games import env_factory
from algoarcade.training.runner import EVAL_SEED_OFFSET, eval_rng

SCALE = 2
# 6x real time: a decision is 24 frames of a 60 fps game.
FPS = 15
LINE = 12
PAD = 4
# Where the bars start; the component name and total sit to the left.
LABEL_WIDTH = 100
BACKGROUND = (16, 16, 24)
TEXT = (232, 232, 232)
TRACK = (64, 64, 72)
GAIN = (88, 200, 104)
COST = (224, 88, 88)
FONT = ImageFont.load_default()


def frame(
    screen: NDArray[np.uint8],
    step: int,
    episode_return: float,
    info: dict[str, Any],
    totals: dict[str, float],
) -> NDArray[np.uint8]:
    """The screen scaled up by SCALE, over a panel with one bar per entry in `totals`. Each bar is
    its total's share of the largest total in the frame."""
    game = screen.repeat(SCALE, axis=0).repeat(SCALE, axis=1)
    height, width = game.shape[:2]
    panel = -(-(2 * PAD + (2 + len(totals)) * LINE) // 16) * 16
    image = Image.new("RGB", (width, height + panel), BACKGROUND)
    image.paste(Image.fromarray(game))
    draw = ImageDraw.Draw(image)
    top = height + PAD
    draw.text((PAD, top), f"step {step}  return {episode_return:.2f}", fill=TEXT, font=FONT)
    location = info["location"]
    badges = info["badge_count"]
    draw.text((PAD, top + LINE), f"{location}  badges {badges}", fill=TEXT, font=FONT)
    largest = max(abs(total) for total in totals.values()) or 1.0
    right = width - PAD
    for row, (name, total) in enumerate(totals.items(), start=2):
        y = top + row * LINE
        draw.text((PAD, y), f"{name} {total:+.2f}", fill=TEXT, font=FONT)
        draw.rectangle((LABEL_WIDTH, y + 2, right, y + LINE - 3), fill=TRACK)
        if total:
            end = LABEL_WIDTH + round(abs(total) / largest * (right - LABEL_WIDTH))
            colour = GAIN if total > 0 else COST
            draw.rectangle((LABEL_WIDTH, y + 2, end, y + LINE - 3), fill=colour)
    return np.asarray(image)


def render(  # noqa: C901 - the eval replay adds a branch and a loop
    checkpoint: Path, out: Path, seed: int = 0, grid: int = 1, eval_episode: int | None = None
) -> list[tuple[float, tuple[int, ...]]]:
    """Writes the video and returns each episode's return and final location."""
    config = load_config(checkpoint.parents[1] / "config.json")
    deterministic, rng, device = True, contextlib.nullcontext(), "cpu"
    if eval_episode is not None:
        if grid != 1:
            raise ValueError("--eval-episode replays one episode; leave --grid at 1")
        seed = config.seed + EVAL_SEED_OFFSET
        deterministic, rng, device = config.eval_deterministic, eval_rng(config), "auto"
    weights = config.reward_weights
    make, _, _ = env_factory(config.env_id, config.start_state, weights, render=True)
    envs = [make() for _ in range(grid)]
    model = PPO.load(checkpoint, device=device)
    normalize = None
    if config.norm_obs:
        # Only normalize_obs is called on it, which never updates the statistics.
        normalize = VecNormalize.load(
            str(checkpoint.with_suffix(".pkl")), DummyVecEnv([lambda: envs[0]])
        )

    def act(observation: Any) -> Any:
        policy_obs = normalize.normalize_obs(observation) if normalize else observation
        return model.predict(policy_obs, deterministic=deterministic)[0]

    obs = [env.reset(seed=seed + i)[0] for i, env in enumerate(envs)]
    returns = [0.0] * grid
    totals = [dict.fromkeys(weights, 0.0) for _ in envs]
    infos: list[dict[str, Any]] = [{} for _ in envs]
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    step, truncated = 0, False
    with rng:
        try:
            for _ in range(eval_episode or 0):
                done = False
                while not done:
                    obs[0], _, _, done, _ = envs[0].step(act(obs[0]))
                obs[0] = envs[0].reset()[0]
            # Every episode runs to the same step budget; AlgoArcadeEnv never terminates.
            while not truncated:
                step += 1
                panels = []
                for i, env in enumerate(envs):
                    obs[i], reward, _, truncated, infos[i] = env.step(act(obs[i]))
                    returns[i] += float(reward)
                    for name, weight in weights.items():
                        totals[i][name] += weight * infos[i][f"reward/{name}"]
                    screen = env.get_wrapper_attr("backend").screen()
                    panels.append(frame(screen, step, returns[i], infos[i], totals[i]))
                image = np.hstack(panels)
                if writer is None:
                    height, width = image.shape[:2]
                    writer = imageio_ffmpeg.write_frames(
                        str(out), (width, height), fps=FPS, codec="libx264", pix_fmt_out="yuv420p"
                    )
                    writer.send(None)
                writer.send(image.tobytes())
        finally:
            if writer is not None:
                writer.close()
            for env in envs:
                env.close()
    return [(ret, info["location"]) for ret, info in zip(returns, infos, strict=True)]


def main(
    checkpoint: Path, out: Path, seed: int = 0, grid: int = 1, eval_episode: int | None = None
) -> None:
    """Render a checkpoint's episode from seed `seed`, `grid` episodes side by side, or the
    checkpoint's evaluation episode `eval_episode`."""
    results = render(checkpoint, out, seed, grid, eval_episode)
    for i, (episode_return, location) in enumerate(results):
        name = f"seed {seed + i}" if eval_episode is None else f"eval episode {eval_episode}"
        print(f"{name}: return {episode_return:.6f}, final location {location}")  # noqa: T201
    print(out)  # noqa: T201


if __name__ == "__main__":
    tyro.cli(main)
