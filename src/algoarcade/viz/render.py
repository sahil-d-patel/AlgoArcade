"""Play one episode of a checkpoint and write it as mp4, for the weekly video review.

    uv run python -m algoarcade.viz.render --checkpoint <run dir>/checkpoints/step_<n>.zip \\
        --seed 0 --out data/videos/<name>.mp4

`--grid 4` plays seeds `seed` to `seed + 3` side by side in one video. The game, start state and
reward weights come from the run's config.json, two folders above the checkpoint, and the frozen
normalisation statistics from step_<n>.pkl beside it. The policy acts deterministically on the
CPU from the manifest start state, reset with `seed`, so a checkpoint and seed always play the
same episode. Frame 0 is the start state; each decision after it is one video frame, the screen
after it, with the run, the press, the return, the location, the badge count and one bar per
reward component's weighted episode total.

`compose` writes any cells side by side: `Policy` here, and `video_grid.Replay`, an LLM attempt
played again from its logged presses.
"""

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import tyro
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from algoarcade.training.config import load_config
from algoarcade.training.games import env_factory

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


@dataclass(frozen=True)
class Shot:
    """What a cell shows after a press: the screen as the backend returned it, the text under it
    and one bar per reward component's weighted episode total. Every shot of a cell has the same
    number of lines and totals, so the cell keeps one size."""

    screen: NDArray[np.uint8]
    lines: tuple[str, ...]
    totals: Mapping[str, float]


def frame(shot: Shot) -> NDArray[np.uint8]:
    """The screen scaled up by SCALE, over a panel with the lines and one bar per total. Each
    bar is its total's share of the largest total in the frame."""
    game = shot.screen.repeat(SCALE, axis=0).repeat(SCALE, axis=1)
    height, width = game.shape[:2]
    panel = -(-(2 * PAD + (len(shot.lines) + len(shot.totals)) * LINE) // 16) * 16
    image = Image.new("RGB", (width, height + panel), BACKGROUND)
    image.paste(Image.fromarray(game))
    draw = ImageDraw.Draw(image)
    top = height + PAD
    for row, line in enumerate(shot.lines):
        draw.text((PAD, top + row * LINE), line, fill=TEXT, font=FONT)
    largest = max((abs(total) for total in shot.totals.values()), default=0.0) or 1.0
    right = width - PAD
    for row, (name, total) in enumerate(shot.totals.items(), start=len(shot.lines)):
        y = top + row * LINE
        draw.text((PAD, y), f"{name} {total:+.2f}", fill=TEXT, font=FONT)
        draw.rectangle((LABEL_WIDTH, y + 2, right, y + LINE - 3), fill=TRACK)
        if total:
            end = LABEL_WIDTH + round(abs(total) / largest * (right - LABEL_WIDTH))
            colour = GAIN if total > 0 else COST
            draw.rectangle((LABEL_WIDTH, y + 2, end, y + LINE - 3), fill=colour)
    return np.asarray(image)


class Policy:
    """One episode of a checkpoint, a cell for `compose`. After it is played, `episode_return`,
    `location` and `goal_step` (the first press after which the start state's goal held) hold
    the outcome. With `until_goal` the cell ends at that press instead of the step budget."""

    def __init__(self, checkpoint: Path, seed: int = 0, until_goal: bool = False) -> None:
        self.checkpoint = checkpoint
        self.config = load_config(checkpoint.parents[1] / "config.json")
        self.seed = seed
        self.until_goal = until_goal
        run = checkpoint.parents[1].name.removeprefix("run_id=")
        self.label = f"{run} {checkpoint.stem} seed {seed}"
        self.episode_return = 0.0
        self.location: tuple[int, ...] = ()
        self.goal_step: int | None = None

    def __iter__(self) -> Iterator[Shot]:
        weights = self.config.reward_weights
        make, _, _ = env_factory(self.config.env_id, self.config.start_state, weights, render=True)
        env = make()
        model = PPO.load(self.checkpoint, device="cpu")
        normalize = None
        if self.config.norm_obs:
            # Only normalize_obs is called on it, which never updates the statistics.
            normalize = VecNormalize.load(
                str(self.checkpoint.with_suffix(".pkl")), DummyVecEnv([lambda: env])
            )
        backend = env.get_wrapper_attr("backend")
        totals = dict.fromkeys(weights, 0.0)
        try:
            # Shot 0 is the start state as loaded, before reset's seeded no-op frames, so it is
            # the same screen in every cell that starts from this state.
            backend.load_state(env.get_wrapper_attr("_start_state"))
            start = (self.label, "press 0  return 0.00", self.config.start_state)
            yield Shot(backend.screen(), start, dict(totals))
            obs, _ = env.reset(seed=self.seed)
            step, truncated = 0, False
            # Every episode runs to the step budget; AlgoArcadeEnv never terminates.
            while not truncated:
                step += 1
                policy_obs = normalize.normalize_obs(obs) if normalize else obs
                action, _ = model.predict(policy_obs, deterministic=True)
                obs, reward, _, truncated, info = env.step(action)
                self.episode_return += float(reward)
                for name, weight in weights.items():
                    totals[name] += weight * info[f"reward/{name}"]
                self.location = info["location"]
                if info["goal"] and self.goal_step is None:
                    self.goal_step = step
                lines = (
                    self.label,
                    f"press {step}  return {self.episode_return:.2f}",
                    f"{self.location}  badges {info['badge_count']}",
                )
                yield Shot(backend.screen(), lines, dict(totals))
                if self.until_goal and self.goal_step is not None:
                    return
        finally:
            env.close()


def compose(cells: Sequence[Iterable[Shot]], out: Path) -> None:
    """Writes the cells side by side, one video frame per button press, so the cells line up
    on presses rather than time. Frame 0 is every cell's start screen, and must be the same
    screen in all of them. A cell that ends holds its last frame until the longest ends."""
    streams = [iter(cell) for cell in cells]
    starts = [next(stream) for stream in streams]
    if any(not np.array_equal(shot.screen, starts[0].screen) for shot in starts):
        raise ValueError("the cells do not start from the same screen")
    images = [frame(shot) for shot in starts]
    height = max(image.shape[0] for image in images)
    width = sum(image.shape[1] for image in images)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(
        str(out), (width, height), fps=FPS, codec="libx264", pix_fmt_out="yuv420p"
    )
    writer.send(None)
    try:
        while True:
            row = np.full((height, width, 3), BACKGROUND, np.uint8)
            left = 0
            for image in images:
                row[: image.shape[0], left : left + image.shape[1]] = image
                left += image.shape[1]
            writer.send(row.tobytes())
            shots = [next(stream, None) for stream in streams]
            if all(shot is None for shot in shots):
                return
            images = [
                image if shot is None else frame(shot)
                for shot, image in zip(shots, images, strict=True)
            ]
    finally:
        writer.close()


def render(
    checkpoint: Path, out: Path, seed: int = 0, grid: int = 1
) -> list[tuple[float, tuple[int, ...]]]:
    """Writes the video and returns each episode's return and final location."""
    cells = [Policy(checkpoint, seed + i) for i in range(grid)]
    compose(cells, out)
    return [(cell.episode_return, cell.location) for cell in cells]


def main(checkpoint: Path, out: Path, seed: int = 0, grid: int = 1) -> None:
    """Render a checkpoint's episode from seed `seed`, or `grid` episodes side by side."""
    results = render(checkpoint, out, seed, grid)
    for i, (episode_return, location) in enumerate(results):
        print(f"seed {seed + i}: return {episode_return:.6f}, final location {location}")  # noqa: T201
    print(out)  # noqa: T201


if __name__ == "__main__":
    tyro.cli(main)
