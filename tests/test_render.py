import dataclasses
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")

import imageio_ffmpeg

from algoarcade.training.config import PRESETS
from algoarcade.training.games import FAKE_STEP_BUDGET, fake_env
from algoarcade.training.runner import train
from algoarcade.viz import render

WEIGHTS = {"events": 1.0, "explore": 0.1, "time": -0.001}
RIGHT, DOWN = 3, 1
FIRERED = PRESETS["firered"][1]
FAKE = dataclasses.replace(
    FIRERED,
    env_id="fake",
    start_state="",
    total_timesteps=FAKE_STEP_BUDGET,
    n_envs=1,
    checkpoint_every=FAKE_STEP_BUDGET,
    eval_episodes=1,
    reward_weights=WEIGHTS,
    ppo=dataclasses.replace(FIRERED.ppo, n_steps=FAKE_STEP_BUDGET, batch_size=32),
)


def bars(image: np.ndarray, game_height: int) -> int:
    """Bands of non-background rows in the bar column, below the two header lines."""
    top = game_height + render.PAD + 2 * render.LINE
    column = image[top:, render.LABEL_WIDTH + 1]
    drawn = (column != render.BACKGROUND).any(axis=1)
    return int(np.count_nonzero(drawn[1:] & ~drawn[:-1]) + drawn[0])


def test_ten_frames_show_one_bar_per_enabled_component():
    env = fake_env(WEIGHTS)
    env.reset(seed=0)
    episode_return = 0.0
    totals = dict.fromkeys(WEIGHTS, 0.0)
    for step in range(1, 11):
        # Right, then down: the walk pays explore and leaves events at zero.
        _, reward, _, _, info = env.step(RIGHT if step % 2 else DOWN)
        episode_return += float(reward)
        for name, weight in WEIGHTS.items():
            totals[name] += weight * info[f"reward/{name}"]
        screen = env.get_wrapper_attr("backend").screen()
        image = render.frame(screen, step, episode_return, info, totals)
        height, width = screen.shape[0] * render.SCALE, screen.shape[1] * render.SCALE
        assert image.shape[1] == width
        assert image.shape[0] % 16 == 0
        assert width % 16 == 0
        assert bars(image, height) == len(WEIGHTS)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    run_dir = train(dataclasses.replace(FAKE, runs_dir=tmp_path_factory.mktemp("runs")))
    return run_dir / "checkpoints" / f"step_{FAKE_STEP_BUDGET}.zip"


def read_back(path: Path) -> tuple[int, tuple[int, int]]:
    reader = imageio_ffmpeg.read_frames(str(path))
    # The first item is the metadata, typed as a frame.
    size = cast("dict[str, Any]", next(reader))["size"]
    return sum(1 for _ in reader), size


def test_an_episode_writes_one_frame_per_step(checkpoint: Path, tmp_path: Path):
    out = tmp_path / "episode.mp4"
    render.render(checkpoint, out, seed=0)
    frames, (width, height) = read_back(out)
    assert frames == FAKE_STEP_BUDGET
    assert width % 16 == 0
    assert height % 16 == 0


def test_the_same_seed_plays_the_same_episode(checkpoint: Path, tmp_path: Path):
    first = render.render(checkpoint, tmp_path / "a.mp4", seed=3)
    assert render.render(checkpoint, tmp_path / "b.mp4", seed=3) == first


def test_grid_puts_four_seeds_side_by_side(checkpoint: Path, tmp_path: Path):
    render.render(checkpoint, tmp_path / "one.mp4", seed=0)
    results = render.render(checkpoint, tmp_path / "grid.mp4", seed=0, grid=4)
    _, (width, height) = read_back(tmp_path / "one.mp4")
    frames, size = read_back(tmp_path / "grid.mp4")
    assert len(results) == 4
    assert frames == FAKE_STEP_BUDGET
    assert size == (4 * width, height)
