import asyncio
import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")

import imageio_ffmpeg
from anthropic.types import Message, ToolUseBlock, Usage

from algoarcade.agent import loop, server
from algoarcade.agent.tools import Game
from algoarcade.backends.base import Button
from algoarcade.training.config import PRESETS
from algoarcade.training.games import FAKE_STEP_BUDGET
from algoarcade.training.runner import train
from algoarcade.viz.render import Policy, compose
from algoarcade.viz.video_grid import Replay

# The chest is two tiles right of the fake game's start.
CHEST = ["RIGHT", "RIGHT", "A"]
FIRERED = PRESETS["firered"][1]
FAKE = dataclasses.replace(
    FIRERED,
    env_id="fake",
    start_state="",
    total_timesteps=FAKE_STEP_BUDGET,
    n_envs=1,
    checkpoint_every=FAKE_STEP_BUDGET,
    eval_every=FAKE_STEP_BUDGET,
    eval_episodes=1,
    reward_weights={"events": 1.0, "explore": 0.1},
    ppo=dataclasses.replace(FIRERED.ppo, n_steps=FAKE_STEP_BUDGET, batch_size=32),
)


async def press_chest(**_: Any) -> Message:
    block = ToolUseBlock(type="tool_use", id="toolu_1", name="press", input={"buttons": CHEST})
    return Message(
        id="msg_1",
        type="message",
        role="assistant",
        model="stub",
        content=[block],
        stop_reason="tool_use",
        stop_sequence=None,
        usage=Usage(input_tokens=1_000, output_tokens=0),
    )


@pytest.fixture(scope="module")
def attempt(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A scripted L2 attempt on the fake game, logged by the real loop and server."""
    root = tmp_path_factory.mktemp("llm")
    settings = loop.Run(
        name="E7-stub-L2-fake-00",
        segment="fake",
        level=2,
        max_actions=10,
        max_dollars=1.0,
        out=root / "llm",
        runs=root / "runs",
    )
    asyncio.run(loop.run(settings, press_chest))
    return root / "llm" / settings.name


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    run_dir = train(dataclasses.replace(FAKE, runs_dir=tmp_path_factory.mktemp("runs")))
    return run_dir / "checkpoints" / f"step_{FAKE_STEP_BUDGET}.zip"


def copy(attempt: Path, tmp_path: Path) -> Path:
    folder = tmp_path / attempt.name
    folder.mkdir()
    for name in ("attempt.json", "presses.jsonl"):
        (folder / name).write_text((attempt / name).read_text())
    return folder


def test_replaying_a_successful_attempt_reaches_its_goal_on_the_same_press(attempt: Path):
    summary = json.loads((attempt / "attempt.json").read_text())
    replay = Replay(attempt)
    shots = list(replay)
    assert summary["outcome"] == "goal"
    assert replay.goal_step == summary["goal_step"] == len(CHEST)
    assert len(shots) == 1 + len(CHEST)


def test_frame_0_of_every_cell_is_the_same_screen(checkpoint: Path, attempt: Path):
    cells = [Policy(checkpoint, seed=0), Policy(checkpoint, seed=5), Replay(attempt)]
    firsts = [next(iter(cell)).screen for cell in cells]
    for screen in firsts[1:]:
        np.testing.assert_array_equal(screen, firsts[0])


def test_the_grid_has_a_frame_per_press_of_its_longest_cell(
    checkpoint: Path, attempt: Path, tmp_path: Path
):
    out = tmp_path / "grid.mp4"
    compose([Policy(checkpoint), Replay(attempt)], out)
    reader = imageio_ffmpeg.read_frames(str(out))
    width, height = cast("dict[str, Any]", next(reader))["size"]
    assert sum(1 for _ in reader) == 1 + FAKE_STEP_BUDGET
    assert width % 16 == 0
    assert height % 16 == 0


def test_a_replay_that_reads_a_state_the_trace_did_not_log_is_refused(
    attempt: Path, tmp_path: Path
):
    folder = copy(attempt, tmp_path)
    lines = [json.loads(line) for line in (folder / "presses.jsonl").read_text().splitlines()]
    lines[1]["button"] = "LEFT"
    (folder / "presses.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines))
    with pytest.raises(ValueError, match="press 1"):
        list(Replay(folder))


def test_a_replay_from_another_start_is_refused(attempt: Path, tmp_path: Path):
    folder = copy(attempt, tmp_path)
    lines = (folder / "presses.jsonl").read_text().splitlines()
    first = json.loads(lines[0]) | {"start_sha256": "0" * 64}
    (folder / "presses.jsonl").write_text("\n".join([json.dumps(first), *lines[1:]]) + "\n")
    with pytest.raises(ValueError, match="starts from"):
        list(Replay(folder))


# Bedroom (6, 6) to the stairs, down to the front door: scripts/capture_states.py's route to
# seg_01_house_goal. The first press in a new direction only turns the player, and the first
# press after the stairs is lost to the warp; found by trying on rev 1.
SEG_01 = "R" * 5 + "U" * 5 + "L" * 2 + "D" * 8 + "L" * 7 + "D" * 2
BUTTONS = {"R": Button.RIGHT, "L": Button.LEFT, "U": Button.UP, "D": Button.DOWN}


@pytest.mark.rom
def test_a_scripted_seg_01_trace_replays_to_the_goal(tmp_path: Path):
    if not (server.DATA / "roms" / "firered.gba").is_file():
        pytest.skip("no FireRed ROM at data/roms/firered.gba")
    if not (server.DATA / "states" / "seg_01_house.state").is_file():
        pytest.skip("no seg_01_house state in data/states/")
    pytest.importorskip(
        "mgba.core", reason="mGBA binding did not load; see docs/setup.md", exc_type=ImportError
    )
    folder = tmp_path / "seg01"
    folder.mkdir()
    game: Game = server.GAMES["firered"]()
    game.backend.load_state(server.start_state("seg_01_house"))
    with (folder / "presses.jsonl").open("w") as game.trace:
        start = hashlib.sha256(game.backend.save_state()).hexdigest()
        game.record(None, pid=0, start_sha256=start)
        for letter in SEG_01:
            game.press(BUTTONS[letter])
            game.read()
    game.backend.close()
    attempt = {"model": "scripted", "level": 2, "segment": "seg_01_house"}
    (folder / "attempt.json").write_text(json.dumps(attempt))
    replay = Replay(folder)
    shots = list(replay)
    # The replay grades its own reads with the manifest's left_house; any read that differed
    # from the trace would have raised.
    print(f"seg_01_house replay: goal at press {replay.goal_step} of {len(SEG_01)}")  # noqa: T201
    assert replay.goal_step == len(SEG_01)
    assert len(shots) == 1 + len(SEG_01)
