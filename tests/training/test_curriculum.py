"""Curriculum resets on the fake game (AGT-06, E8)."""

import json
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")

from stable_baselines3.common.vec_env import DummyVecEnv

from algoarcade.agent.curriculum import keep
from algoarcade.agent.tools import Game
from algoarcade.backends import fake
from algoarcade.backends.base import Button
from algoarcade.backends.fake import FakeBackend
from algoarcade.envs.env import AlgoArcadeEnv
from algoarcade.games.fake import FEATURES, FakeField, FakeReader, FakeState
from algoarcade.rewards.registry import COMPONENTS, DEFAULT_WEIGHTS, WEIGHTS, RewardRegistry
from algoarcade.training.curriculum import CurriculumResets, load
from algoarcade.training.games import fake_env

START = (0, 1, 1)
A, RIGHT = 4, 3


def capture(tmp_path: Path, buttons: list[Button], every: int, goal_step: int) -> Path:
    """A fake attempt pressing `buttons` through the server's Game, kept as a curriculum with
    the goal at `goal_step`. Between the first and second press every count rises."""
    captured = tmp_path / "captured"
    captured.mkdir()
    backend = FakeBackend()
    game = Game(backend, FakeReader(), FakeField(), capture=captured, capture_every=every)
    with (tmp_path / "presses.jsonl").open("w") as game.trace:
        game.record(None, start_sha256="")
        for i, button in enumerate(buttons):
            if i == 1:
                raise_counts(backend)
            game.press(button)
    presses = [json.loads(line) for line in (tmp_path / "presses.jsonl").read_text().splitlines()]
    folder = tmp_path / "curriculum"
    keep(presses, captured, folder, goal_step, lambda s: FakeState(**s), "fake", every)
    return folder


def raise_counts(backend: FakeBackend) -> None:
    ram = bytearray(backend.save_state())
    # A story flag but not the chest, which is the goal.
    for offset, value in (
        (fake.FLAGS, 0b10),
        (fake.LEVEL_SUM, 9),
        (fake.BADGES, 1),
        (fake.MILESTONES, 3),
        (fake.HP_PERCENT, 50),
    ):
        ram[offset] = value
    backend.load_state(bytes(ram))


def walk_to_chest(tmp_path: Path) -> Path:
    # Right, right, A on the chest at (3, 1): states after presses 1 and 2, the goal on 3.
    return capture(tmp_path, [Button.RIGHT, Button.RIGHT, Button.A], every=1, goal_step=3)


def chest_policy(location: tuple[Any, ...]) -> int:
    """Right to the chest's column, then A."""
    return A if location[1] == 3 else RIGHT


def play(venv: CurriculumResets, episodes: int, policy: Any) -> tuple[list[Any], list[int]]:
    """The start location of each episode and the stage after each episode ends."""
    venv.reset()
    location = venv.reset_infos[0]["location"]
    starts, stages = [location], []
    while len(stages) < episodes:
        _, _, dones, infos = venv.step(np.array([policy(location)]))
        location = infos[0]["location"]
        if dones[0]:
            location = venv.reset_infos[0]["location"]
            starts.append(location)
            stages.append(venv.stage)
    return starts, stages


def test_starts_at_the_last_state_then_steps_one_earlier_once_success_crosses_the_threshold(
    tmp_path: Path,
) -> None:
    cells = load(walk_to_chest(tmp_path), "fake")
    assert [cell.key for cell in cells] == [(1,), (2,)]
    env = partial(fake_env, DEFAULT_WEIGHTS, curriculum=True)
    venv = CurriculumResets(DummyVecEnv([env]), cells, threshold=1.0, window=2)
    starts, stages = play(venv, 6, chest_policy)
    # Two goals from (3, 1) move the stage back. The env had already reset into (3, 1) for
    # episode 3, which does not count; episodes 4 and 5 from (2, 1) move it to the fixed start.
    assert starts == [(0, 3, 1)] * 3 + [(0, 2, 1)] * 3 + [START]
    assert stages == [1, 0, 0, 0, -1, -1]


def test_stays_on_the_last_state_while_success_is_under_the_threshold(tmp_path: Path) -> None:
    cells = load(walk_to_chest(tmp_path), "fake")
    env = partial(fake_env, DEFAULT_WEIGHTS, curriculum=True)
    venv = CurriculumResets(DummyVecEnv([env]), cells, threshold=0.5, window=2)
    starts, stages = play(venv, 4, lambda _location: 5)  # B every step: never opens the chest
    assert starts == [(0, 3, 1)] * 5
    assert stages == [1] * 4
    assert venv.success == 0.0


def test_an_env_built_without_curriculum_always_starts_from_the_start_state(
    tmp_path: Path,
) -> None:
    cells = load(walk_to_chest(tmp_path), "fake")
    venv = CurriculumResets(DummyVecEnv([partial(fake_env, DEFAULT_WEIGHTS)]), cells, 1.0, 2)
    starts, _ = play(venv, 4, chest_policy)
    assert starts == [START] * 5


@pytest.mark.parametrize("action", [5, 2], ids=["no-op", "back along the path"])
def test_every_reward_component_pays_zero_on_the_first_step_after_a_curriculum_reset(
    tmp_path: Path, action: int
) -> None:
    # The counts rise between presses 1 and 2, and the state after press 2 is saved, so its
    # read shows a rise no later read confirmed. A reset that paid it, or the path to the
    # state, would show here.
    folder = capture(tmp_path, [Button.DOWN, Button.RIGHT, Button.UP], every=2, goal_step=3)
    (cell,) = load(folder, "fake")
    backend = FakeBackend()
    env = AlgoArcadeEnv(
        backend,
        FakeReader(),
        RewardRegistry(COMPONENTS, WEIGHTS),
        backend.save_state(),
        50,
        FEATURES,
        curriculum=True,
    )
    env.reset(seed=0)
    env.set_next_start(cell)
    _, info = env.reset(seed=1)
    assert info["location"] == (0, 2, 2)
    assert info["archive_start"] is False
    # B does nothing in the fake; left goes to (1, 2), visited on the way to the state.
    _, reward, _, _, info = env.step(action)
    raw = {name: info[f"reward/{name}"] for name in WEIGHTS}
    # time charges every step by design (docs/harness-spec.md, Reward registry).
    assert raw == dict.fromkeys(WEIGHTS, 0.0) | {"time": 1.0}
    assert reward == WEIGHTS["time"]


def test_a_curriculum_from_another_start_state_or_a_changed_file_is_refused(
    tmp_path: Path,
) -> None:
    folder = walk_to_chest(tmp_path)
    with pytest.raises(ValueError, match="captured from fake"):
        load(folder, "seg_05_forest")
    (folder / "00001.state").write_bytes(b"\0" * fake.RAM_SIZE)
    with pytest.raises(ValueError, match="manifest hash"):
        load(folder, "fake")
