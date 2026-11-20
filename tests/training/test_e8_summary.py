"""The E8 summary on fake-game runs from the runner."""

import dataclasses
import warnings
from pathlib import Path

import pytest

pytest.importorskip("stable_baselines3")

from e8_summary import score, scores, summary

from algoarcade.training.config import FIRERED
from algoarcade.training.games import FAKE_STEP_BUDGET
from algoarcade.training.runner import train


def test_a_run_scores_its_last_checkpoint_and_its_first_goal() -> None:
    rows = [
        {"checkpoint_step": 100, "goal_reached": False},
        {"checkpoint_step": 200, "goal_reached": True},
        {"checkpoint_step": 200, "goal_reached": False},
        {"checkpoint_step": 300, "goal_reached": True},
    ]
    assert score(rows) == (1.0, 200)
    assert score(rows[:1]) == (0.0, None)


def test_three_seeds_are_labelled_exploratory() -> None:
    arms: dict[str, list[tuple[float, int | None]]] = {
        "curriculum": [(1.0, 200), (0.5, 300), (0.0, None)],
        "none": [(0.0, None)] * 3,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lines = summary(arms).splitlines()
    assert lines[0] == (
        "curriculum: final eval goal rate IQM 0.50 [0.00, 1.00] (n=3); "
        "first goal at 200, 300, never steps; 3 seeds, exploratory"
    )
    assert lines[1].endswith("first goal at never, never, never steps; 3 seeds, exploratory")
    assert lines[2] == "curriculum over none: P(X > Y) 0.83 (n=3, m=3)"


def test_scores_read_each_finished_e8_run_by_arm(tmp_path: Path) -> None:
    # 1 env, two 64-step episodes, an evaluation at 64 and 128.
    config = dataclasses.replace(
        FIRERED,
        env_id="fake",
        start_state="",
        total_timesteps=128,
        n_envs=1,
        checkpoint_every=64,
        eval_episodes=1,
        reward_weights={"explore": 0.1},
        ppo=dataclasses.replace(FIRERED.ppo, n_steps=FAKE_STEP_BUDGET, batch_size=32),
        experiment_id="E8",
        runs_dir=tmp_path,
    )
    for arm in ("none", "curriculum"):
        train(dataclasses.replace(config, arm=arm))
    arms = scores(tmp_path)
    assert sorted(arms) == ["curriculum", "none"]
    assert all(len(runs) == 1 for runs in arms.values())
