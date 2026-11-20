from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

from algoarcade.logs.schema import EPISODES, RUNS
from algoarcade.logs.writer import RollingWriter, run_dir
from algoarcade.viz import cost_table
from algoarcade.viz.site import connect

SEGMENT = "seg_01_house"
HOUR = 3600.0


def run_row(run_id: str, status: str, **fields: Any) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "status": status,
        "experiment_id": "E1",
        "arm": "ppo",
        "algorithm": "ppo",
        "trainer": "sb3",
        "seed": 0,
        "env_id": "firered",
        "backend": "mgba",
        "rom_sha1": "x",
        "start_state": SEGMENT,
        "obs_mode": "symbolic",
        "frames_per_decision": 24,
        "num_envs": 16,
        "reward_config_hash": "x",
        "git_commit": "x",
        "hyperparameters": "{}",
        "step_log_every": 4,
        "machine": "trainingbox",
        "env_steps": 0,
        "wall_clock_s": 0.0,
    } | fields


def episode(run_id: str, goal_step: int | None, clock: float, **fields: Any) -> dict[str, Any]:
    return (
        dict.fromkeys(EPISODES.names)
        | {
            "run_id": run_id,
            "seed": 0,
            "env_idx": 0,
            "episode_id": 0,
            "env_steps": 0,
            "wall_clock_s": clock,
            "episode_return": 0.0,
            "episode_length": 2048,
            "badge_count": 0,
            "party_level_sum": 5,
            "story_flags": 0,
            "unique_tiles": 1,
            "milestone_count": 0,
            "start_state": SEGMENT,
            "archive_start": False,
            "end_reason": "truncated",
            "goal_reached": goal_step is not None,
            "goal_step": goal_step,
            "is_eval": False,
        }
        | fields
    )


def write(root: Path, run_id: str, runs: list[dict], episodes: list[dict]) -> None:
    for table, rows, schema in (("runs", runs, RUNS), ("episodes", episodes, EPISODES)):
        writer = RollingWriter(run_dir(run_id, root), table, "test")
        writer.write(pa.Table.from_pylist(rows, schema=schema))
        writer.close()


def rl(root: Path, run_id: str, arm: str, evals: list[tuple[int, int | None, float]]) -> None:
    """A finished RL run with eval episodes (checkpoint_step, goal_step, wall_clock_s)."""
    rows = [
        episode(run_id, goal, clock, is_eval=True, checkpoint_step=step, env_steps=step)
        for step, goal, clock in evals
    ]
    # A training episode that reached the goal before any checkpoint did: not an evaluation.
    rows.append(episode(run_id, 10, 60.0, env_steps=1_000))
    runs = [run_row(run_id, "running", arm=arm), run_row(run_id, "finished", arm=arm)]
    write(root, run_id, runs, rows)


def llm(root: Path, run_id: str, model: str, goal_step: int | None, dollars: float) -> None:
    usage = {
        "llm_model": model,
        "llm_tool_level": 2,
        "llm_tokens_in": 1_000,
        "llm_tokens_out": 100,
        "llm_cost_usd": dollars,
        "end_reason": "goal" if goal_step else "action_cap",
    }
    fields = {"experiment_id": "E7", "arm": f"{model}-L2", "algorithm": "llm"}
    runs = [run_row(run_id, "running", **fields), run_row(run_id, "finished", **fields)]
    write(root, run_id, runs, [episode(run_id, goal_step, 30.0, **usage)])


@pytest.fixture
def table(tmp_path: Path) -> dict[str, dict[str, Any]]:
    # Seed 0 first succeeds at 1M steps, its evaluation ending at 3,800 s; seed 1 at 500k, ending
    # at 2,100 s. The rnd arm never succeeds.
    rl(tmp_path, "ppo0", "ppo", [(500_000, None, 1_800.0), (1_000_000, 300, 3_700.0)])
    rl(tmp_path, "ppo1", "ppo", [(500_000, 200, 2_000.0), (500_000, 400, 2_100.0)])
    rl(tmp_path, "rnd0", "rnd", [(500_000, None, 1_800.0), (1_000_000, None, 3_600.0)])
    for index, (goal, dollars) in enumerate([(40, 0.5), (60, 0.7), (None, 1.0), (None, 0.8)]):
        llm(tmp_path, f"E7-haiku-{index}", "claude-haiku-5-5", goal, dollars)
    for index in range(3):
        llm(tmp_path, f"E7-opus-{index}", "claude-opus-5-5", None, 2.0)
    # A crashed attempt has no finished row and does not count.
    write(
        tmp_path,
        "E7-haiku-crashed",
        [run_row("E7-haiku-crashed", "running", experiment_id="E7", algorithm="llm")],
        [],
    )
    rows = cost_table.rows(connect(tmp_path))
    return {row["agent"]: row for row in rows}


def test_rl_cells_are_medians_over_the_seeds_that_succeeded(table: dict[str, dict]) -> None:
    ppo = table["E1 ppo"]
    assert (ppo["tried"], ppo["successes"]) == (2, 2)
    assert ppo["steps"] == 750_000
    assert ppo["hours"] == pytest.approx((3_700.0 + 2_100.0) / 2 / HOUR)
    assert ppo["presses"] == 300


def test_llm_cells_count_failed_attempts(table: dict[str, dict]) -> None:
    haiku = table["claude-haiku-5-5 L2"]
    assert (haiku["tried"], haiku["successes"]) == (4, 2)
    # Clopper-Pearson for 2 of 4: the 2.5% and 97.5% points of Beta(2, 3) and Beta(3, 2).
    assert haiku["interval"] == pytest.approx((0.067586, 0.932414), abs=1e-6)
    assert haiku["presses"] == 50
    assert haiku["per_success"] == pytest.approx(3.0 / 2)


def test_no_success_shows_none_and_what_was_tried(table: dict[str, dict]) -> None:
    text = cost_table.markdown(list(table.values()))
    rnd = next(line for line in text.splitlines() if line.startswith("| E1 rnd "))
    opus = next(line for line in text.splitlines() if line.startswith("| claude-opus-5-5 L2 "))
    assert rnd.endswith("| 1 seed | 0 | n/a | none in 1,000,000 steps | none | none | n/a |")
    assert "| 3 attempts | 0 |" in opus
    assert opus.endswith("| none | none |")


def test_the_caption_names_the_machines_the_runs_trained_on(tmp_path: Path) -> None:
    rl(tmp_path, "ppo0", "ppo", [(500_000, 200, 2_000.0)])
    assert "trainingbox" in cost_table.caption(connect(tmp_path))
