import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from algoarcade.logs.fake import (
    EVAL_EPISODES,
    MAPS,
    NUM_ENVS,
    STEP_LOG_EVERY,
    WEIGHTS,
    fake_run,
    main,
)
from algoarcade.logs.schema import REWARD_COMPONENTS, TABLES, validate

ENV_STEPS = 40_000


@pytest.fixture(scope="module")
def run() -> dict:
    return fake_run("fake0", seed=0, env_steps=ENV_STEPS)


def test_every_table_passes_validation(run: dict) -> None:
    assert sorted(run) == sorted(TABLES)
    for table, batch in run.items():
        validate(batch, table)
        assert batch.schema == TABLES[table]


def test_the_same_seed_gives_identical_tables(run: dict) -> None:
    again = fake_run("fake0", seed=0, env_steps=ENV_STEPS)
    assert all(again[t].equals(run[t]) for t in TABLES)


def test_another_seed_gives_another_walk(run: dict) -> None:
    other = fake_run("fake0", seed=1, env_steps=ENV_STEPS)
    assert not other["steps"].equals(run["steps"])


def test_only_real_map_ids_and_tiles_on_them(run: dict) -> None:
    sizes = {(g, n): (w, h) for _, g, n, w, h in MAPS}
    for table in ("steps", "visits"):
        t = run[table].to_pydict()
        for g, n, x, y in zip(t["map_group"], t["map_num"], t["x"], t["y"], strict=True):
            w, h = sizes[(g, n)]
            assert 0 <= x < w
            assert 0 <= y < h


def test_some_episodes_earn_a_badge_and_some_do_not(run: dict) -> None:
    badges = set(run["episodes"]["badge_count"].to_pylist())
    assert badges == {0, 1}


def test_every_step_is_one_visit(run: dict) -> None:
    assert run["visits"]["visits"].to_numpy().sum() == ENV_STEPS


def test_the_run_has_a_start_row_and_an_end_row(run: dict) -> None:
    runs = run["runs"].to_pydict()
    assert runs["status"] == ["running", "finished"]
    assert runs["env_steps"] == [0, ENV_STEPS]
    assert runs["step_log_every"] == [STEP_LOG_EVERY] * 2


def test_goal_step_is_set_exactly_when_the_goal_is_reached(run: dict) -> None:
    e = run["episodes"].to_pydict()
    assert e["goal_reached"] == [b == 1 for b in e["badge_count"]]
    for reached, step, length in zip(
        e["goal_reached"], e["goal_step"], e["episode_length"], strict=True
    ):
        assert (step is not None) == reached
        assert step is None or 1 <= step <= length


def test_a_badge_pays_on_the_step_after_it_is_earned_as_in_real_runs() -> None:
    run = fake_run("n1", seed=0, env_steps=ENV_STEPS, step_log_every=1)
    e = run["episodes"].to_pydict()
    goals = {
        (env, episode): (goal, length)
        for env, episode, goal, length, is_eval in zip(
            e["env_idx"],
            e["episode_id"],
            e["goal_step"],
            e["episode_length"],
            e["is_eval"],
            strict=True,
        )
        if not is_eval
    }
    s = run["steps"].to_pydict()
    paid = {
        (env, episode): step
        for env, episode, step, badge in zip(
            s["env_idx"], s["episode_id"], s["episode_step"], s["reward_badge"], strict=True
        )
        if badge
    }
    assert paid
    assert all(step == goals[key][0] + 1 for key, step in paid.items())
    # A badge earned on an episode's last step is never confirmed.
    assert all(
        (key in paid) == (goal is not None and goal < length)
        for key, (goal, length) in goals.items()
    )


def test_evaluation_episodes_run_at_each_checkpoint_and_add_no_steps() -> None:
    run = fake_run("eval", seed=0, env_steps=1_000_000)
    e = run["episodes"].to_pydict()
    checkpoints = [
        c for c, is_eval in zip(e["checkpoint_step"], e["is_eval"], strict=True) if is_eval
    ]
    assert checkpoints == [500_000] * EVAL_EPISODES + [1_000_000] * EVAL_EPISODES
    assert all(
        c is None
        for c, is_eval in zip(e["checkpoint_step"], e["is_eval"], strict=True)
        if not is_eval
    )
    assert run["visits"]["visits"].to_numpy().sum() == 1_000_000
    assert run["steps"].num_rows == 1_000_000 // STEP_LOG_EVERY


def test_an_llm_style_run_logs_every_step() -> None:
    run = fake_run("n1", seed=0, env_steps=4_000, step_log_every=1)
    assert run["steps"].num_rows == 4_000
    assert run["runs"]["step_log_every"].to_pylist() == [1, 1]


def test_steps_are_sampled_every_nth_episode_step(run: dict) -> None:
    episode_steps = run["steps"]["episode_step"].to_pylist()
    assert all(s % STEP_LOG_EVERY == 0 for s in episode_steps)
    assert run["steps"].num_rows == ENV_STEPS // STEP_LOG_EVERY


def test_total_reward_is_the_weighted_sum_of_the_components(run: dict) -> None:
    for table, total in (("steps", "reward_total"), ("episodes", "episode_return")):
        t = run[table]
        weighted = sum(t[f"reward_{c}"].to_numpy() * WEIGHTS[c] for c in WEIGHTS)
        assert abs(weighted - t[total].to_numpy()).max() < 1e-9
    off = [c for c in REWARD_COMPONENTS if c not in WEIGHTS]
    assert all(run["steps"][f"reward_{c}"].null_count == run["steps"].num_rows for c in off)


def test_the_run_row_records_the_weights_and_envs(run: dict) -> None:
    row = run["runs"].to_pylist()[-1]
    assert json.loads(row["hyperparameters"])["reward_weights"] == WEIGHTS
    assert row["num_envs"] == NUM_ENVS


def test_main_writes_every_table_under_the_run_directory(tmp_path: Path) -> None:
    main(["--run-id", "fake1", "--seed", "3", "--env-steps", "10000", "--root", str(tmp_path)])
    expected = fake_run("fake1", seed=3, env_steps=10_000)
    for table in TABLES:
        assert pq.read_table(tmp_path / "run_id=fake1" / table).equals(expected[table])
    assert len(list((tmp_path / "run_id=fake1" / "runs").glob("*.parquet"))) == 2


def test_main_refuses_a_run_id_that_already_has_files(tmp_path: Path) -> None:
    main(["--run-id", "r", "--seed", "0", "--env-steps", "10000", "--root", str(tmp_path)])
    with pytest.raises(FileExistsError):
        main(["--run-id", "r", "--seed", "1", "--env-steps", "10000", "--root", str(tmp_path)])
    for table in TABLES:
        seeds = pq.read_table(tmp_path / "run_id=r" / table).to_pydict().get("seed", [0])
        assert set(seeds) == {0}
