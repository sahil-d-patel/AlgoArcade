import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from algoarcade.logs.fake import MAPS, NUM_ENVS, SAMPLE_EVERY, WEIGHTS, fake_run, main
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
    assert run["runs"]["env_steps"].to_pylist() == [ENV_STEPS]


def test_steps_are_sampled_every_nth_episode_step(run: dict) -> None:
    episode_steps = run["steps"]["episode_step"].to_pylist()
    assert all(s % SAMPLE_EVERY == 0 for s in episode_steps)
    assert run["steps"].num_rows == ENV_STEPS // SAMPLE_EVERY


def test_total_reward_is_the_weighted_sum_of_the_components(run: dict) -> None:
    for table, total in (("steps", "reward_total"), ("episodes", "episode_return")):
        t = run[table]
        weighted = sum(t[f"reward_{c}"].to_numpy() * WEIGHTS[c] for c in WEIGHTS)
        assert abs(weighted - t[total].to_numpy()).max() < 1e-9
    off = [c for c in REWARD_COMPONENTS if c not in WEIGHTS]
    assert all(run["steps"][f"reward_{c}"].null_count == run["steps"].num_rows for c in off)


def test_the_run_row_records_the_weights_and_envs(run: dict) -> None:
    row = run["runs"].to_pylist()[0]
    assert json.loads(row["hyperparameters"])["reward_weights"] == WEIGHTS
    assert row["num_envs"] == NUM_ENVS


def test_main_writes_every_table_under_the_run_directory(tmp_path: Path) -> None:
    main(["--run-id", "fake1", "--seed", "3", "--env-steps", "10000", "--root", str(tmp_path)])
    expected = fake_run("fake1", seed=3, env_steps=10_000)
    for table in TABLES:
        files = sorted((tmp_path / "run_id=fake1" / table).glob("*.parquet"))
        assert files
        assert pq.read_table(files[0]).equals(expected[table])
