from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from algoarcade.logs import fake
from algoarcade.viz.registry import registry

# Enough for each of the fake logger's walkers to finish one episode.
ENV_STEPS = fake.NUM_ENVS * fake.EPISODE_STEPS + 1_000


def write_fake_run(root: Path, run_id: str, seed: int) -> None:
    fake.main(
        [
            "--run-id",
            run_id,
            "--seed",
            str(seed),
            "--env-steps",
            str(ENV_STEPS),
            "--root",
            str(root),
        ]
    )


@pytest.fixture
def root(tmp_path: Path) -> Path:
    for seed in range(3):
        write_fake_run(tmp_path, f"fake{seed}", seed)
    return tmp_path


def by_run(table: pa.Table) -> dict[str, dict]:
    return {row["run_id"]: row for row in table.to_pylist()}


def test_one_row_per_run_folder(root: Path) -> None:
    folders = sorted(path.name.removeprefix("run_id=") for path in root.glob("run_id=*"))
    assert registry(root).column("run_id").to_pylist() == folders


def test_a_run_reads_its_end_row_and_a_killed_run_reads_running(root: Path) -> None:
    # The fake logger writes the start row to fake-00000.parquet and the end row to the next file.
    (root / "run_id=fake1" / "runs" / "fake-00001.parquet").unlink()
    rows = by_run(registry(root))
    assert rows["fake0"]["status"] == "finished"
    assert rows["fake0"]["steps"] == ENV_STEPS
    assert rows["fake1"]["status"] == "running"
    assert rows["fake1"]["steps"] == 0


def test_final_score_is_the_mean_eval_return_at_the_last_checkpoint(root: Path) -> None:
    episodes = pq.read_table(root / "run_id=fake2" / "episodes")
    evals = episodes.slice(0, 3)
    evals = evals.set_column(
        evals.schema.get_field_index("is_eval"), "is_eval", pa.array([True] * 3)
    )
    for name, values, type_ in (
        ("checkpoint_step", [500_000, 1_000_000, 1_000_000], pa.int64()),
        ("episode_return", [100.0, 2.0, 4.0], pa.float64()),
    ):
        evals = evals.set_column(evals.schema.get_field_index(name), name, pa.array(values, type_))
    pq.write_table(evals, root / "run_id=fake2" / "episodes" / "evals-0.parquet")
    rows = by_run(registry(root))
    assert rows["fake2"]["final_score"] == pytest.approx(3.0)
    assert rows["fake0"]["final_score"] is None


def test_furthest_milestone_is_the_best_episode_milestone(root: Path) -> None:
    episodes = pq.read_table(root / "run_id=fake0" / "episodes")
    best = max(episodes.column("milestone_count").to_pylist())
    assert by_run(registry(root))["fake0"]["furthest_milestone"] == best


def test_an_empty_runs_folder_gives_an_empty_registry(tmp_path: Path) -> None:
    assert registry(tmp_path).num_rows == 0


def test_archive_is_null_when_the_config_does_not_set_it(root: Path) -> None:
    # The fake logger's hyperparameters have no archive key; a training config always writes one.
    assert registry(root).column("archive").to_pylist() == [None, None, None]
