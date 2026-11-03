from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest

from algoarcade.logs.fake import STEP_LOG_EVERY, fake_run
from algoarcade.logs.schema import STEPS, SchemaError
from algoarcade.logs.writer import RollingWriter, run_dir


def test_closed_files_are_readable_while_the_writer_is_open(tmp_path: Path) -> None:
    steps = fake_run("w0", seed=0, env_steps=10_000 * STEP_LOG_EVERY)["steps"]
    assert steps.num_rows == 10_000
    writer = RollingWriter(run_dir("w0", tmp_path), "steps", "main", rows_per_file=1_500)
    steps_dir = tmp_path / "run_id=w0" / "steps"
    written = 0
    for batch in steps.to_batches(max_chunksize=400):
        writer.write(pa.Table.from_batches([batch]))
        written += batch.num_rows
        closed = sorted(steps_dir.glob("*.parquet"))
        # A file closes on the first write that takes it to 1,500 rows or more: 4 batches, 1,600.
        assert len(closed) == written // 1_600
        if closed:
            read = pa.concat_tables(pq.read_table(p) for p in closed)
            assert read.schema == STEPS
            assert read.equals(steps.slice(0, read.num_rows))
            # A folder read skips the open file, which starts with an underscore.
            assert pq.read_table(steps_dir).equals(read)
            assert ds.dataset(steps_dir).count_rows() == read.num_rows
    writer.close()
    files = sorted(steps_dir.glob("*.parquet"))
    assert [p.name for p in files[:2]] == ["main-00000.parquet", "main-00001.parquet"]
    assert pa.concat_tables(pq.read_table(p) for p in files).equals(steps)
    assert not list(steps_dir.glob("*.inprogress"))


def test_a_bad_batch_is_rejected_before_anything_is_written(tmp_path: Path) -> None:
    steps = fake_run("w1", seed=0, env_steps=400)["steps"]
    writer = RollingWriter(run_dir("w1", tmp_path), "steps", "main")
    with pytest.raises(SchemaError):
        writer.write(steps.drop_columns(["x"]))
    writer.close()
    assert not (tmp_path / "run_id=w1").exists()


STEPS_1000 = fake_run("r", seed=0, env_steps=1_000 * STEP_LOG_EVERY)["steps"]


def write(root: Path, writer_id: str, rows: pa.Table, resume: bool = False) -> None:
    writer = RollingWriter(run_dir("r", root), "steps", writer_id, 400, resume=resume)
    for batch in rows.to_batches(max_chunksize=100):
        writer.write(pa.Table.from_batches([batch]))
    writer.close()


def test_reusing_a_writer_id_without_resume_is_refused(tmp_path: Path) -> None:
    write(tmp_path, "main", STEPS_1000)
    with pytest.raises(FileExistsError, match="resume"):
        write(tmp_path, "main", STEPS_1000.slice(0, 10))
    assert pq.read_table(tmp_path / "run_id=r" / "steps").num_rows == 1_000


def test_resume_appends_after_the_highest_file(tmp_path: Path) -> None:
    write(tmp_path, "main", STEPS_1000)
    write(tmp_path, "main", STEPS_1000.slice(0, 10), resume=True)
    steps_dir = tmp_path / "run_id=r" / "steps"
    names = [p.name for p in sorted(steps_dir.glob("*.parquet"))]
    assert names == [f"main-{i:05d}.parquet" for i in range(4)]
    assert pq.read_table(steps_dir).num_rows == 1_010


def test_resume_skips_past_a_file_left_open_by_a_crash(tmp_path: Path) -> None:
    steps_dir = tmp_path / "run_id=r" / "steps"
    steps_dir.mkdir(parents=True)
    (steps_dir / "_main-00000.parquet.inprogress").write_bytes(b"no footer")
    write(tmp_path, "main", STEPS_1000.slice(0, 10), resume=True)
    assert [p.name for p in steps_dir.glob("*.parquet")] == ["main-00001.parquet"]


def test_another_writer_id_in_the_same_run_is_not_refused(tmp_path: Path) -> None:
    write(tmp_path, "env00", STEPS_1000)
    write(tmp_path, "env01", STEPS_1000.slice(0, 10))
    assert pq.read_table(tmp_path / "run_id=r" / "steps").num_rows == 1_010
