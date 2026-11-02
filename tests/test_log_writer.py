from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from algoarcade.logs.fake import SAMPLE_EVERY, fake_run
from algoarcade.logs.schema import STEPS, SchemaError
from algoarcade.logs.writer import RollingWriter


def test_closed_files_are_readable_while_the_writer_is_open(tmp_path: Path) -> None:
    steps = fake_run("w0", seed=0, env_steps=10_000 * SAMPLE_EVERY)["steps"]
    assert steps.num_rows == 10_000
    writer = RollingWriter("w0", "steps", "main", root=tmp_path, rows_per_file=1_500)
    run_dir = tmp_path / "run_id=w0" / "steps"
    written = 0
    for batch in steps.to_batches(max_chunksize=400):
        writer.write(pa.Table.from_batches([batch]))
        written += batch.num_rows
        closed = sorted(run_dir.glob("*.parquet"))
        # A file closes on the first write that takes it to 1,500 rows or more: 4 batches, 1,600.
        assert len(closed) == written // 1_600
        if closed:
            read = pa.concat_tables(pq.read_table(p) for p in closed)
            assert read.schema == STEPS
            assert read.equals(steps.slice(0, read.num_rows))
    writer.close()
    files = sorted(run_dir.glob("*.parquet"))
    assert [p.name for p in files[:2]] == ["main-00000.parquet", "main-00001.parquet"]
    assert pa.concat_tables(pq.read_table(p) for p in files).equals(steps)
    assert not list(run_dir.glob("*.inprogress"))


def test_a_bad_batch_is_rejected_before_anything_is_written(tmp_path: Path) -> None:
    steps = fake_run("w1", seed=0, env_steps=400)["steps"]
    writer = RollingWriter("w1", "steps", "main", root=tmp_path)
    with pytest.raises(SchemaError):
        writer.write(steps.drop_columns(["x"]))
    writer.close()
    assert not (tmp_path / "run_id=w1").exists()
