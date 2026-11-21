from pathlib import Path

import check_runs
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from algoarcade.logs.fake import main as fake_main


def fake(root: Path, run_id: str, *, checkpoint: bool) -> Path:
    fake_main(["--run-id", run_id, "--seed", "0", "--env-steps", "4096", "--root", str(root)])
    run = root / f"run_id={run_id}"
    if checkpoint:
        (run / "checkpoints").mkdir()
        (run / "checkpoints" / "step_4096.zip").write_bytes(b"")
    return run


def test_a_finished_run_with_logs_and_a_checkpoint_passes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake(tmp_path, "good", checkpoint=True)
    assert check_runs.main(tmp_path) == 0
    assert capsys.readouterr().out.splitlines()[-1] == "1 finished runs checked, 0 failed."


def test_a_missing_checkpoint_or_a_column_off_schema_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake(tmp_path, "no-checkpoint", checkpoint=False)
    run = fake(tmp_path, "off-schema", checkpoint=True)
    path = sorted((run / "steps").glob("*.parquet"))[0]
    pq.write_table(pq.read_table(path).drop_columns(["map_group"]), path)
    assert check_runs.main(tmp_path) == 1
    out = capsys.readouterr().out.splitlines()
    assert out[-1] == "2 finished runs checked, 2 failed."
    assert any("no-checkpoint: no checkpoint" in line for line in out)
    assert any("off-schema" in line and "map_group" in line for line in out)


def test_pqn_and_llm_runs_pass_and_a_folder_without_a_runs_table_is_skipped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pqn = fake(tmp_path, "pqn", checkpoint=False)
    (pqn / "checkpoints").mkdir()
    (pqn / "checkpoints" / "step_4096.pt").write_bytes(b"")
    llm = fake(tmp_path, "llm", checkpoint=False)
    for path in (llm / "runs").glob("*.parquet"):
        table = pq.read_table(path)
        column = table.schema.get_field_index("algorithm")
        algorithm = pa.array(["llm"] * table.num_rows, table.schema.field(column).type)
        pq.write_table(table.set_column(column, table.schema.field(column), algorithm), path)
    (tmp_path / "run_id=crashed" / "checkpoints").mkdir(parents=True)
    assert check_runs.main(tmp_path) == 0
    assert capsys.readouterr().out.splitlines()[-1] == "2 finished runs checked, 0 failed."
