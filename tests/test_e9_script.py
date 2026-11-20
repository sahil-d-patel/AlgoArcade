from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from e9 import final_progress, main

from algoarcade.logs.fake import fake_run
from algoarcade.logs.schema import SchemaError


def test_final_progress_takes_each_late_episodes_furthest_tile_in_1_1() -> None:
    # Episode (0, 0) ends before the last 10% and is left out. Episode (0, 1) reaches 1-2,
    # whose tiles count from its own start. Episode (1, 0) is furthest before the window.
    steps = pa.table(
        {
            "env_steps": [40, 92, 96, 100, 50, 95],
            "env_idx": [0, 0, 0, 0, 1, 1],
            "episode_id": [0, 1, 1, 1, 0, 0],
            "map_group": [1, 1, 1, 1, 1, 1],
            "map_num": [1, 1, 1, 2, 1, 1],
            "x": [40, 60, 70, 200, 90, 80],
        }
    )
    assert final_progress(steps, env_steps=100) == (70 + 90) / 2


def in_1_1(tables: dict[str, pa.Table]) -> dict[str, pa.Table]:
    """The fake logger walks FireRed maps; put its steps in world 1, level 1."""
    steps = tables["steps"]
    for name in ("map_group", "map_num"):
        column = pa.array([1] * steps.num_rows, pa.int16())
        steps = steps.set_column(steps.schema.get_field_index(name), name, column)
    return tables | {"steps": steps}


def write_run(run_dir: Path, tables: dict[str, pa.Table]) -> None:
    for name, table in tables.items():
        (run_dir / name).mkdir(parents=True)
        pq.write_table(table, run_dir / name / "w-0.parquet")


def test_main_reports_each_seed_and_the_iqm(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for seed in range(2):
        write_run(tmp_path / f"run{seed}", in_1_1(fake_run(f"fake{seed}", seed, 4096)))
    main(sorted(tmp_path.iterdir()))
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("fake0: seed 0, finished, schema ok, final level_progress")
    assert "nan" not in "\n".join(out)
    assert "4,096 steps" in out[1]
    assert out[2].startswith("final level_progress over finished seeds: IQM")
    assert out[2].endswith("(n=2)")


def test_main_refuses_a_table_outside_the_schema(tmp_path: Path) -> None:
    tables = fake_run("fake0", 0, 4096)
    write_run(tmp_path, tables | {"steps": tables["steps"].drop_columns(["x"])})
    with pytest.raises(SchemaError, match="missing"):
        main([tmp_path])
