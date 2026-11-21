from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from e9 import final_progress, main

from algoarcade.logs.fake import fake_run
from algoarcade.logs.schema import SchemaError


def test_final_progress_takes_finished_late_episodes_furthest_tile_in_1_1() -> None:
    # Training episode (0, 0) ends before the last 10% and is left out. (0, 1) reaches 1-2,
    # whose tiles count from its own start. (1, 0) is furthest before the window. (1, 1) was cut
    # off by the run's end: an evaluation episode shares its ids, but it has no training row.
    steps = pa.table(
        {
            "env_idx": [0, 0, 0, 0, 1, 1, 1],
            "episode_id": [0, 1, 1, 1, 0, 0, 1],
            "map_group": [1, 1, 1, 1, 1, 1, 1],
            "map_num": [1, 1, 1, 2, 1, 1, 1],
            "x": [40, 60, 70, 200, 90, 80, 150],
        }
    )
    episodes = pa.table(
        {
            "env_steps": [40, 100, 95, 100],
            "env_idx": [0, 0, 1, 1],
            "episode_id": [0, 1, 0, 1],
            "is_eval": [False, False, False, True],
        }
    )
    assert final_progress(steps, episodes, env_steps=100) == (70 + 90) / 2


# Ten full 2,048-step episodes per fake env, then a partial one the run's end cuts off.
FAKE_STEPS = 4 * 2048 * 10 + 1024


def as_e9(tables: dict[str, pa.Table], arm: str) -> dict[str, pa.Table]:
    """The fake logger walks FireRed maps; put its steps in world 1, level 1, and name its arm."""
    steps, runs = tables["steps"], tables["runs"]
    for name in ("map_group", "map_num"):
        column = pa.array([1] * steps.num_rows, pa.int16())
        steps = steps.set_column(steps.schema.get_field_index(name), name, column)
    runs = runs.set_column(runs.schema.get_field_index("arm"), "arm", pa.array([arm] * 2))
    return tables | {"steps": steps, "runs": runs}


def write_run(run_dir: Path, tables: dict[str, pa.Table]) -> None:
    for name, table in tables.items():
        (run_dir / name).mkdir(parents=True)
        pq.write_table(table, run_dir / name / "w-0.parquet")


def test_main_reports_each_arm_and_the_runs_it_dropped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for seed, arm in enumerate(("hold16", "hold16", "hold8")):
        write_run(tmp_path / f"run{seed}", as_e9(fake_run(f"fake{seed}", seed, FAKE_STEPS), arm))
    status = pa.array(["running", "crashed"])
    for seed in (3, 4):
        crashed = as_e9(fake_run(f"fake{seed}", seed, FAKE_STEPS), "hold8")
        runs = crashed["runs"].set_column(
            crashed["runs"].schema.get_field_index("status"), "status", status
        )
        # fake4 crashed before its first rollout, so it wrote only the runs table.
        write_run(
            tmp_path / f"run{seed}", crashed | {"runs": runs} if seed == 3 else {"runs": runs}
        )
    main(sorted(tmp_path.iterdir()))
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("fake0: arm hold16, seed 0, finished, schema ok, final level_progress")
    assert f"{FAKE_STEPS:,} steps" in out[0]
    assert out[3] == "fake3: arm hold8, seed 3, crashed, schema ok"
    assert out[4] == "fake4: arm hold8, seed 4, crashed, schema ok"
    assert out[5] == "dropped 2 of 5 runs as not finished: ['fake3', 'fake4']"
    assert out[6].startswith("hold16: final level_progress IQM")
    assert out[6].endswith("(n=2)")
    assert out[7].endswith("(n=1)")
    assert "nan" not in "\n".join(out)


def test_main_refuses_a_table_outside_the_schema(tmp_path: Path) -> None:
    tables = fake_run("fake0", 0, 4096)
    write_run(tmp_path, tables | {"steps": tables["steps"].drop_columns(["x"])})
    with pytest.raises(SchemaError, match="missing"):
        main([tmp_path])
