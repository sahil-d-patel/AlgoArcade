import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytest.importorskip("matplotlib")

from e1 import main, verdict

from algoarcade.logs import fake
from algoarcade.logs.schema import RUNS

SEEDS = range(5)


def write_runs(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Five fake runs per arm, each with evaluations at 4,096 and 8,192 steps."""
    monkeypatch.setattr(fake, "WINDOW_STEPS", 4_096)
    for algorithm in ("ppo", "pqn"):
        for seed in SEEDS:
            run_id = f"E1-{algorithm}-seed{seed}"
            tables = fake.fake_run(run_id, 100 * (algorithm == "pqn") + seed, 8_192)
            rows = tables["runs"].to_pylist()
            for row in rows:
                row |= {"algorithm": algorithm, "seed": seed}
            tables["runs"] = pa.Table.from_pylist(rows, schema=RUNS)
            for name in ("runs", "episodes"):
                (root / f"run_id={run_id}" / name).mkdir(parents=True)
                pq.write_table(tables[name], root / f"run_id={run_id}" / name / "part-0.parquet")


def test_table_verdict_and_figure_from_ten_fake_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    write_runs(tmp_path / "runs", monkeypatch)
    main(tmp_path / "runs", tmp_path / "e1.png")
    out = capsys.readouterr().out.splitlines()
    assert out[2:4] == ["| Arm | IQM | 95% interval | n |", "|---|---|---|---|"]
    assert out[4].startswith("| PPO | ")
    assert out[4].endswith(" | 5 |")
    assert out[5].startswith("| PQN | ")
    assert out[7].startswith("P(PQN > PPO) ")
    assert out[8].startswith(("PQN won", "PQN lost", "The intervals overlap"))
    assert (tmp_path / "e1.png").stat().st_size > 0


def test_unequal_seeds_are_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_runs(tmp_path / "runs", monkeypatch)
    shutil.rmtree(tmp_path / "runs" / "run_id=E1-pqn-seed4")
    with pytest.raises(ValueError, match="different seeds"):
        main(tmp_path / "runs", tmp_path / "e1.png")


@pytest.mark.parametrize(
    ("pqn", "start"),
    [
        ([5.0, 5.1, 5.2, 5.3, 5.4], "PQN won"),
        ([1.0, 1.1, 1.2, 1.3, 1.4], "PQN lost: its interval lies below PPO's (PPO IQM 3.20, PQN "),
        ([2.9, 3.1, 3.2, 3.3, 3.5], "The intervals overlap"),
    ],
)
def test_verdict(pqn: list[float], start: str) -> None:
    ppo = np.array([3.0, 3.1, 3.2, 3.3, 3.4])
    assert verdict({"ppo": ppo, "pqn": np.array(pqn)}).startswith(start)
