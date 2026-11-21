"""Check every finished run for OPS-06: per-step logs, a checkpoint, and every Parquet file valid
under the OPS-03 schema (docs/harness-spec.md, Logging schema). Exits 1 if any run fails.

    uv run python scripts/check_runs.py [--runs-dir data/runs]
"""

from pathlib import Path

import pyarrow.parquet as pq
import tyro

from algoarcade.logs.schema import TABLES, SchemaError, validate


def main(runs_dir: Path = Path("data/runs")) -> int:
    finished = [
        run
        for run in sorted(runs_dir.glob("run_id=*"))
        if "finished" in pq.read_table(run / "runs", columns=["status"])["status"].to_pylist()
    ]
    failed = 0
    for run in finished:
        found = []
        if not any((run / "steps").glob("*.parquet")):
            found.append("no per-step logs")
        if not any((run / "checkpoints").glob("*.zip")):
            found.append("no checkpoint")
        for table in TABLES:
            for path in sorted((run / table).glob("*.parquet")):
                try:
                    validate(pq.read_table(path), table)
                except SchemaError as error:
                    found.append(f"{path.relative_to(run)}: {error}")
        failed += bool(found)
        for problem in found:
            print(f"{run.name.removeprefix('run_id=')}: {problem}")
    print(f"{len(finished)} finished runs checked, {failed} failed.")
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(tyro.cli(main))
