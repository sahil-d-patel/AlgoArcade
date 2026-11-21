"""Check every finished run for OPS-06: per-step logs, a checkpoint, and every Parquet file valid
under the OPS-03 schema (docs/harness-spec.md, Logging schema). Exits 1 if any run fails.

    uv run python scripts/check_runs.py [--runs-dir data/runs]
"""

from pathlib import Path

import pyarrow.parquet as pq
import tyro

from algoarcade.logs.schema import TABLES, SchemaError, validate


def problems(run: Path, algorithm: str) -> list[str]:
    found = []
    if not any((run / "steps").glob("*.parquet")):
        found.append("no per-step logs")
    # SB3 and owned PPO save .zip, PQN .pt; an LLM run has none.
    checkpoints = (run / "checkpoints").glob("step_*")
    if algorithm != "llm" and not any(p.suffix in {".zip", ".pt"} for p in checkpoints):
        found.append("no checkpoint")
    for table in TABLES:
        for path in sorted((run / table).glob("*.parquet")):
            try:
                validate(pq.read_table(path), table)
            except SchemaError as error:
                found.append(f"{path.relative_to(run)}: {error}")
    return found


def main(runs_dir: Path = Path("data/runs")) -> int:
    finished = []
    for run in sorted(runs_dir.glob("run_id=*")):
        # A non-game run, or an attempt that died before its first row, has no runs table.
        if not (run / "runs").is_dir():
            continue
        rows = pq.read_table(run / "runs", columns=["status", "algorithm"]).to_pylist()
        if any(row["status"] == "finished" for row in rows):
            finished.append((run, rows[0]["algorithm"]))
    failed = 0
    for run, algorithm in finished:
        found = problems(run, algorithm)
        failed += bool(found)
        for problem in found:
            print(f"{run.name.removeprefix('run_id=')}: {problem}")
    print(f"{len(finished)} finished runs checked, {failed} failed.")
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(tyro.cli(main))
