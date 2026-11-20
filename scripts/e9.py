"""E9's numbers: each run's schema check, final level_progress and throughput, then the IQM.

    uv run python scripts/e9.py data/runs/run_id=E9-untouched-seed*

level_progress is counted in x_tile, the 8-pixel tile the steps table logs as x (docs/decisions.md,
2026-11-04). A run's final level_progress is the mean, over the training episodes logged in the
last 10% of its env_steps, of each episode's furthest x_tile in 1-1. The steps table keeps every
fourth decision, so an episode's furthest tile can read a few tiles short.
"""

import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from algoarcade.logs.schema import TABLES, validate
from algoarcade.training import stats

FINAL_SHARE = 0.1


def validated(run_dir: Path) -> dict[str, pa.Table]:
    """Every table of the run, each file checked against the frozen schema."""
    tables = {}
    for table in TABLES:
        files = sorted((run_dir / table).glob("*.parquet"))
        for file in files:
            validate(pq.read_table(file), table)
        tables[table] = pa.concat_tables(pq.read_table(file) for file in files)
    return tables


def final_progress(steps: pa.Table, env_steps: int) -> float:
    names = ("env_steps", "env_idx", "episode_id", "map_group", "map_num", "x")
    column = {name: steps[name].to_numpy().astype(np.int64) for name in names}
    episode = column["env_idx"] << 32 | column["episode_id"]
    late = np.unique(episode[column["env_steps"] > (1 - FINAL_SHARE) * env_steps])
    keep = (column["map_group"] == 1) & (column["map_num"] == 1) & np.isin(episode, late)
    ids, index = np.unique(episode[keep], return_inverse=True)
    furthest = np.zeros(len(ids), np.int64)
    np.maximum.at(furthest, index, column["x"][keep])
    return float(furthest.mean())


def main(run_dirs: list[Path]) -> None:
    scores = []
    for run_dir in run_dirs:
        tables = validated(run_dir)
        end = tables["runs"].sort_by("wall_clock_s").to_pylist()[-1]
        line = f"{end['run_id']}: seed {end['seed']}, {end['status']}, schema ok"
        if end["status"] == "finished":
            scores.append(final_progress(tables["steps"], end["env_steps"]))
            rate = end["env_steps"] / end["wall_clock_s"]
            line += (
                f", final level_progress {scores[-1]:.1f} x_tile, {end['env_steps']:,} steps"
                f" in {end['wall_clock_s'] / 3600:.2f} h, {rate:,.0f} steps/s"
            )
        print(line)
    print(f"final level_progress over finished seeds: {stats.describe(scores)}")


if __name__ == "__main__":
    main([Path(arg) for arg in sys.argv[1:]])
