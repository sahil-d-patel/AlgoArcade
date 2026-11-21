"""E9's numbers: each run's schema check, final level_progress and throughput, then the IQM per arm.

    uv run python scripts/e9.py data/runs/run_id=E9-*

level_progress is counted in x_tile, the 8-pixel tile the steps table logs as x (docs/decisions.md,
2026-11-04). A run's final level_progress is the mean, over the training episodes that finished in
the last 10% of its env_steps, of each episode's furthest x_tile in 1-1. An episode the run's end
cut off is left out: it has no row in the episodes table. The steps table keeps every fourth
decision, so an episode's furthest tile can read a few tiles short.
"""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from algoarcade.logs.schema import TABLES, validate
from algoarcade.training import stats

FINAL_SHARE = 0.1


def validated(run_dir: Path) -> dict[str, pa.Table]:
    """Every table the run wrote, each file checked against the frozen schema. A run that
    crashed early can have only `runs`."""
    tables = {}
    for table in TABLES:
        files = sorted((run_dir / table).glob("*.parquet"))
        for file in files:
            validate(pq.read_table(file), table)
        if files:
            tables[table] = pa.concat_tables(pq.read_table(file) for file in files)
    return tables


def ints(table: pa.Table, *names: str) -> list[np.ndarray]:
    return [table[name].to_numpy().astype(np.int64) for name in names]


def final_progress(steps: pa.Table, episodes: pa.Table, env_steps: int) -> float:
    end, env_idx, episode_id = ints(episodes, "env_steps", "env_idx", "episode_id")
    late = ~episodes["is_eval"].to_numpy() & (end > (1 - FINAL_SHARE) * env_steps)
    finished = np.unique(env_idx[late] << 32 | episode_id[late])
    env_idx, episode_id, world, level, x = ints(
        steps, "env_idx", "episode_id", "map_group", "map_num", "x"
    )
    episode = env_idx << 32 | episode_id
    keep = (world == 1) & (level == 1) & np.isin(episode, finished)
    ids, index = np.unique(episode[keep], return_inverse=True)
    furthest = np.zeros(len(ids), np.int64)
    np.maximum.at(furthest, index, x[keep])
    return float(furthest.mean())


def main(run_dirs: list[Path]) -> None:
    scores: dict[str, list[float]] = defaultdict(list)
    dropped = []
    for run_dir in run_dirs:
        tables = validated(run_dir)
        end = tables["runs"].sort_by("wall_clock_s").to_pylist()[-1]
        line = f"{end['run_id']}: arm {end['arm']}, seed {end['seed']}, {end['status']}, schema ok"
        if end["status"] == "finished":
            score = final_progress(tables["steps"], tables["episodes"], end["env_steps"])
            scores[end["arm"]].append(score)
            rate = end["env_steps"] / end["wall_clock_s"]
            line += (
                f", final level_progress {score:.1f} x_tile, {end['env_steps']:,} steps"
                f" in {end['wall_clock_s'] / 3600:.2f} h, {rate:,.0f} steps/s"
            )
        else:
            dropped.append(end["run_id"])
        print(line)
    print(f"dropped {len(dropped)} of {len(run_dirs)} runs as not finished: {dropped}")
    for arm, arm_scores in scores.items():
        print(f"{arm}: final level_progress {stats.describe(arm_scores)}")


if __name__ == "__main__":
    main([Path(arg) for arg in sys.argv[1:]])
