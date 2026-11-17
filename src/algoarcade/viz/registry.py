"""The run registry: one row per run, joining its config to its results, so a reader can go from a
figure to the run that produced it. Contract: docs/technical-plan.md, Visualization, and
docs/harness-spec.md, Logging schema.

    uv run python -m algoarcade.viz.registry
"""

from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.dataset as ds
import tyro

from algoarcade.logs.schema import EPISODES, RUNS

ROOT = Path("data/runs")

# A run's current row is its end row if one exists, else its start row (harness-spec, Per-run
# table). Its final score is the mean evaluation return at its last evaluated checkpoint, and its
# furthest milestone the best milestone count any of its episodes reached.
QUERY = """
WITH current AS (
    SELECT * FROM runs
    QUALIFY row_number() OVER (
        PARTITION BY run_id ORDER BY status <> 'running' DESC, env_steps DESC
    ) = 1
),
last_evals AS (
    SELECT run_id, episode_return
    FROM episodes
    WHERE is_eval
    QUALIFY checkpoint_step = max(checkpoint_step) OVER (PARTITION BY run_id)
),
scores AS (
    SELECT run_id, avg(episode_return) AS final_score FROM last_evals GROUP BY run_id
),
milestones AS (
    SELECT run_id, max(milestone_count) AS furthest_milestone FROM episodes GROUP BY run_id
)
SELECT
    run_id,
    experiment_id AS experiment,
    arm,
    seed,
    git_commit,
    algorithm,
    env_id,
    start_state,
    obs_mode,
    reward_config_hash,
    env_steps AS steps,
    status,
    final_score,
    furthest_milestone,
    wall_clock_s / 3600 AS wall_clock_hours
FROM current
LEFT JOIN scores USING (run_id)
LEFT JOIN milestones USING (run_id)
ORDER BY experiment, arm, seed, run_id
"""


def _logs(root: Path, table: str, schema: pa.Schema) -> ds.Dataset:
    # Closed files only: an open one ends in .parquet.inprogress (harness-spec, Logging schema).
    files = [str(path) for path in sorted(root.glob(f"run_id=*/{table}/*.parquet"))]
    return ds.dataset(files, schema=schema, format="parquet")


def registry(root: Path = ROOT) -> pa.Table:
    con = duckdb.connect()
    con.register("runs", _logs(root, "runs", RUNS))
    con.register("episodes", _logs(root, "episodes", EPISODES))
    return con.sql(QUERY).to_arrow_table()


def main(root: Path = ROOT) -> None:
    """Print the registry's row count beside the number of run folders; they should match."""
    rows = registry(root).num_rows
    folders = sum(1 for path in root.glob("run_id=*") if path.is_dir())
    print(f"registry rows: {rows}, run_id= folders: {folders}")  # noqa: T201


if __name__ == "__main__":
    tyro.cli(main)
