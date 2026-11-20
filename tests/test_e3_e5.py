from pathlib import Path

import pyarrow as pa
import pytest

from algoarcade.logs import fake
from algoarcade.logs.writer import RollingWriter, run_dir
from algoarcade.training import stats
from algoarcade.viz import e3_e5, site

# Two checkpoints. Seed 1 scores 8.5 at the first and 8.0 at the last.
ENV_STEPS = 2 * fake.WINDOW_STEPS


@pytest.fixture(scope="module")
def run() -> dict[str, pa.Table]:
    return fake.fake_run("E3-none", 1, ENV_STEPS, experiment_id="E3", arm="none")


def write(root: Path, run: dict[str, pa.Table], run_id: str, finished: bool) -> None:
    for table, batch in run.items():
        if table == "runs" and not finished:
            batch = batch.slice(0, 1)
        batch = batch.set_column(0, "run_id", pa.array([run_id] * batch.num_rows))
        writer = RollingWriter(run_dir(run_id, root), table, "fake")
        writer.write(batch)
        writer.close()


def test_score_is_the_last_checkpoints_mean_milestone_count_of_finished_runs(
    tmp_path: Path, run: dict[str, pa.Table]
) -> None:
    write(tmp_path, run, "done", finished=True)
    write(tmp_path, run, "running", finished=False)
    counts = [
        row["milestone_count"]
        for row in run["episodes"].to_pylist()
        if row["checkpoint_step"] == ENV_STEPS
    ]
    assert len(counts) == fake.EVAL_EPISODES
    expected = sum(counts) / len(counts)
    assert e3_e5.scores(site.connect(tmp_path), "E3", "none") == [(1, expected)]


def test_an_arm_above_in_every_seed_has_probability_of_improvement_1() -> None:
    low = [(seed, 1.0 + seed / 10) for seed in range(5)]
    high = [(seed, 3.0 + seed / 10) for seed in range(5)]
    text = e3_e5.table("E3", {"none": low, "count": high}, [("count", "none")])
    assert "| count over none | P(X > Y) 1.00 (n=5, m=5) |" in text
    low_ci, high_ci = stats.interval([score for _, score in high])
    by_seed = "0: 3.00, 1: 3.10, 2: 3.20, 3: 3.30, 4: 3.40"
    assert f"| count | 5 | 3.20 | [{low_ci:.2f}, {high_ci:.2f}] | {by_seed} |" in text
    assert "exploratory" not in text


def test_three_seeds_are_labelled_exploratory_and_report_the_mean() -> None:
    three = [(seed, float(seed)) for seed in range(3)]
    with pytest.warns(UserWarning, match="3 seeds"):
        text = e3_e5.table("E5", {"a": three, "b": three}, [("b", "a")])
    assert text.startswith("## E5, exploratory: under five seeds\n")
    assert "| Arm | n | Mean | 95% interval | Score by seed |" in text


def test_an_arm_with_no_finished_runs_is_listed_and_not_compared() -> None:
    five = [(seed, 1.0) for seed in range(5)]
    text = e3_e5.table("E3", {"none": five, "rnd": []}, [("rnd", "none")])
    assert "| rnd | 0 | | | |" in text
    assert "| rnd over none | no finished runs in rnd |" in text
