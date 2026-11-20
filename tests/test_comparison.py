from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from algoarcade.logs import fake
from algoarcade.logs.writer import RollingWriter, run_dir
from algoarcade.viz import comparison, site

# A checkpoint every 8,192 fake steps instead of 500,000 gives four in a 40,000-step run.
CHECKPOINT = 8_192
ENV_STEPS = 40_000
pytestmark = pytest.mark.filterwarnings("ignore:.*seeds:UserWarning")


def write(  # noqa: PLR0913 the run's identity and what each test varies
    root: Path,
    experiment: str,
    arm: str,
    seed: int,
    bonus: int = 0,
    machine: str = "fake",
    finished: bool = True,
) -> None:
    """One fake-logger run, with `bonus` added to every evaluation milestone_count."""
    run_id = f"{experiment}-{arm}-seed{seed}"
    tables = fake.fake_run(run_id, seed, ENV_STEPS, experiment_id=experiment, arm=arm)
    episodes = tables["episodes"]
    bonus_on_eval = episodes["is_eval"].to_numpy() * bonus
    milestones = pa.array(episodes["milestone_count"].to_numpy() + bonus_on_eval, pa.int16())
    index = episodes.schema.get_field_index("milestone_count")
    tables["episodes"] = episodes.set_column(index, "milestone_count", milestones)
    runs = tables["runs"]
    runs = runs.set_column(runs.schema.get_field_index("machine"), "machine", [[machine] * 2])
    tables["runs"] = runs if finished else runs.slice(0, 1)
    for table, batch in tables.items():
        writer = RollingWriter(run_dir(run_id, root), table, "fake")
        writer.write(batch)
        writer.close()


@pytest.fixture(scope="module")
def root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """E1 with five baseline seeds, three pqn seeds 100 milestones above, and one pqn run that
    did not finish."""
    root = tmp_path_factory.mktemp("comparison") / "runs"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(fake, "WINDOW_STEPS", CHECKPOINT)
        for seed in range(5):
            write(root, comparison.BASELINE, "ppo", seed)
        for seed in range(3):
            write(root, "E1", "pqn", seed, bonus=100, machine="mac")
        write(root, "E1", "pqn", 3, finished=False)
    return root


def test_arm_above_in_every_seed_has_probability_of_improvement_one(root: Path) -> None:
    arms = comparison.load(site.connect(root), "E1")
    assert list(arms) == ["baseline", "pqn"]
    assert comparison.improvement(arms)["pqn"] == 1.0


def test_load_keeps_finished_runs_and_every_checkpoint(root: Path) -> None:
    arms = comparison.load(site.connect(root), "E1")
    assert arms["baseline"].seeds == [0, 1, 2, 3, 4]
    assert arms["pqn"].seeds == [0, 1, 2]
    assert arms["pqn"].machines == ["mac"]
    steps = [CHECKPOINT * k for k in range(1, ENV_STEPS // CHECKPOINT + 1)]
    assert arms["pqn"].steps.tolist() == steps
    assert arms["pqn"].scores.shape == (3, len(steps))
    assert (arms["pqn"].scores >= 100).all()
    # The fake logger runs at 500 steps per second.
    np.testing.assert_allclose(arms["pqn"].hours[0], np.array(steps) / 500 / 3600)


def test_table_lists_every_seed_and_labels_exploratory_arms(root: Path) -> None:
    con = site.connect(root)
    text = comparison.table(con, "E1")
    arms = comparison.load(con, "E1")
    assert "Finished runs: 8. Unfinished, left out: 1." in text
    [baseline] = [line for line in text.splitlines() if line.startswith("| baseline |")]
    [pqn] = [line for line in text.splitlines() if line.startswith("| pqn |")]
    assert "| 5 |" in baseline
    assert "| 3, exploratory |" in pqn
    assert "| 1.00 |" in pqn
    for seed, score in zip(arms["pqn"].seeds, arms["pqn"].scores[:, -1], strict=True):
        assert f"{seed}: {score:.2f}" in pqn
    assert "Seed sets differ" in text


def test_an_experiment_with_no_runs_shows_the_shared_baseline(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = comparison.table(site.connect(root), "E2").splitlines()
    assert [line.split(" |")[0] for line in rows if line.startswith("| ")] == [
        "| Arm",
        "| baseline",
    ]
    monkeypatch.setattr(comparison, "BASELINE", "none")
    assert "No finished runs." in comparison.table(site.connect(root), "E2")


def test_every_figure_names_each_arm_with_its_seed_count(root: Path) -> None:
    figures = comparison.figures("E1", comparison.load(site.connect(root), "E1"))
    assert len(figures) == 4
    for figure in figures:
        text = str(figure.to_dict())
        assert figure.to_dict()["layout"]["title"]["text"].startswith("E1: ")
        assert "baseline (n=5)" in text
        assert "pqn (n=3, exploratory)" in text


def test_wall_clock_axis_names_the_machines(root: Path) -> None:
    _, hours, _, _ = comparison.figures("E1", comparison.load(site.connect(root), "E1"))
    title = hours.to_dict()["layout"]["xaxis"]["title"]["text"]
    assert "baseline: fake" in title
    assert "pqn: mac" in title


def test_iqm_curve_draws_the_iqm_inside_its_band() -> None:
    rng = np.random.default_rng(0)
    scores = rng.normal(size=(5, 4)).cumsum(axis=1)
    figure = comparison.iqm_curve(
        {"level_1_1": ([1, 2, 3, 4], scores)}, "SML", "environment steps", "level_progress"
    )
    upper, lower, line = figure.select_traces()
    assert line["name"] == "level_1_1 (n=5)"
    assert not upper["showlegend"]
    assert not lower["showlegend"]
    assert (lower["y"] <= line["y"]).all()
    assert (line["y"] <= upper["y"]).all()
    assert figure.to_dict()["layout"]["xaxis"]["title"]["text"] == "environment steps"
