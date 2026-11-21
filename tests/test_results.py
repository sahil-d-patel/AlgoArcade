"""The results freeze on fake-logger runs: manifest, checksum check and headline numbers."""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

pytest.importorskip("scipy")

from algoarcade import results
from algoarcade.logs import fake
from algoarcade.logs.writer import RollingWriter, run_dir

# 2 episodes per walker and 2 checkpoints of 2 evaluation episodes each, in a second.
ENV_STEPS = 4_096


def write_run(  # noqa: PLR0913 one fake run's identity
    root: Path,
    run_id: str,
    experiment: str,
    arm: str,
    seed: int,
    *,
    start_state: str = "s1",
    total_timesteps: int = 15_000_000,
    algorithm: str = "ppo",
    end_row: bool = True,
) -> None:
    """A fake-logger run relabelled as `experiment` and `arm`. Without its end row it reads as a
    run killed at the cutoff."""
    tables = fake.fake_run(run_id, seed, ENV_STEPS)
    runs = [
        row
        | {
            "experiment_id": experiment,
            "arm": arm,
            "start_state": start_state,
            "algorithm": algorithm,
            "hyperparameters": json.dumps({"total_timesteps": total_timesteps}),
        }
        for row in tables["runs"].to_pylist()
    ]
    tables["runs"] = pa.Table.from_pylist(
        runs if end_row else runs[:1], schema=tables["runs"].schema
    )
    episodes = [row | {"start_state": start_state} for row in tables["episodes"].to_pylist()]
    tables["episodes"] = pa.Table.from_pylist(episodes, schema=tables["episodes"].schema)
    for table, batch in tables.items():
        rows_per_file = 1 if table == "runs" else fake.ROWS_PER_FILE
        writer = RollingWriter(run_dir(run_id, root), table, "train", rows_per_file)
        for chunk in batch.to_batches(max_chunksize=rows_per_file):
            writer.write(pa.Table.from_batches([chunk]))
        writer.close()
    checkpoints = run_dir(run_id, root) / "checkpoints"
    checkpoints.mkdir()
    for step in (500_000, ENV_STEPS, 2 * ENV_STEPS):
        (checkpoints / f"step_{step}.zip").touch()


@pytest.fixture(scope="module")
def frozen(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """A runs folder as the box leaves it at the cutoff, and the results folder with its
    manifest."""
    root = tmp_path_factory.mktemp("runs")
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(fake, "WINDOW_STEPS", ENV_STEPS // 2)
        monkeypatch.setattr(fake, "EPISODE_STEPS", ENV_STEPS // 8)
        for seed in range(5):
            write_run(root, f"E1-ppo-seed{seed}-20261120-010000", "E1", "ppo", seed)
            write_run(root, f"E1-pqn-seed{seed}-20261120-020000", "E1", "pqn", seed)
        for seed in range(3):
            write_run(root, f"E5-positive-seed{seed}-20261120-030000", "E5", "positive", seed)
        # The queue names it E3-rnd-seed0; the owned trainer's run id adds -owned.
        write_run(root, "E3-rnd-owned-seed0-20261120-010000", "E3", "rnd", 0)
        brock = {"total_timesteps": 100_000_000}
        # A first attempt that crashed and was retried, and a run cut off with no end row.
        write_run(root, "E6-s1-seed0-20261119-010000", "E6", "s1", 0, **brock, end_row=False)
        write_run(root, "E6-s1-seed0-20261120-010000", "E6", "s1", 0, **brock)
        cut = {"start_state": "s0", "end_row": False}
        write_run(root, "E6-s0-seed0-20261120-010000", "E6", "s0", 0, **brock, **cut)
        llm = {"start_state": "seg_01_house", "algorithm": "llm"}
        write_run(root, "E7-haiku-L1-seg_01_house-00", "E7", "haiku-L1", 0, **llm)
    (root / "queue").mkdir()
    (root / "queue" / "queue.json").write_text(
        json.dumps(
            [
                {"name": "E3-rnd-seed0", "status": "stopped"},
                {"name": "E6-s1-seed0", "status": "stopped"},
                {"name": "E6-s0-seed0", "status": "stopped"},
                {"name": "E6-s0-seed1", "status": "stopped"},
            ]
        )
    )
    out = tmp_path_factory.mktemp("results-v1")
    results.main(["manifest", "--runs", str(root), "--out", str(out)])
    return root, out


def headline(frozen: tuple[Path, Path]) -> dict:
    root, out = frozen
    results.main(["headline", "--runs", str(root), "--out", str(out)])
    return json.loads((out / "headline.json").read_text())


def group(numbers: dict, experiment: str, arm: str) -> dict:
    (found,) = (g for g in numbers["groups"] if (g["experiment_id"], g["arm"]) == (experiment, arm))
    return found


def test_two_headline_runs_are_byte_identical(frozen: tuple[Path, Path]) -> None:
    root, out = frozen
    results.main(["headline", "--runs", str(root), "--out", str(out)])
    first = (out / "headline.json").read_bytes()
    # A second process, so set order and hash seeds differ.
    subprocess.run(
        [sys.executable, "-m", "algoarcade.results", "headline", "--runs", root, "--out", out],
        check=True,
    )

    assert hashlib.sha256((out / "headline.json").read_bytes()).hexdigest() == (
        hashlib.sha256(first).hexdigest()
    )


@pytest.mark.parametrize("change", ["modified", "missing", "added"])
def test_check_finds_a_changed_parquet_file(
    frozen: tuple[Path, Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    change: str,
) -> None:
    root, out = frozen
    copy = shutil.copytree(root, tmp_path / "runs")
    elsewhere = tmp_path / "results-v1"
    elsewhere.mkdir()
    shutil.copy(out / "manifest.json", elsewhere)
    episodes = run_dir("E1-pqn-seed3-20261120-020000", copy) / "episodes"
    (path,) = episodes.glob("*.parquet")
    if change == "modified":
        path.write_bytes(path.read_bytes() + b"\0")
    elif change == "missing":
        path.unlink()
    else:
        path = episodes / "late-00000.parquet"
        shutil.copy(next(episodes.glob("*.parquet")), path)

    with pytest.raises(SystemExit) as raised:
        results.main(["check", "--runs", str(copy), "--out", str(elsewhere)])

    assert raised.value.code == 1
    lines = capsys.readouterr().out.splitlines()
    assert f"E1-pqn-seed3-20261120-020000/episodes/{path.name}: {change}" in lines
    assert lines[-1].endswith(": 1 mismatches")
    with pytest.raises(SystemExit):
        results.main(["headline", "--runs", str(copy), "--out", str(elsewhere)])
    assert not (elsewhere / "headline.json").exists()


def test_check_passes_on_the_frozen_runs(
    frozen: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    root, out = frozen
    results.main(["check", "--runs", str(root), "--out", str(out)])

    assert capsys.readouterr().out.splitlines()[-1].endswith(": 0 mismatches")


def test_an_arm_under_five_seeds_is_exploratory(frozen: tuple[Path, Path]) -> None:
    numbers = headline(frozen)

    assert group(numbers, "E1", "pqn")["n"] == 5
    assert not group(numbers, "E1", "pqn")["exploratory"]
    assert group(numbers, "E5", "positive")["n"] == 3
    assert group(numbers, "E5", "positive")["exploratory"]


def test_comparisons_cover_arms_of_one_task_across_experiments(frozen: tuple[Path, Path]) -> None:
    # E2 to E4 queue only their new arm; the shared baseline arm runs under another id.
    pairs = {(c["x"], c["y"]): c for c in headline(frozen)["comparisons"]}

    assert set(pairs) == {
        ("E1/ppo", "E1/pqn"),
        ("E1/ppo", "E3/rnd"),
        ("E1/ppo", "E5/positive"),
        ("E1/pqn", "E3/rnd"),
        ("E1/pqn", "E5/positive"),
        ("E3/rnd", "E5/positive"),
    }
    assert pairs["E1/ppo", "E1/pqn"]["same_seeds"]
    assert not pairs["E1/ppo", "E1/pqn"]["exploratory"]
    assert not pairs["E1/ppo", "E5/positive"]["same_seeds"]
    assert pairs["E1/ppo", "E5/positive"]["exploratory"]


def test_every_e6_run_keeps_its_row(frozen: tuple[Path, Path]) -> None:
    numbers = headline(frozen)
    rows = [run for arm in ("s0", "s1") for run in group(numbers, "E6", arm)["runs"]]

    assert [r["run_id"] for r in rows] == [
        "E6-s0-seed0-20261120-010000",
        "E6-s1-seed0-20261120-010000",
    ]
    for row in rows:
        assert set(row) >= {"start_state", "seed", "steps", "furthest_milestone", "badge"}
        assert row["steps"] == ENV_STEPS
        assert isinstance(row["badge"], bool)
    assert [r["start_state"] for r in rows] == ["s0", "s1"]


def test_the_manifest_names_every_queued_run_and_marks_retries(frozen: tuple[Path, Path]) -> None:
    _, out = frozen
    manifest = json.loads((out / "manifest.json").read_text())
    entries = {e["run_id"]: e for e in manifest["runs"]}

    assert len(entries) == 18
    retried = entries["E6-s1-seed0-20261119-010000"]
    assert retried["superseded_by"] == "E6-s1-seed0-20261120-010000"
    assert retried["status"] == "running"
    cut = entries["E6-s0-seed0-20261120-010000"]
    assert (cut["queue_status"], cut["steps"]) == ("stopped", ENV_STEPS)
    assert cut["checkpoint"] == "checkpoints/step_500000.zip"
    assert all(len(digest) == 64 for e in entries.values() for digest in e["files"].values())
    assert manifest["left_out"] == [
        {"name": "E6-s0-seed1", "reason": "queue status stopped; no run directory"}
    ]
    assert entries["E3-rnd-owned-seed0-20261120-010000"]["queue_status"] == "stopped"


def test_a_badge_won_in_an_episode_left_open_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Seed 4 wins the badge and seed 0 does not. Seed 4 loses its badge episodes and gains a
    # step on milestone 9, as when the cutoff ends the run inside the episode that won it.
    monkeypatch.setattr(fake, "WINDOW_STEPS", ENV_STEPS // 2)
    monkeypatch.setattr(fake, "EPISODE_STEPS", ENV_STEPS // 8)
    for seed in (0, 4):
        write_run(tmp_path, f"E6-s1-seed{seed}-20261120-010000", "E6", "s1", seed, end_row=False)
    run = run_dir("E6-s1-seed4-20261120-010000", tmp_path)
    dropped = 0
    for path in (run / "episodes").glob("*.parquet"):
        table = pq.read_table(path)
        kept = table.filter(pc.equal(table["badge_count"], 0))
        dropped += len(table) - len(kept)
        pq.write_table(kept, path)
    steps = pq.read_table(run / "steps")
    last = steps.slice(len(steps) - 1).to_pylist()[0] | {"milestone_count": 9}
    pq.write_table(pa.Table.from_pylist([last], steps.schema), run / "steps" / "late.parquet")

    rows = [results.run_result(tmp_path, e)[0] for e in results.manifest(tmp_path)["runs"]]

    assert dropped > 0
    assert [(r["seed"], r["furthest_milestone"], r["badge"]) for r in rows] == [
        (0, 8, False),
        (4, 9, True),
    ]


def test_an_llm_attempt_scores_all_its_episodes(frozen: tuple[Path, Path]) -> None:
    root, _ = frozen
    (run,) = group(headline(frozen), "E7", "haiku-L1")["runs"]

    episodes = pq.read_table(run_dir(run["run_id"], root) / "episodes")["milestone_count"]
    assert run["milestones"] == round(sum(episodes.to_pylist()) / len(episodes), 4)
