"""The results freeze (OPS-07): a manifest of every run's Parquet checksums, and the headline
numbers recomputed from that Parquet. Contract: docs/technical-plan.md, Experiments.

    uv run python -m algoarcade.results manifest   # writes docs/results-v1/manifest.json
    uv run python -m algoarcade.results check      # 0 mismatches, or exit 1
    uv run poe results                             # check, then write docs/results-v1/headline.json

Git ignores Parquet, so the manifest's checksums are the freeze. Every number in headline.json
comes from runs in the manifest whose files still match it.

A run's score comes from the evaluation episodes of its last checkpoint: `milestones` is their
mean milestone_count (E1's metric, fixed before any run) and `success` the share that reached
the start state's goal. An LLM attempt has no checkpoint; its episodes are scored. A group is
the runs of one (experiment, arm, start state), reported as the IQM of the scores with its 95%
bootstrap interval, `exploratory` under five runs. Every pair of groups on the same task, one
env, start state and step budget, gets the probability of improvement, so E2 to E4's arms meet
the shared baseline arm whatever experiment id it ran under.
"""

import argparse
import hashlib
import itertools
import json
import re
import warnings
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from algoarcade.logs.writer import run_dir
from algoarcade.training import stats

RUNS = Path("data/runs")
RESULTS = Path("docs/results-v1")
# The runner's run id ends in its launch time, so a retry of a queued run differs only there.
LAUNCHED = re.compile(r"(.+)-\d{8}-\d{6}")
METRICS = {"milestones": "milestone_count", "success": "goal_reached"}
DIGITS = 4


def file_sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def parquet_files(run: Path) -> dict[str, Path]:
    """Closed Parquet files only: an open one ends in .inprogress and cannot be read."""
    return {path.relative_to(run).as_posix(): path for path in sorted(run.rglob("*.parquet"))}


def read(run: Path, table: str, columns: list[str]) -> list[dict[str, Any]]:
    folder = run / table
    if not any(folder.glob("*.parquet")):
        return []
    return pq.read_table(folder, columns=columns).to_pylist()


def column_max(run: Path, table: str, column: str) -> int:
    """0 when the table has no closed file."""
    folder = run / table
    if not any(folder.glob("*.parquet")):
        return 0
    return int(pq.read_table(folder, columns=[column])[column].to_numpy().max(initial=0))


def current_row(run: Path) -> dict[str, Any]:
    """The end row if the run wrote one, else its start row."""
    rows = pq.read_table(run / "runs").to_pylist()
    return next((row for row in rows if row["status"] != "running"), rows[0])


def logged_steps(run: Path, row: dict[str, Any]) -> int:
    """A run killed without its end row still logged its steps up to its last closed file."""
    logged = (column_max(run, table, "env_steps") for table in ("steps", "episodes"))
    return max(row["env_steps"], *logged)


def last_checkpoint(run: Path) -> str | None:
    found = sorted(run.glob("checkpoints/step_*.zip"), key=lambda p: int(p.stem.split("_")[1]))
    return found[-1].relative_to(run).as_posix() if found else None


def manifest(root: Path) -> dict[str, Any]:
    queue = {
        entry["name"]: entry["status"]
        for path in sorted(root.glob("*/queue.json"))
        for entry in json.loads(path.read_text())
    }
    runs, left_out = [], []
    attempts: dict[str, list[dict[str, Any]]] = {}
    for run in sorted(root.glob("run_id=*")):
        run_id = run.name.removeprefix("run_id=")
        if not any((run / "runs").glob("*.parquet")):
            left_out.append({"name": run_id, "reason": "no runs table: not a logged game run"})
            continue
        row = current_row(run)
        launched = LAUNCHED.fullmatch(run_id)
        name = launched[1] if launched else run_id
        entry = {
            "run_id": run_id,
            "experiment_id": row["experiment_id"],
            "arm": row["arm"],
            "seed": row["seed"],
            "status": row["status"],
            "queue_status": queue.get(name),
            "steps": logged_steps(run, row),
            "git_commit": row["git_commit"],
            "checkpoint": last_checkpoint(run),
            "superseded_by": None,
            "files": {file: file_sha256(path) for file, path in parquet_files(run).items()},
        }
        runs.append(entry)
        attempts.setdefault(name, []).append(entry)
    for tries in attempts.values():
        for earlier in tries[:-1]:
            earlier["superseded_by"] = tries[-1]["run_id"]
    left_out += [
        {"name": name, "reason": f"queue status {status}; no run directory"}
        for name, status in sorted(queue.items())
        if name not in attempts
    ]
    return {"runs": runs, "left_out": left_out}


def mismatches(root: Path, frozen: dict[str, Any]) -> tuple[int, list[str]]:
    """The number of files checked and one line per file missing, changed or added."""
    checked, problems = 0, []
    for entry in frozen["runs"]:
        on_disk = parquet_files(run_dir(entry["run_id"], root))
        for name in sorted(entry["files"].keys() | on_disk.keys()):
            checked += 1
            if name not in on_disk:
                problem = "missing"
            elif name not in entry["files"]:
                problem = "added"
            elif file_sha256(on_disk[name]) != entry["files"][name]:
                problem = "modified"
            else:
                continue
            problems.append(f"{entry['run_id']}/{name}: {problem}")
    return checked, problems


def number(value: float) -> float:
    return round(float(value), DIGITS)


def mean(rows: list[dict[str, Any]], column: str) -> float | None:
    return number(sum(row[column] for row in rows) / len(rows)) if rows else None


def run_result(root: Path, entry: dict[str, Any]) -> tuple[dict[str, Any], tuple[Any, ...]]:
    """The run's row in headline.json, and its task: env, start state and step budget."""
    run = run_dir(entry["run_id"], root)
    row = current_row(run)
    columns = ["is_eval", "checkpoint_step", "milestone_count", "goal_reached", "badge_count"]
    episodes = read(run, "episodes", columns)
    evaluations = [e for e in episodes if e["is_eval"]]
    checkpoint = max((e["checkpoint_step"] for e in evaluations), default=None)
    if row["algorithm"] == "llm":
        scored = episodes
    else:
        scored = [e for e in evaluations if e["checkpoint_step"] == checkpoint]
    logged = (column_max(run, table, "milestone_count") for table in ("steps", "episodes"))
    budget = json.loads(row["hyperparameters"]).get("total_timesteps")
    result = {
        "run_id": entry["run_id"],
        "seed": entry["seed"],
        "start_state": row["start_state"],
        "status": entry["status"],
        "queue_status": entry["queue_status"],
        "steps": entry["steps"],
        "checkpoint_step": checkpoint,
        "furthest_milestone": max(logged),
        "badge": any(e["badge_count"] > 0 for e in episodes),
        **{metric: mean(scored, column) for metric, column in METRICS.items()},
    }
    return result, (row["env_id"], row["start_state"], budget)


def summary(scores: list[float]) -> dict[str, Any]:
    if not scores:
        return {"iqm": None, "ci": None}
    low, high = stats.interval(scores)
    return {"iqm": number(stats.iqm(scores)), "ci": [number(low), number(high)]}


def headline(root: Path, frozen: dict[str, Any], manifest_sha256: str) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    tasks: dict[tuple[str, str, str], tuple[Any, ...]] = {}
    for entry in frozen["runs"]:
        if entry["superseded_by"] is None:
            result, task = run_result(root, entry)
            key = (entry["experiment_id"], entry["arm"], result["start_state"])
            grouped.setdefault(key, []).append(result)
            tasks[key] = task
    groups: list[dict[str, Any]] = []
    scores: list[dict[str, list[float]]] = []
    for (experiment, arm, start_state), runs in sorted(grouped.items()):
        runs.sort(key=lambda r: (r["seed"], r["run_id"]))
        scored = [r for r in runs if r["milestones"] is not None]
        scores.append({m: [r[m] for r in scored] for m in METRICS})
        groups.append(
            {
                "experiment_id": experiment,
                "arm": arm,
                "start_state": start_state,
                "n": len(scored),
                "seeds": [r["seed"] for r in scored],
                "exploratory": len(scored) < stats.HEADLINE_SEEDS,
                **{m: summary(scores[-1][m]) for m in METRICS},
                "runs": runs,
            }
        )
    comparisons = []
    for (i, x_key), (j, y_key) in itertools.combinations(enumerate(sorted(grouped)), 2):
        x, y = groups[i], groups[j]
        if tasks[x_key] != tasks[y_key] or not (x["n"] and y["n"]):
            continue
        comparisons.append(
            {
                "x": f"{x['experiment_id']}/{x['arm']}",
                "y": f"{y['experiment_id']}/{y['arm']}",
                "start_state": x["start_state"],
                "p_x_beats_y": {
                    m: number(stats.probability_of_improvement(scores[i][m], scores[j][m]))
                    for m in METRICS
                },
                "same_seeds": sorted(x["seeds"]) == sorted(y["seeds"]),
                "exploratory": x["exploratory"] or y["exploratory"],
            }
        )
    return {"manifest_sha256": manifest_sha256, "groups": groups, "comparisons": comparisons}


def write_json(path: Path, value: Any) -> None:
    """Sorted keys and \\n line ends, so the same numbers give the same bytes on every OS."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def check(root: Path, out: Path) -> dict[str, Any]:
    frozen = json.loads((out / "manifest.json").read_text())
    checked, problems = mismatches(root, frozen)
    for problem in problems:
        print(problem)
    print(f"{checked} files in {len(frozen['runs'])} runs checked: {len(problems)} mismatches")
    if problems:
        raise SystemExit(1)
    return frozen


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m algoarcade.results", description=__doc__)
    parser.add_argument("command", choices=("manifest", "check", "headline"))
    parser.add_argument("--runs", type=Path, default=RUNS, help="The frozen runs folder.")
    parser.add_argument("--out", type=Path, default=RESULTS, help="Holds manifest.json.")
    args = parser.parse_args(argv)
    if args.command == "manifest":
        frozen = manifest(args.runs)
        write_json(args.out / "manifest.json", frozen)
        print(
            f"{len(frozen['runs'])} runs, {len(frozen['left_out'])} left out: "
            f"{args.out / 'manifest.json'}"
        )
        for entry in frozen["left_out"]:
            print(f"left out {entry['name']}: {entry['reason']}")
        return
    frozen = check(args.runs, args.out)
    if args.command == "headline":
        digest = file_sha256(args.out / "manifest.json")
        # stats warns under five seeds; headline.json labels those groups exploratory instead.
        with warnings.catch_warnings(action="ignore", category=UserWarning):
            numbers = headline(args.runs, frozen, digest)
        write_json(args.out / "headline.json", numbers)
        print(f"{len(numbers['groups'])} groups: {args.out / 'headline.json'}")


if __name__ == "__main__":
    main()
