"""The TRN-07 parity gate: SB3 PPO against the owned trainer, one panel per task.

    uv run python scripts/parity.py --runs-dir data/runs/parity --out <png>

Each task is a folder of run directories from both trainers; a run directory with `-owned-` in
its name is the owned trainer's. The curve is the IQM over seeds of eval.csv's mean_return, with
a 95% percentile bootstrap band over seeds, both from algoarcade.training.stats (OPS-05). The gate passes on a task when the two bands
intersect at the last checkpoint and at 8 or more of 10 evenly spaced checkpoints.
"""

import csv
import json
from pathlib import Path

import matplotlib as mpl
import numpy as np
import tyro

from algoarcade.training import stats

mpl.use("Agg")
import matplotlib.pyplot as plt

TRAINERS = ("sb3", "owned")
POINTS = 10
NEEDED = 8


def curve(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """IQM and 95% bootstrap interval at each checkpoint (a column of seeds x checkpoints)."""
    bounds = np.array([stats.interval(column) for column in scores.T])
    return np.array([stats.iqm(column) for column in scores.T]), bounds[:, 0], bounds[:, 1]


def intersect(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    return (a[0] <= b[1]) & (b[0] <= a[1])


def read_runs(task: Path) -> dict[str, tuple[list[int], np.ndarray]]:
    """Per trainer: the checkpoint steps and a seeds x checkpoints array of mean returns. Both
    trainers must have run the same seeds and checkpointed at the same steps."""
    runs: dict[str, dict[int, tuple[list[int], list[float]]]] = {t: {} for t in TRAINERS}
    for run in sorted(task.glob("run_id=*")):
        seed = json.loads((run / "config.json").read_text())["seed"]
        with (run / "eval.csv").open() as f:
            rows = list(csv.DictReader(f))
        trainer = "owned" if "-owned-" in run.name else "sb3"
        runs[trainer][seed] = (
            [int(r["step"]) for r in rows],
            [float(r["mean_return"]) for r in rows],
        )
    if runs["sb3"].keys() != runs["owned"].keys():
        raise ValueError(
            f"{task.name}: seeds differ, {sorted(runs['sb3'])} against {sorted(runs['owned'])}"
        )
    steps = next(iter(runs["sb3"].values()))[0]
    if any(s != steps for by_seed in runs.values() for s, _ in by_seed.values()):
        raise ValueError(f"{task.name}: runs checkpoint at different steps")
    return {t: (steps, np.array([runs[t][seed][1] for seed in sorted(runs[t])])) for t in TRAINERS}


def main(runs_dir: Path, out: Path) -> None:
    tasks = sorted(p for p in runs_dir.iterdir() if p.is_dir())
    fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 4), squeeze=False)
    print(
        "| Task | Seeds (SB3, owned) | Final IQM SB3 | Final IQM owned | Bands meet | Last | Gate |"
    )
    print("|---|---|---|---|---|---|---|")
    for ax, task in zip(axes[0], tasks, strict=True):
        runs = read_runs(task)
        steps = runs["sb3"][0]
        bands, centres = {}, {}
        for trainer in TRAINERS:
            centres[trainer], low, high = curve(runs[trainer][1])
            bands[trainer] = (low, high)
            ax.plot(steps, centres[trainer], label=f"{trainer} (n={len(runs[trainer][1])})")
            ax.fill_between(steps, *bands[trainer], alpha=0.25)
        points = np.linspace(0, len(steps) - 1, POINTS).round().astype(int)
        meet = intersect(bands["sb3"], bands["owned"])
        hits, last = int(meet[points].sum()), bool(meet[-1])
        seeds = [len(runs[t][1]) for t in TRAINERS]
        label = "" if min(seeds) >= 5 else ", exploratory"
        gate = "pass" if last and hits >= NEEDED else "fail"
        finals = [
            f"{centres[t][-1]:.1f} [{bands[t][0][-1]:.1f}, {bands[t][1][-1]:.1f}]" for t in TRAINERS
        ]
        print(
            f"| {task.name} | {seeds[0]}, {seeds[1]} | {finals[0]} | {finals[1]} "
            f"| {hits} of {POINTS} | {'yes' if last else 'no'} | {gate}{label} |"
        )
        ax.set(
            title=f"{task.name}{label}",
            xlabel="env steps",
            ylabel="eval return, IQM and 95% bootstrap band",
        )
        ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)


if __name__ == "__main__":
    tyro.cli(main)
