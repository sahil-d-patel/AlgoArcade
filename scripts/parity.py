"""The TRN-07 parity gate: SB3 PPO against the owned trainer, one panel per task.

    uv run python scripts/parity.py --runs-dir data/runs/parity --out <png>

Each task is a folder of run directories from both trainers; a run directory with `-owned-` in
its name is the owned trainer's. The curve is the IQM over seeds of eval.csv's mean_return, with
a 95% percentile bootstrap band over seeds. The gate passes on a task when the two bands
intersect at the last checkpoint and at 8 or more of 10 evenly spaced checkpoints.
"""

import csv
from pathlib import Path

import matplotlib as mpl
import numpy as np
import tyro

mpl.use("Agg")
import matplotlib.pyplot as plt

TRAINERS = ("sb3", "owned")
POINTS = 10
NEEDED = 8


def iqm(scores: np.ndarray) -> np.ndarray:
    """The interquartile mean over axis 0: scipy.stats.trim_mean(scores, 0.25), which cuts
    int(0.25 * n) values from each end. Below four seeds it cuts nothing and is the mean."""
    n = len(scores)
    cut = int(0.25 * n)
    return np.sort(scores, axis=0)[cut : n - cut].mean(axis=0)


def bootstrap(
    scores: np.ndarray, resamples: int = 2000, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """A 95% percentile interval of the IQM, resampling whole seeds (rows) with replacement."""
    rng = np.random.default_rng(seed)
    rows = rng.integers(len(scores), size=(resamples, len(scores)))
    estimates = np.stack([iqm(scores[r]) for r in rows])
    low, high = np.percentile(estimates, [2.5, 97.5], axis=0)
    return low, high


def intersect(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    return (a[0] <= b[1]) & (b[0] <= a[1])


def read_runs(task: Path) -> dict[str, tuple[list[int], np.ndarray]]:
    """Per trainer: the checkpoint steps and a seeds x checkpoints array of mean returns."""
    curves: dict[str, list[tuple[list[int], list[float]]]] = {t: [] for t in TRAINERS}
    for run in sorted(task.glob("run_id=*")):
        with (run / "eval.csv").open() as f:
            rows = list(csv.DictReader(f))
        trainer = "owned" if "-owned-" in run.name else "sb3"
        curves[trainer].append(
            ([int(r["step"]) for r in rows], [float(r["mean_return"]) for r in rows])
        )
    out = {}
    for trainer, runs in curves.items():
        steps = runs[0][0]
        if any(s != steps for s, _ in runs):
            raise ValueError(f"{task}: {trainer} runs checkpoint at different steps")
        out[trainer] = (steps, np.array([r for _, r in runs]))
    return out


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
        if runs["owned"][0] != steps:
            raise ValueError(f"{task.name}: the trainers checkpoint at different steps")
        bands = {}
        for trainer in TRAINERS:
            scores = runs[trainer][1]
            bands[trainer] = bootstrap(scores)
            ax.plot(steps, iqm(scores), label=f"{trainer} (n={len(scores)})")
            ax.fill_between(steps, *bands[trainer], alpha=0.25)
        points = np.linspace(0, len(steps) - 1, POINTS).round().astype(int)
        meet = intersect(bands["sb3"], bands["owned"])
        hits, last = int(meet[points].sum()), bool(meet[-1])
        seeds = [len(runs[t][1]) for t in TRAINERS]
        label = "" if min(seeds) >= 5 else ", exploratory"
        gate = "pass" if last and hits >= NEEDED else "fail"
        finals = [
            f"{iqm(runs[t][1])[-1]:.1f} [{bands[t][0][-1]:.1f}, {bands[t][1][-1]:.1f}]"
            for t in TRAINERS
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
