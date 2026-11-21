"""E1's table, verdict and figure: PPO against PQN on the metric experiments/e1_pqn.py names.

    uv run python scripts/e1.py --runs-dir <dir> --out <png>

<dir> holds the finished run directories of both arms; the runs table's `algorithm` column says
which arm a run is. A run's score is the mean milestone_count over the evaluation episodes of its
last checkpoint. The table is the OPS-05 one: IQM, 95% bootstrap interval and n per arm, then the
probability that a PQN run beats a PPO run. The figure has every seed's curve and each arm's IQM.

algoarcade.viz.comparison prints the same table for E1 beside E2 and E4. This script adds the
verdict line, the per-seed curves and a refusal when the arms ran different seeds, and it
tells the arms apart by `algorithm`, so it does not need the baseline's experiment id.
"""

from pathlib import Path

import matplotlib as mpl
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import tyro

from algoarcade.training import stats

mpl.use("Agg")
import matplotlib.pyplot as plt

ARMS = ("ppo", "pqn")
METRIC = "milestone_count"


def curve(run_dir: Path) -> tuple[str, int, dict[int, float]]:
    """The run's algorithm, seed and {checkpoint step: mean metric over its eval episodes}."""
    runs = pq.read_table(run_dir / "runs").to_pylist()
    if not any(r["status"] == "finished" for r in runs):
        raise ValueError(f"{run_dir.name} has not finished")
    episodes = pq.read_table(run_dir / "episodes", columns=["is_eval", "checkpoint_step", METRIC])
    evals = episodes.filter(pc.field("is_eval"))
    means = evals.group_by("checkpoint_step").aggregate([(METRIC, "mean")]).to_pylist()
    points = {r["checkpoint_step"]: r[f"{METRIC}_mean"] for r in means}
    return runs[0]["algorithm"], runs[0]["seed"], dict(sorted(points.items()))


def verdict(scores: dict[str, np.ndarray]) -> str:
    bounds = {arm: stats.interval(s) for arm, s in scores.items()}
    (ppo_low, ppo_high), (pqn_low, pqn_high) = bounds["ppo"], bounds["pqn"]
    numbers = ", ".join(f"{arm.upper()} IQM {stats.iqm(s):.2f}" for arm, s in scores.items())
    if pqn_low > ppo_high:
        return f"PQN won: its interval lies above PPO's ({numbers})."
    if pqn_high < ppo_low:
        return f"PQN lost: its interval lies below PPO's ({numbers})."
    return f"The intervals overlap: neither arm won ({numbers})."


def main(runs_dir: Path, out: Path) -> None:
    curves: dict[str, dict[int, dict[int, float]]] = {arm: {} for arm in ARMS}
    for run_dir in sorted(runs_dir.glob("run_id=*")):
        algorithm, seed, points = curve(run_dir)
        curves[algorithm][seed] = points
    if set(curves["ppo"]) != set(curves["pqn"]):
        seeds = {arm: sorted(curves[arm]) for arm in ARMS}
        raise ValueError(f"the arms ran different seeds: {seeds}")
    scores = {
        arm: np.array([points[max(points)] for points in curves[arm].values()]) for arm in ARMS
    }
    print(f"Metric: mean {METRIC} over the last checkpoint's evaluation episodes.\n")
    print("| Arm | IQM | 95% interval | n |")
    print("|---|---|---|---|")
    for arm in ARMS:
        low, high = stats.interval(scores[arm])
        print(
            f"| {arm.upper()} | {stats.iqm(scores[arm]):.2f} | [{low:.2f}, {high:.2f}] | "
            f"{len(scores[arm])} |"
        )
    print(f"\nP(PQN > PPO) {stats.probability_of_improvement(scores['pqn'], scores['ppo']):.2f}")
    print(verdict(scores))

    fig, ax = plt.subplots(figsize=(7, 4))
    for arm, color in zip(ARMS, ("tab:blue", "tab:orange"), strict=True):
        for points in curves[arm].values():
            ax.plot(list(points), list(points.values()), color=color, alpha=0.35, lw=1)
        steps = sorted(set.intersection(*(set(p) for p in curves[arm].values())))
        iqm = [stats.iqm([p[s] for p in curves[arm].values()]) for s in steps]
        ax.plot(steps, iqm, color=color, lw=2.5, label=f"{arm.upper()} IQM")
    ax.set_xlabel("env steps")
    ax.set_ylabel(f"eval {METRIC}")
    ax.set_title("E1: every seed, and the IQM per arm")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)


if __name__ == "__main__":
    tyro.cli(main)
