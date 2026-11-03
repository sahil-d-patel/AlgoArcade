"""Print a Markdown table of finished runs and save one learning-curve PNG per env.

uv run python scripts/summarize_runs.py --out-dir <dir>
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import gymnasium as gym
import matplotlib as mpl
import tyro

mpl.use("Agg")
import matplotlib.pyplot as plt


def main(out_dir: Path, runs_dir: Path = Path("data/runs")) -> None:
    curves = defaultdict(list)
    print("| Env | Seed | Steps | Eval return | Wall time (s) |")
    print("|---|---|---|---|---|")
    for run_dir in sorted(runs_dir.glob("run_id=*")):
        config = json.loads((run_dir / "config.json").read_text())
        with (run_dir / "eval.csv").open() as f:
            rows = list(csv.DictReader(f))
        final = rows[-1]
        print(
            f"| {config['env_id']} | {config['seed']} | {int(final['step']):,} "
            f"| {float(final['mean_return']):.1f} | {float(final['wall_time_s']):.0f} |"
        )
        steps = [int(row["step"]) for row in rows]
        returns = [float(row["mean_return"]) for row in rows]
        curves[config["env_id"]].append((config["seed"], steps, returns))

    out_dir.mkdir(parents=True, exist_ok=True)
    for env_id, runs in curves.items():
        fig, ax = plt.subplots(figsize=(6, 4))
        for seed, steps, returns in runs:
            ax.plot(steps, returns, label=f"seed {seed}")
        threshold = gym.spec(env_id).reward_threshold
        if threshold is not None:
            ax.axhline(threshold, color="gray", linestyle="--", label="threshold")
        ax.set(title=env_id, xlabel="env steps", ylabel="mean eval return (deterministic)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"{env_id}.png", dpi=120)
        plt.close(fig)


if __name__ == "__main__":
    tyro.cli(main)
