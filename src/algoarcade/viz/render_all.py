"""Render the final checkpoint of every run in the registry that has one, one clip per run, for the
dashboard. A clip that already exists is skipped, so a second run renders nothing new. Contract:
docs/technical-plan.md, Visualization.

    uv run python -m algoarcade.viz.render_all

Never start it on the training box while final runs are going: queue it, or run it on a laptop
with copied checkpoints.
"""

from pathlib import Path

import tyro

from algoarcade.viz.registry import ROOT, registry
from algoarcade.viz.render import render

VIDEOS = Path("data/videos")


def final_checkpoint(run_dir: Path) -> Path | None:
    """The run's checkpoint with the most steps, compared as numbers: step_500000 sorts after
    step_1000000 as text."""
    checkpoints = {
        int(path.stem.removeprefix("step_")): path
        for path in (run_dir / "checkpoints").glob("step_*.zip")
    }
    return checkpoints[max(checkpoints)] if checkpoints else None


def main(root: Path = ROOT, out: Path = VIDEOS) -> None:
    """Render every registry row's final checkpoint into `out` and print the counts."""
    rendered, skipped, failed = 0, 0, []
    for run_id in registry(root).column("run_id").to_pylist():
        checkpoint = final_checkpoint(root / f"run_id={run_id}")
        if checkpoint is None:
            continue
        clip = out / f"{run_id}-{checkpoint.stem}.mp4"
        if clip.exists():
            skipped += 1
            continue
        # Rendered aside and moved in when done: an interrupted render must not leave a clip
        # the next run would skip.
        partial = out / "partial" / clip.name
        try:
            render(checkpoint, partial)
        except Exception as error:  # noqa: BLE001 one broken run must not stop the batch
            failed.append(f"{run_id}: {error}")
            continue
        partial.replace(clip)
        rendered += 1
    for line in failed:
        print(f"failed {line}")  # noqa: T201
    print(f"rendered {rendered}, skipped {skipped}, failed {len(failed)}")  # noqa: T201


if __name__ == "__main__":
    tyro.cli(main)
