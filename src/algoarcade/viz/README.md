# viz

Renderers and dashboard sources: the schematic Kanto map, the exploration heatmap, comparison panels, video rendering.

`render.py` plays one seeded episode of a checkpoint from the run's start state and writes an mp4 with the step, return, location, badges and one bar per reward component:

```
uv run python -m algoarcade.viz.render --checkpoint <run dir>/checkpoints/step_<n>.zip --seed 0 --out data/videos/<name>.mp4
```

`--grid 4` plays seeds `seed` to `seed + 3` side by side. It prints each episode's return and final location; the same checkpoint and seed give the same episode and the same file. Clips stay in `data/videos/`. The weekly review goes in `docs/video-review.md`.

`registry.py` joins each run's config to its results, one row per run, and prints its row count beside the number of `run_id=` folders, then any folder with no runs row:

```
uv run python -m algoarcade.viz.registry
```

`render_all.py` renders the final checkpoint of every registry row that has one, skips clips that exist, and prints `rendered, skipped, failed` counts:

```
uv run python -m algoarcade.viz.render_all
```

Contract: `docs/technical-plan.md`, Visualization.
