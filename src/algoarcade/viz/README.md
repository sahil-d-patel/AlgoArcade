# viz

Renderers and dashboard sources: the schematic Kanto map, the exploration heatmap, comparison panels, video rendering.

Empty until week 6 on purpose, apart from the video renderer (OPS-05, Akshaj, week 4). The dashboard (VIZ-01) builds on fake-logger data by Tuesday Nov 17, and every panel lands that week.

`render.py` plays one seeded episode of a checkpoint from the run's start state and writes an mp4 with the step, return, location, badges and one bar per reward component:

```
uv run python -m algoarcade.viz.render --checkpoint <run dir>/checkpoints/step_<n>.zip --seed 0 --out data/videos/<name>.mp4
```

`--grid 4` plays seeds `seed` to `seed + 3` side by side. It prints each episode's return and final location; the same checkpoint and seed give the same episode and the same file. Clips stay in `data/videos/`. The weekly review goes in `docs/video-review.md`.

Contract: `docs/technical-plan.md`, Visualization.
