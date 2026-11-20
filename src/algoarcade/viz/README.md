# viz

Renderers and dashboard sources: the schematic Kanto map, the exploration heatmap, comparison panels, video rendering.

The video renderer arrived in week 4; the Kanto schematic, heatmap and recorded-progress timeline are the week 6 work. The dashboard (VIZ-01) builds on fake-logger data by Tuesday Nov 17, and every panel lands that week.

`render.py` plays one seeded episode of a checkpoint from the run's start state and writes an mp4 with the step, return, location, badges and one bar per reward component:

```
uv run python -m algoarcade.viz.render --checkpoint <run dir>/checkpoints/step_<n>.zip --seed 0 --out data/videos/<name>.mp4
```

`--grid 4` plays seeds `seed` to `seed + 3` side by side. It prints each episode's return and final location; the same checkpoint and seed give the same episode and the same file. Clips stay in `data/videos/`. The weekly review goes in `docs/video-review.md`.

Contract: `docs/technical-plan.md`, Visualization.

## Kanto heatmap and progress timeline

```sh
uv run python scripts/fetch_maps.py
uv run python -m algoarcade.viz.kanto_map --out /tmp/kanto.png
```

Terrain of the 37 maps ENV-10's `place()` connects to Pallet Town, with an inset panel for each map in `kanto_map.INSETS`.

```sh
uv run python -m algoarcade.logs.fake --run-id heatmap-demo --seed 0 --env-steps 1000040
uv run python -m algoarcade.viz.heatmap --run data/runs/run_id=heatmap-demo --out /tmp/heatmap.html
```

Exact visits per tile for one finished run, checked against its `env_steps`: `.html` has a slider frame per 500,000-step window, `.png` sums the run, and visits no panel can draw are counted in the title.

```sh
uv run python -m algoarcade.viz.timeline --runs data/runs/run_id=<seed 0> data/runs/run_id=<seed 1> --out /tmp/timeline.html
```

One row per seed. Schema v1 logs only `milestone_count`, so each point is the first sampled step at or beyond a stage, under `step_log_every` times `num_envs` steps after the true step.

`dashboard/heatmap.qmd` shows both, reading `data/runs/` or the folder in `ALGOARCADE_RUNS`.
