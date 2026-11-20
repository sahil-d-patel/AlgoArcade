# viz

Renderers and dashboard sources: the schematic Kanto map, the exploration heatmap, comparison panels, video rendering.

`render.py` plays one seeded episode of a checkpoint from the run's start state and writes an mp4 with the step, return, location, badges and one bar per reward component:

```
uv run python -m algoarcade.viz.render --checkpoint <run dir>/checkpoints/step_<n>.zip --seed 0 --out data/videos/<name>.mp4
```

`--grid 4` plays seeds `seed` to `seed + 3` side by side. It prints each episode's return and final location; the same checkpoint and seed give the same episode and the same file. `--eval-episode k` replays the k-th evaluation episode the runner logged at that checkpoint, sampled actions included, so a logged success can be clipped. Clips stay in `data/videos/`. The weekly review goes in `docs/video-review.md`.

`registry.py` joins each run's config to its results, one row per run, and prints its row count beside the number of `run_id=` folders, then any folder with no runs row:

```
uv run python -m algoarcade.viz.registry
```

`render_all.py` renders the final checkpoint of every registry row that has one, skips clips that exist, and prints `rendered, skipped, failed` counts:

```
uv run python -m algoarcade.viz.render_all
```

`dashboard/` is the Quarto site, rendered with `uv run poe dashboard`. `site.py` gives its pages a DuckDB view of the logs and runs before each render. `dashboard/README.md` says how to add a panel.

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
