# viz

Renderers and dashboard sources: the schematic Kanto map, the exploration heatmap, comparison panels, video rendering.

Empty until week 6 on purpose, apart from the video renderer (OPS-05, Akshaj, week 4). The dashboard (VIZ-01) builds on fake-logger data by Tuesday Nov 17, and every panel lands that week.

`render.py` plays one seeded episode of a checkpoint from the run's start state and writes an mp4 with the step, return, location, badges and one bar per reward component:

```
uv run python -m algoarcade.viz.render --checkpoint <run dir>/checkpoints/step_<n>.zip --seed 0 --out data/videos/<name>.mp4
```

`--grid 4` plays seeds `seed` to `seed + 3` side by side. It prints each episode's return and final location; the same checkpoint and seed give the same episode and the same file. Clips stay in `data/videos/`. The weekly review goes in `docs/video-review.md`.

Contract: `docs/technical-plan.md`, Visualization.

## Kanto terrain schematic

`kanto_map.py` draws the 37 mainland maps placed by ENV-10, with enlarged separate panels for Viridian Forest and Pewter Gym. Colors identify walkable ground, blocked tiles, grass, water and ledges. This is terrain only; it does not show visits or account for NPC blockers.

```sh
uv run python scripts/fetch_maps.py
uv run python -m algoarcade.viz.kanto_map --out .planning/kanto-preview.png
```

The output directory must already exist. Keep generated images local. The ticket's original count of 63 includes disconnected areas; this renderer reuses the existing 37-map placement without inventing connections.

## Visit heatmap preview

The static heatmap sums exact visit rows across environments and checkpoint windows for one finished run. It checks the total against the run's final step count, rejects mixed runs and invalid coordinates, and reports visits in areas outside the displayed panels. Zero-visit tiles keep their terrain colors; visited tiles use a shared logarithmic magma scale. Synthetic logs are labeled as demonstrations, not training results.

```sh
uv run python -m algoarcade.logs.fake --run-id heatmap-demo --seed 0 --env-steps 20000
uv run python -m algoarcade.viz.heatmap --run data/runs/run_id=heatmap-demo --out .planning/heatmap-preview.png
```

Generate a given demo run only once: the logger refuses to overwrite it. The PNG preview sums the whole run; use HTML output below for the training-time slider. The milestone timeline is not implemented yet.

## Interactive training windows

An `.html` output uses Plotly and embeds its JavaScript for offline viewing. Each slider position shows visits within that window, not cumulative visits. All frames share one logarithmic color scale. Hover gives the map, local tile coordinates and exact visit count. Adjacent 500,000-step windows are combined only when needed to stay at or below 50 frames; the title shows the actual range, including a partial final window. Counts must match both the finished run and every original window. These are raw counts, not visit rates: a partial final window contains fewer decisions and should not be interpreted as reduced exploration by itself.

```sh
uv run python -m algoarcade.logs.fake --run-id heatmap-slider-demo --seed 0 --env-steps 1000040
uv run python -m algoarcade.viz.heatmap --run data/runs/run_id=heatmap-slider-demo --out .planning/heatmap-slider.html
```

For the dashboard, export to `src/algoarcade/viz/dashboard/heatmap-view.html` instead. The adjacent `heatmap.qmd` embeds that file and lists it as a Quarto resource, so the export travels with the rendered page. The generated HTML is ignored by git. Add `heatmap.qmd` to the dashboard navigation when its skeleton is ready. No milestone timeline or real baseline validation is included yet.
