# dashboard

The Quarto site that shows every panel. Owner: Akshaj (VIZ-01). It renders to static HTML that opens from disk with no network. Contract: `docs/technical-plan.md`, Visualization.

```sh
uv run python -m algoarcade.logs.fake --run-id fake0 --seed 0   # once, if data/runs/ is empty
uv run poe dashboard
```

Then open `src/algoarcade/viz/dashboard/_site/index.html` in a browser.

`uv run poe dashboard` runs `uv run --group dashboard quarto render` on this folder. The first time, that installs the `dashboard` group into `.venv`; nothing else needs installing by hand. The group is opt-in because `quarto-cli` downloads Quarto from GitHub when it installs (139 MB on Linux, 141 MB on Windows, 237 MB on macOS) and pulls in 76 more packages, mostly Jupyter. Members who never render the site, and CI, skip all of it. The next plain `uv sync` removes the group again; `poe dashboard` puts it back.

Before rendering, Quarto runs `python -m algoarcade.viz.site`, which reads every run under `data/runs/` and writes:

- `_variables.yml`: the git commit, the render date and the number of runs read, shown in every page's footer;
- `plotly.min.js`, which every page's header loads;
- `videos/`: a copy of `data/videos/`, plus `walk.mp4`, a clip of the first run's first logged episode.

These, `_site/` and `.quarto/` are git-ignored. Run the command through `uv run`: the page code runs on the project's Python.

Each page is one HTML file with its scripts, styles and plotly.js inlined (`embed-resources`), because Chrome will not load Quarto's module scripts from disk. Videos stay separate files in `_site/videos/`. Copy the whole `_site/` folder, not one page.

## Add a panel

1. Put the figure code in a module next to this folder, for example `src/algoarcade/viz/my_panel.py`, as a function that takes a DuckDB connection and returns a Plotly figure. Test it on fake-logger data.
2. Create `<page>.qmd` at the top of this folder, not in a subfolder: the header loads `plotly.min.js` from beside the page.

   ````markdown
   ---
   title: Comparison
   ---

   One sentence on what the figure shows and how many seeds it uses.

   ```{python}
   #| output: asis
   from algoarcade.viz.my_panel import my_figure
   from algoarcade.viz.site import connect, show

   show(my_figure(connect()))
   ```
   ````

   `connect()` gives one view per logged table: `steps`, `visits`, `episodes`, `runs`. The `runs` view has a start row and, once the run finishes, an end row for each run. `show()` with `output: asis` is required. A figure left for Quarto to display loads Plotly, jQuery and require.js from CDNs, and the page breaks offline.
3. For a clip, put the mp4 under `data/videos/` (the renderer writes there) and add `<video controls src="videos/<path>.mp4" data-external="1"></video>` to the page. Without `data-external="1"` the clip is inlined into the page as base64. Quarto's `{{< video >}}` shortcode inlined it that way, and Chrome would not play it. Never commit an mp4.
4. Add the page to `website.navbar.left` in `_quarto.yml`.
5. Run `uv run poe dashboard`. It must exit 0. Then check your page offline: turn Wi-Fi off, open `_site/<page>.html`, hover a figure, play each clip, and look for failed requests in the browser console.

No `http://` or `https://` script, stylesheet, image or iframe may appear in a page. A W&B chart stands in as an exported image, never an iframe. Do not use `scattergeo`, `choropleth` or Plotly's map traces: plotly.js fetches their topojson and tiles from the web at runtime.
