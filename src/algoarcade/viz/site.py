"""The dashboard's data access, its overview panel and its pre-render step. Contract:
docs/technical-plan.md, Visualization. How to add a page: src/algoarcade/viz/dashboard/README.md.

Quarto runs `python -m algoarcade.viz.site` before every render (dashboard/_quarto.yml). It writes
the footer values to `_variables.yml`, the site's copy of plotly.js, and `videos/`: everything in
data/videos/ plus a clip of the first run's first logged episode.
"""

import os
import shutil
import subprocess
from datetime import date
from pathlib import Path

import duckdb
import imageio_ffmpeg
import numpy as np
import plotly.graph_objects as go
import pyarrow.dataset as ds
from PIL import Image, ImageDraw
from plotly.offline import get_plotlyjs

from algoarcade.logs.schema import TABLES

# Anchored to the checkout: Quarto runs each page's cells from the page's own folder.
RUNS = Path(os.environ.get("ALGOARCADE_RUNS", Path(__file__).resolve().parents[3] / "data/runs"))
DASHBOARD = Path(__file__).parent / "dashboard"
SCALE = 4
FPS = 15


def connect(root: Path = RUNS) -> duckdb.DuckDBPyConnection:
    """DuckDB with one view per logged table (steps, visits, episodes, runs) over the closed
    Parquet files under `root`. The runs view holds a start row and, once finished, an end row
    per run."""
    con = duckdb.connect()
    for table, schema in TABLES.items():
        # Closed files only: an open one ends in .parquet.inprogress.
        files = [str(path) for path in sorted(root.glob(f"run_id=*/{table}/*.parquet"))]
        con.register(table, ds.dataset(files, schema=schema, format="parquet"))
    return con


def show(fig: go.Figure) -> None:
    """Prints `fig` as raw HTML, for a page cell marked `#| output: asis`. A figure Quarto
    displays itself loads plotly.js, jQuery and require.js from CDNs. This one uses the
    plotly.min.js every page's header loads (_quarto.yml)."""
    html = fig.to_html(full_html=False, include_plotlyjs=False)
    print(f"```{{=html}}\n{html}\n```")  # noqa: T201


def returns_figure(con: duckdb.DuckDBPyConnection) -> go.Figure:
    fig = go.Figure(
        layout={
            "title": "Training episode return",
            "xaxis_title": "environment steps",
            "yaxis_title": "episode return",
        }
    )
    rows = con.sql(
        """
        SELECT run_id, list(env_steps ORDER BY env_steps, env_idx),
            list(episode_return ORDER BY env_steps, env_idx)
        FROM episodes WHERE NOT is_eval GROUP BY run_id ORDER BY run_id
        """
    ).fetchall()
    for run_id, steps, returns in rows:
        fig.add_scatter(x=steps, y=returns, mode="markers", name=run_id)
    return fig


def walk_clip(con: duckdb.DuckDBPyConnection, out: Path) -> None:
    """One frame per logged step of the first run's env 0, episode 0: the tile as a square on its
    map, with the map and step written above it."""
    rows = con.sql(
        """
        SELECT map_group, map_num, x, y, episode_step FROM steps
        WHERE run_id = (SELECT min(run_id) FROM steps) AND env_idx = 0 AND episode_id = 0
        ORDER BY episode_step
        """
    ).fetchall()
    # Sizes rounded up to 16 so libx264 does not rescale.
    width = -(-(max(row[2] for row in rows) + 1) * SCALE // 16) * 16
    height = -(-(max(row[3] for row in rows) + 1) * SCALE // 16) * 16 + 16
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(
        str(out), (width, height), fps=FPS, codec="libx264", pix_fmt_out="yuv420p"
    )
    writer.send(None)
    for group, num, x, y, step in rows:
        image = Image.new("RGB", (width, height))
        draw = ImageDraw.Draw(image)
        draw.text((2, 2), f"map {group}.{num} step {step}", fill=(232, 232, 232))
        top = 16 + y * SCALE
        draw.rectangle(
            (x * SCALE, top, x * SCALE + SCALE - 1, top + SCALE - 1), fill=(88, 200, 104)
        )
        writer.send(np.asarray(image).tobytes())
    writer.close()


def main(root: Path = RUNS, out: Path = DASHBOARD) -> None:
    con = connect(root)
    [(runs,)] = con.sql("SELECT count(DISTINCT run_id) FROM runs").fetchall()
    if not runs:
        raise SystemExit(
            f"No runs under {root}. Write a fake one with: "
            "uv run python -m algoarcade.logs.fake --run-id fake0 --seed 0"
        )
    commit = subprocess.run(
        ["git", "describe", "--always", "--dirty"], capture_output=True, text=True, check=True
    ).stdout.strip()
    # Quoted: an all-digit or "1e5"-shaped commit would otherwise read as a number.
    (out / "_variables.yml").write_text(
        f'commit: "{commit}"\ndate: "{date.today().isoformat()}"\nruns: {runs}\n'
    )
    (out / "plotly.min.js").write_text(get_plotlyjs(), encoding="utf-8")
    if (root.parent / "videos").is_dir():
        shutil.copytree(root.parent / "videos", out / "videos", dirs_exist_ok=True)
    walk_clip(con, out / "videos" / "walk.mp4")


if __name__ == "__main__":
    main()
