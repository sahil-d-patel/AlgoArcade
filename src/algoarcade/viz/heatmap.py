"""Exact visit counts on the Kanto schematic for one completed run."""

import os
from collections import Counter
from html import escape
from math import ceil
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import pyarrow as pa
import pyarrow.parquet as pq
import tyro
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, to_rgb
from numpy.typing import NDArray
from PIL import Image
from plotly.subplots import make_subplots

from algoarcade.games.firered.maps import MapGrid, load_maps, place
from algoarcade.logs.schema import validate
from algoarcade.viz.kanto_map import COLORS, INSETS, LABELS, draw, mainland, origins, terrain

WINDOW_STEPS = 500_000
# Anchored to the checkout like VIZ-01's site.RUNS: Quarto runs a page's cells from its folder.
RUNS = Path(os.environ.get("ALGOARCADE_RUNS", Path(__file__).resolve().parents[3] / "data/runs"))
PIXELATED = "<style>.imagelayer image {image-rendering: pixelated;}</style>"

Location = tuple[int, int, int, int]
MapId = tuple[int, int]


def load(run: Path) -> tuple[pa.Table, pa.Table]:
    """The visits and runs tables of one run directory, from its closed Parquet files."""
    tables = []
    for name in ("visits", "runs"):
        files = sorted((run / name).glob("*.parquet"))
        if not files:
            raise ValueError(f"No closed Parquet files in {run / name}")
        tables.append(pa.concat_tables([pq.ParquetFile(path).read() for path in files]))
    return tables[0], tables[1]


def counts(visits: pa.Table, runs: pa.Table) -> Counter[Location]:
    validate(visits, "visits")
    validate(runs, "runs")
    rows = runs.to_pylist()
    if len({row["run_id"] for row in rows}) != 1:
        raise ValueError("Select exactly one run")
    ends = [row for row in rows if row["status"] == "finished"]
    if len(ends) != 1:
        raise ValueError("A finished run is required to verify total visits")
    end = ends[0]
    if end["env_id"] not in ("firered", "fake"):
        raise ValueError("The Kanto heatmap requires FireRed or synthetic FireRed logs")
    result: Counter[Location] = Counter()
    for row in visits.to_pylist():
        if row["run_id"] != end["run_id"]:
            raise ValueError("Visit rows belong to another run")
        if row["visits"] < 0:
            raise ValueError("Visit counts cannot be negative")
        result[(row["map_group"], row["map_num"], row["x"], row["y"])] += row["visits"]
    if result.total() != end["env_steps"]:
        raise ValueError(f"Visits {result.total()} != run steps {end['env_steps']}")
    return result


def windows(visits: pa.Table, runs: pa.Table) -> list[tuple[int, int, Counter[Location]]]:
    total = counts(visits, runs).total()
    if total == 0:
        raise ValueError("A slider requires at least one visit")
    width = WINDOW_STEPS * ceil(ceil(total / WINDOW_STEPS) / 50)
    frames: dict[int, Counter[Location]] = {start: Counter() for start in range(0, total, width)}
    totals: Counter[int] = Counter()
    for row in visits.to_pylist():
        start = row["window_start_step"]
        if start < 0 or start >= total or start % WINDOW_STEPS:
            raise ValueError(f"Invalid checkpoint window: {start}")
        totals[start] += row["visits"]
        key = (row["map_group"], row["map_num"], row["x"], row["y"])
        frames[start // width * width][key] += row["visits"]
    for start in range(0, total, WINDOW_STEPS):
        if totals[start] != min(WINDOW_STEPS, total - start):
            raise ValueError(f"Visit total does not match checkpoint window {start}")
    return [(start, min(start + width, total), frame) for start, frame in frames.items()]


def tiles(
    batch: Counter[Location],
    maps: dict[MapId, MapGrid],
    drawn: dict[MapId, tuple[int, tuple[int, int]]],
) -> tuple[dict[MapId, NDArray[np.int64]], int, int]:
    """Visits per drawn map as [y, x] arrays, the visits on maps no panel draws, and the visits
    off their map's grid or on an unknown map. A real run can log such a tile; it is counted and
    reported, not drawn."""
    grids: dict[MapId, NDArray[np.int64]] = {}
    hidden = off = 0
    for (group, num, x, y), value in batch.items():
        grid = maps.get((group, num))
        if grid is None or not (0 <= x < grid.width and 0 <= y < grid.height):
            off += value
        elif (group, num) not in drawn:
            hidden += value
        else:
            cells = grids.setdefault((group, num), np.zeros(grid.blocks.shape, dtype=np.int64))
            cells[y, x] += value
    return grids, hidden, off


def skipped(hidden: int, off: int) -> str:
    return f"{hidden:,} in areas not shown" + (f" · {off:,} off the map grid" if off else "")


def interactive(visits: pa.Table, runs: pa.Table, maps: dict[MapId, MapGrid]) -> go.Figure:
    batches = windows(visits, runs)
    offsets = place(maps)
    drawn = origins(maps, offsets)
    rows = (len(INSETS) + 1) // 2
    slots = [{} if i < len(INSETS) else None for i in range(2 * rows)]
    fig = make_subplots(
        rows=rows,
        cols=3,
        specs=[
            [{"rowspan": rows} if r == 0 else None, *slots[2 * r : 2 * r + 2]] for r in range(rows)
        ],
        column_widths=[0.6, 0.2, 0.2],
        horizontal_spacing=0.03,
        vertical_spacing=0.06,
        subplot_titles=[f"Kanto · {len(offsets)} connected maps", *INSETS.values()],
    )
    # Panel i of locate() is subplot i + 1 in row-major order.
    cells = [(1, 1)] + [(i // 2 + 1, i % 2 + 2) for i in range(len(INSETS))]
    palette = (np.array([to_rgb(color) for color in COLORS]) * 255).astype(np.uint8)
    for key, (panel, (x, y)) in drawn.items():
        grid = maps[key]
        fig.add_layout_image(
            source=Image.fromarray(palette[terrain(grid)]),
            x=x,
            y=y,
            sizex=grid.width,
            sizey=grid.height,
            xanchor="left",
            yanchor="top",
            sizing="stretch",
            layer="below",
            row=cells[panel][0],
            col=cells[panel][1],
        )
    tallies = [tiles(batch, maps, drawn) for _, _, batch in batches]
    shown = sorted({key for grids, _, _ in tallies for key in grids})
    peak = max(2, max((int(g.max()) for grids, _, _ in tallies for g in grids.values()), default=0))
    for index, key in enumerate(shown):
        grid = maps[key]
        panel, (x, y) = drawn[key]
        fig.add_trace(
            go.Heatmap(
                name=str(index),
                x0=x + 0.5,
                y0=y + 0.5,
                dx=1,
                dy=1,
                z=np.full(grid.blocks.shape, np.nan),
                coloraxis="coloraxis",
                text=[
                    [
                        f"{grid.name.removeprefix('MAP_')} ({key[0]}, {key[1]}) · tile ({tx}, {ty})"
                        for tx in range(grid.width)
                    ]
                    for ty in range(grid.height)
                ],
                hovertemplate="%{text}<br>Visits in this window: %{customdata:,}<extra></extra>",
                hoverongaps=False,
                zsmooth=False,
            ),
            row=cells[panel][0],
            col=cells[panel][1],
        )
    synthetic = runs["env_id"][0].as_py() == "fake"
    heading = "Synthetic exploration · demo only" if synthetic else "Agent exploration"
    run_id = escape(runs["run_id"][0].as_py())
    frames = []
    for index, ((start, end, batch), (grids, hidden, off)) in enumerate(
        zip(batches, tallies, strict=True)
    ):
        values = [grids.get(key, np.zeros(maps[key].blocks.shape, dtype=np.int64)) for key in shown]
        data = [
            go.Heatmap(z=np.where(grid > 0, np.log10(np.maximum(grid, 1)), np.nan), customdata=grid)
            for grid in values
        ]
        title = (
            f"{heading}<br><sup>{run_id} · steps {start + 1:,}-{end:,} · "
            f"{batch.total():,} visits · {skipped(hidden, off)}</sup>"
        )
        frames.append(
            go.Frame(
                name=str(index),
                data=data,
                traces=list(range(len(shown))),
                layout={"title": {"text": title}},
            )
        )
    for index, first in enumerate(frames[0].data):
        fig.update_traces(z=first.z, customdata=first.customdata, selector={"name": str(index)})
    fig.frames = frames
    for panel, (row, col) in enumerate(cells):
        if panel == 0:
            left, top, right, bottom = mainland(maps, offsets)
        else:
            grid = maps[list(INSETS)[panel - 1]]
            left, top, right, bottom = 0, 0, grid.width, grid.height
        anchor = "x" if panel == 0 else f"x{panel + 1}"
        fig.update_xaxes(range=[left, right], visible=False, constrain="domain", row=row, col=col)
        fig.update_yaxes(
            range=[bottom, top], visible=False, scaleanchor=anchor, scaleratio=1, row=row, col=col
        )
    ticks = list(range(ceil(np.log10(peak))))
    fig.update_layout(
        title={"text": frames[0].layout.title.text, "x": 0.03},
        height=920,
        margin={"l": 25, "r": 80, "t": 110, "b": 180},
        paper_bgcolor="#faf9f5",
        plot_bgcolor="#faf9f5",
        font={"family": "Arial", "size": 13},
        coloraxis={
            "colorscale": "Magma",
            "cmin": 0,
            "cmax": np.log10(peak),
            "colorbar": {
                "title": "Visits<br>(log scale)",
                "tickvals": ticks,
                "ticktext": [f"{10**tick:,}" for tick in ticks],
                "len": 0.7,
            },
        },
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "Window "},
                "pad": {"t": 35},
                "steps": [
                    {
                        "label": str(index + 1),
                        "method": "animate",
                        "args": [
                            [str(index)],
                            {
                                "mode": "immediate",
                                "transition": {"duration": 0},
                                "frame": {"duration": 0, "redraw": True},
                            },
                        ],
                    }
                    for index in range(len(frames))
                ],
            }
        ],
    )
    legend = " · ".join(
        f'<span style="color:{color}">■</span> {name}'
        for color, name in zip(COLORS, LABELS, strict=True)
    )
    fig.add_annotation(
        text=legend
        + "<br>Each window is separate, not cumulative. Unvisited tiles retain terrain colors.",
        x=0.5,
        y=-0.24,
        xref="paper",
        yref="paper",
        showarrow=False,
    )
    return fig


def main(run: Path, out: Path) -> None:
    visits, runs = load(run)
    maps = load_maps()
    if out.suffix.lower() == ".html":
        html = interactive(visits, runs, maps).to_html(
            include_plotlyjs=True, auto_play=False, config={"displaylogo": False}
        )
        out.write_text(html.replace("<head>", "<head>" + PIXELATED), encoding="utf-8")
        print(f"Saved interactive heatmap to {out} ({out.stat().st_size:,} bytes)")  # noqa: T201
        return
    total = counts(visits, runs)
    offsets = place(maps)
    drawn = origins(maps, offsets)
    grids, hidden, off = tiles(total, maps, drawn)
    footer = f"{total.total():,} visits verified against run steps · {skipped(hidden, off)}"
    figure, panels = draw(maps, offsets, footer)
    norm = LogNorm(vmin=1, vmax=max(2, max((int(g.max()) for g in grids.values()), default=0)))
    xlim, ylim = panels[0].get_xlim(), panels[0].get_ylim()
    for key, grid in grids.items():
        panel, (x, y) = drawn[key]
        panels[panel].imshow(
            np.ma.masked_equal(grid, 0),
            cmap="magma",
            norm=norm,
            interpolation="nearest",
            extent=(x, x + grid.shape[1], y + grid.shape[0], y),
        )
    panels[0].set_xlim(xlim)
    panels[0].set_ylim(ylim)
    synthetic = runs["env_id"][0].as_py() == "fake"
    figure.suptitle(
        "Synthetic visits · demonstration only" if synthetic else "Agent visits · complete run",
        fontsize=22,
        y=0.97,
    )
    colorbar = figure.add_axes((0.19, 0.095, 0.38, 0.014))
    figure.colorbar(ScalarMappable(norm=norm, cmap="magma"), cax=colorbar, orientation="horizontal")
    colorbar.set_title("Visits per tile · logarithmic scale", fontsize=10, pad=6)
    figure.savefig(out, dpi=160, facecolor=figure.get_facecolor())
    print(f"{total.total()} visits = run steps; {skipped(hidden, off)}; saved {out}")  # noqa: T201


if __name__ == "__main__":
    tyro.cli(main)
