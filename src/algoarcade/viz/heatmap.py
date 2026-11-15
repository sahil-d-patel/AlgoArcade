"""Exact visit counts on the Kanto schematic for one completed run."""

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
from PIL import Image
from plotly.subplots import make_subplots

from algoarcade.games.firered.maps import MapGrid, load_maps, place
from algoarcade.logs.schema import validate
from algoarcade.viz.kanto_map import COLORS, LABELS, draw, terrain

WINDOW_STEPS = 500_000


def counts(visits: pa.Table, runs: pa.Table) -> Counter[tuple[int, int, int, int]]:
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
    result: Counter[tuple[int, int, int, int]] = Counter()
    for row in visits.to_pylist():
        if row["run_id"] != end["run_id"]:
            raise ValueError("Visit rows belong to another run")
        if row["visits"] < 0:
            raise ValueError("Visit counts cannot be negative")
        result[(row["map_group"], row["map_num"], row["x"], row["y"])] += row["visits"]
    if result.total() != end["env_steps"]:
        raise ValueError(f"Visits {result.total()} != run steps {end['env_steps']}")
    return result


def windows(
    visits: pa.Table, runs: pa.Table
) -> list[tuple[int, int, Counter[tuple[int, int, int, int]]]]:
    total = counts(visits, runs).total()
    if total == 0:
        raise ValueError("A slider requires at least one visit")
    width = WINDOW_STEPS * ceil(ceil(total / WINDOW_STEPS) / 50)
    frames: dict[int, Counter[tuple[int, int, int, int]]] = {
        start: Counter() for start in range(0, total, width)
    }
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


def interactive(
    visits: pa.Table, runs: pa.Table, maps: dict[tuple[int, int], MapGrid]
) -> go.Figure:
    batches = windows(visits, runs)
    offsets = place(maps)
    panels = dict.fromkeys(offsets, (1, 1)) | {(1, 0): (1, 2), (6, 2): (2, 2)}
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"rowspan": 2}, {}], [None, {}]],
        column_widths=[0.72, 0.28],
        horizontal_spacing=0.05,
        subplot_titles=[f"Kanto · {len(offsets)} connected maps", "Viridian Forest", "Pewter Gym"],
    )
    palette = (np.array([to_rgb(color) for color in COLORS]) * 255).astype(np.uint8)
    for key, (row, col) in panels.items():
        grid = maps[key]
        x, y = offsets.get(key, (0, 0))
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
            row=row,
            col=col,
        )
    shown = sorted({key[:2] for _, _, batch in batches for key in batch} & panels.keys())
    for index, key in enumerate(shown):
        grid = maps[key]
        x, y = offsets.get(key, (0, 0))
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
            row=panels[key][0],
            col=panels[key][1],
        )
    synthetic = runs["env_id"][0].as_py() == "fake"
    heading = "Synthetic exploration · demo only" if synthetic else "Agent exploration"
    run_id = escape(runs["run_id"][0].as_py())
    peak = max(2, max(value for _, _, batch in batches for value in batch.values()))
    frames = []
    for index, (start, end, batch) in enumerate(batches):
        grids = {key: np.zeros(maps[key].blocks.shape, dtype=np.int64) for key in shown}
        omitted = 0
        for (group, num, x, y), value in batch.items():
            key = (group, num)
            if key not in maps or not (0 <= x < maps[key].width and 0 <= y < maps[key].height):
                raise ValueError(f"Visit outside map bounds: {(group, num, x, y)}")
            if key in grids:
                grids[key][y, x] += value
            else:
                omitted += value
        data = [
            go.Heatmap(
                z=np.where(grid > 0, np.log10(np.maximum(grid, 1)), np.nan),
                customdata=grid,
            )
            for grid in grids.values()
        ]
        title = (
            f"{heading}<br><sup>{run_id} · steps {start + 1:,}-{end:,} · "
            f"{batch.total():,} visits · {omitted:,} outside shown panels</sup>"
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
    bounds = (
        min(x for x, _ in offsets.values()),
        min(y for _, y in offsets.values()),
        max(x + maps[key].width for key, (x, _) in offsets.items()),
        max(y + maps[key].height for key, (_, y) in offsets.items()),
    )
    for row, col, key, anchor in ((1, 1, None, "x"), (1, 2, (1, 0), "x2"), (2, 2, (6, 2), "x3")):
        left, top, right, bottom = (
            bounds if key is None else (0, 0, maps[key].width, maps[key].height)
        )
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
    tables = {}
    for name in ("visits", "runs"):
        files = sorted((run / name).glob("*.parquet"))
        if not files:
            raise ValueError(f"No closed Parquet files in {run / name}")
        tables[name] = pa.concat_tables([pq.ParquetFile(path).read() for path in files])
    visits = counts(tables["visits"], tables["runs"])
    maps = load_maps()
    if out.suffix.lower() == ".html":
        plot = interactive(tables["visits"], tables["runs"], maps)
        html = plot.to_html(include_plotlyjs=True, auto_play=False, config={"displaylogo": False})
        html = html.replace(
            "<head>", "<head><style>.imagelayer image {image-rendering: pixelated;}</style>"
        )
        out.write_text(html, encoding="utf-8")
        print(f"Saved interactive heatmap to {out} ({out.stat().st_size:,} bytes)")  # noqa: T201
        return
    grids = {}
    for (group, num, x, y), count in visits.items():
        key = (group, num)
        if key not in maps or not (0 <= x < maps[key].width and 0 <= y < maps[key].height):
            raise ValueError(f"Visit outside map bounds: {(group, num, x, y)}")
        if key not in grids:
            grids[key] = np.zeros(maps[key].blocks.shape, dtype=np.int64)
        grids[key][y, x] += count
    figure, axes = draw(maps)
    offsets = place(maps)
    norm = LogNorm(vmin=1, vmax=max(2, max(visits.values(), default=0)))
    mainland = axes[(3, 0)]
    xlim, ylim = mainland.get_xlim(), mainland.get_ylim()
    omitted = 0
    for key, grid in grids.items():
        if key not in axes:
            omitted += int(grid.sum())
            continue
        base = axes[key].images[0] if key not in offsets else None
        if base is not None:
            extent = base.get_extent()
        else:
            x, y = offsets[key]
            extent = (x, x + maps[key].width, y + maps[key].height, y)
        axes[key].imshow(
            np.ma.masked_equal(grid, 0),
            cmap="magma",
            norm=norm,
            interpolation="nearest",
            extent=extent,
        )
    mainland.set_xlim(xlim)
    mainland.set_ylim(ylim)
    synthetic = tables["runs"]["env_id"][0].as_py() == "fake"
    figure.suptitle(
        "Synthetic visits · demonstration only" if synthetic else "Agent visits · complete run",
        fontsize=22,
        y=0.97,
    )
    figure.texts[-1].set_text(
        f"{visits.total():,} visits verified against run steps · {omitted:,} visits in areas not shown"
    )
    colorbar = figure.add_axes((0.19, 0.095, 0.38, 0.014))
    figure.colorbar(
        ScalarMappable(norm=norm, cmap="magma"),
        cax=colorbar,
        orientation="horizontal",
    )
    colorbar.set_title("Visits per tile · logarithmic scale", fontsize=10, pad=6)
    figure.savefig(out, dpi=160, facecolor=figure.get_facecolor())
    print(  # noqa: T201
        f"{visits.total()} visits = run steps; {omitted} visits outside displayed panels; saved {out}"
    )


if __name__ == "__main__":
    tyro.cli(main)
