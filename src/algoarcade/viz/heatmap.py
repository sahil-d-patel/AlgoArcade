"""Exact visit counts on the Kanto schematic for one completed run."""

from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tyro
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm

from algoarcade.games.firered.maps import load_maps, place
from algoarcade.logs.schema import validate
from algoarcade.viz.kanto_map import draw


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


def main(run: Path, out: Path) -> None:
    tables = {}
    for name in ("visits", "runs"):
        files = sorted((run / name).glob("*.parquet"))
        if not files:
            raise ValueError(f"No closed Parquet files in {run / name}")
        tables[name] = pa.concat_tables([pq.ParquetFile(path).read() for path in files])
    visits = counts(tables["visits"], tables["runs"])
    maps = load_maps()
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
