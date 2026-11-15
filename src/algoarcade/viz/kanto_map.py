"""Schematic terrain from ENV-10 grids; no game artwork is used."""

from pathlib import Path

import matplotlib
import numpy as np
import tyro
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from numpy.typing import NDArray

from algoarcade.games.firered.maps import MapGrid, load_maps, place

COLORS = ("#e7e2cf", "#535b66", "#53945c", "#63b8db", "#d8a34a")
LABELS = ("Walkable", "Wall / blocked", "Grass", "Water", "Ledge")
# include/constants/metatile_behaviors.h at LAYOUT_COMMIT e93eb12a:
# MB_TALL_GRASS and MB_CYCLING_ROAD_PULL_DOWN_GRASS.
GRASS = (0x02, 0xD1)


def terrain(grid: MapGrid) -> NDArray[np.uint8]:
    cells = np.zeros(grid.blocks.shape, dtype=np.uint8)
    cells[np.isin(grid.behavior, GRASS)] = 2
    cells[grid.blocked] = 1
    cells[grid.water] = 3
    cells[grid.ledge] = 4
    return cells


def main(out: Path) -> None:
    maps = load_maps()
    offsets = place(maps)
    cmap = ListedColormap(COLORS)
    with matplotlib.rc_context({"font.family": "DejaVu Sans", "font.size": 10}):
        figure = Figure(figsize=(14, 11), facecolor="#faf9f5")
        FigureCanvasAgg(figure)
        panels = figure.add_gridspec(2, 2, width_ratios=(3, 1), hspace=0.3)
        mainland = figure.add_subplot(panels[:, 0])
        for map_id, (x, y) in offsets.items():
            grid = maps[map_id]
            mainland.imshow(
                terrain(grid),
                cmap=cmap,
                vmin=0,
                vmax=4,
                interpolation="nearest",
                extent=(x, x + grid.width, y + grid.height, y),
            )
        left = min(x for x, _ in offsets.values())
        top = min(y for _, y in offsets.values())
        right = max(x + maps[key].width for key, (x, _) in offsets.items())
        bottom = max(y + maps[key].height for key, (_, y) in offsets.items())
        mainland.set_xlim(left - 8, right + 8)
        mainland.set_ylim(bottom + 8, top - 8)
        mainland.set_title(f"Kanto mainland · {len(offsets)} connected maps", pad=14)
        mainland.set_axis_off()
        for map_id, name in (((3, 0), "Pallet"), ((3, 1), "Viridian"), ((3, 2), "Pewter")):
            x, y = offsets[map_id]
            mainland.annotate(
                name,
                (x + maps[map_id].width / 2, y + maps[map_id].height / 2),
                xytext=(-42, 0),
                textcoords="offset points",
                ha="right",
                fontsize=9,
                arrowprops={"arrowstyle": "-", "color": "#333333"},
                bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none"},
            )
        for row, (map_id, name) in enumerate((((1, 0), "Viridian Forest"), ((6, 2), "Pewter Gym"))):
            axis = figure.add_subplot(panels[row, 1])
            axis.imshow(terrain(maps[map_id]), cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
            axis.set_title(f"{name}\nSeparate area · enlarged")
            axis.set_axis_off()
        figure.suptitle("Where the agent can explore", fontsize=22, y=0.97)
        figure.legend(
            handles=[
                Patch(facecolor=color, label=label)
                for color, label in zip(COLORS, LABELS, strict=True)
            ],
            loc="lower center",
            ncol=5,
            frameon=False,
            bbox_to_anchor=(0.5, 0.04),
        )
        figure.text(
            0.5,
            0.025,
            "Terrain schematic · visits are not shown yet · other disconnected areas omitted",
            ha="center",
            color="#555555",
        )
        figure.savefig(out, dpi=160, facecolor=figure.get_facecolor())
    print(f"{len(offsets)} placed maps; 2 separate panels; saved {out}")  # noqa: T201


if __name__ == "__main__":
    tyro.cli(main)
