"""Schematic terrain from ENV-10 grids; no game artwork is used."""

from pathlib import Path

import numpy as np
import tyro
from matplotlib.axes import Axes
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
# Maps place() does not reach, each drawn in its own panel. Panel i is the i-th entry, from 1.
INSETS = {
    (1, 0): "Viridian Forest",
    (6, 2): "Pewter Gym",
    (4, 0): "Player's house 1F",
    (4, 1): "Player's house 2F",
    (4, 3): "Oak's lab",
    (5, 3): "Viridian Mart",
    (15, 3): "Forest north gate",
}
Offsets = dict[tuple[int, int], tuple[int, int]]


def terrain(grid: MapGrid) -> NDArray[np.uint8]:
    cells = np.zeros(grid.blocks.shape, dtype=np.uint8)
    cells[np.isin(grid.behavior, GRASS)] = 2
    cells[grid.blocked] = 1
    cells[grid.water] = 3
    cells[grid.ledge] = 4
    return cells


def locate(
    offsets: Offsets, map_group: int, map_num: int, x: int, y: int
) -> tuple[int, tuple[int, int]] | None:
    """The panel that draws a tile (0 the mainland, i the i-th inset) and the tile's (x, y) in
    that panel, one unit per tile. None when no panel draws the map."""
    key = (map_group, map_num)
    if key in offsets:
        dx, dy = offsets[key]
        return 0, (x + dx, y + dy)
    if key in INSETS:
        return list(INSETS).index(key) + 1, (x, y)
    return None


def origins(
    maps: dict[tuple[int, int], MapGrid], offsets: Offsets
) -> dict[tuple[int, int], tuple[int, tuple[int, int]]]:
    """The panel and top-left corner of every map a panel draws."""
    return {key: spot for key in maps if (spot := locate(offsets, *key, 0, 0))}


def mainland(maps: dict[tuple[int, int], MapGrid], offsets: Offsets) -> tuple[int, int, int, int]:
    """Left, top, right and bottom of the placed maps."""
    return (
        min(x for x, _ in offsets.values()),
        min(y for _, y in offsets.values()),
        max(x + maps[key].width for key, (x, _) in offsets.items()),
        max(y + maps[key].height for key, (_, y) in offsets.items()),
    )


def draw(
    maps: dict[tuple[int, int], MapGrid], offsets: Offsets, footer: str
) -> tuple[Figure, list[Axes]]:
    """The terrain figure and its panels, indexed as locate() numbers them."""
    cmap = ListedColormap(COLORS)
    figure = Figure(figsize=(16, 11), facecolor="#faf9f5")
    FigureCanvasAgg(figure)
    layout = figure.add_gridspec(4, 3, width_ratios=(4, 1, 1), hspace=0.35)
    panels = [figure.add_subplot(layout[:, 0])] + [
        figure.add_subplot(layout[i // 2, 1 + i % 2]) for i in range(len(INSETS))
    ]
    for key, (panel, (x, y)) in origins(maps, offsets).items():
        grid = maps[key]
        panels[panel].imshow(
            terrain(grid),
            cmap=cmap,
            vmin=0,
            vmax=4,
            interpolation="nearest",
            extent=(x, x + grid.width, y + grid.height, y),
        )
    left, top, right, bottom = mainland(maps, offsets)
    panels[0].set_xlim(left - 8, right + 8)
    panels[0].set_ylim(bottom + 8, top - 8)
    panels[0].set_title(f"Kanto mainland · {len(offsets)} connected maps", pad=14)
    for map_id, name in (((3, 0), "Pallet"), ((3, 1), "Viridian"), ((3, 2), "Pewter")):
        x, y = offsets[map_id]
        panels[0].annotate(
            name,
            (x + maps[map_id].width / 2, y + maps[map_id].height / 2),
            xytext=(-42, 0),
            textcoords="offset points",
            ha="right",
            fontsize=9,
            arrowprops={"arrowstyle": "-", "color": "#333333"},
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none"},
        )
    for panel, name in zip(panels[1:], INSETS.values(), strict=True):
        panel.set_title(name, fontsize=9)
    for panel in panels:
        panel.set_axis_off()
    figure.suptitle("Where the agent can explore", fontsize=22, y=0.97)
    figure.legend(
        handles=[
            Patch(facecolor=color, label=label) for color, label in zip(COLORS, LABELS, strict=True)
        ],
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.04),
    )
    figure.text(0.5, 0.025, footer, ha="center", color="#555555")
    return figure, panels


def main(out: Path) -> None:
    maps = load_maps()
    offsets = place(maps)
    figure, _ = draw(maps, offsets, "Terrain schematic · other separate areas omitted")
    figure.savefig(out, dpi=160, facecolor=figure.get_facecolor())
    print(f"{len(offsets)} placed maps; {len(INSETS)} inset panels; saved {out}")  # noqa: T201


if __name__ == "__main__":
    tyro.cli(main)
