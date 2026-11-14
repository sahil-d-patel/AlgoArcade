"""FireRed map grids built offline from pret/pokefirered. Contract: docs/harness-spec.md, FireRed map data.

Fetch the files first with: uv run python scripts/fetch_maps.py
"""

import json
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from algoarcade.games.firered.addresses import LAYOUT_COMMIT

MAP_DATA = Path(__file__).resolve().parents[4] / "data" / "pokefirered"
MARKER = "maps.commit"
PALLET_TOWN = (3, 0)
# include/global.fieldmap.h at LAYOUT_COMMIT: bits 10 and 11 of a map.bin cell are collision, and
# MapGridGetCollisionAt treats MAPGRID_UNDEFINED as blocked.
COLLISION_MASK = 0x0C00
METATILE_ID_MASK = 0x03FF  # MAPGRID_METATILE_ID_MASK
MAPGRID_UNDEFINED = 0x03FF
# GetAttributeByMetatileIdAndMapLayout in src/fieldmap.c: ids below NUM_METATILES_IN_PRIMARY (640)
# index the primary tileset's attributes, the rest the secondary's. Behaviour is bits 0 to 8.
NUM_METATILES_IN_PRIMARY = 640
BEHAVIOR_MASK = 0x01FF
# MB_JUMP_EAST to MB_JUMP_SOUTH in include/constants/metatile_behaviors.h.
LEDGES = (0x38, 0x39, 0x3A, 0x3B)
# sBehaviorSurfable in src/metatile_behavior.c.
WATER = (0x10, 0x11, 0x12, 0x13, 0x15, 0x1A, 0x1B, 0x50, 0x51, 0x52, 0x53)


@dataclass(frozen=True)
class Connection:
    direction: str  # up, down, left or right, as in map.json
    offset: int
    map_id: tuple[int, int]


@dataclass(frozen=True, eq=False)
class MapGrid:
    name: str
    # map.bin cells and metatile behaviours, indexed [y, x] in the frame of the map's events.
    blocks: NDArray[np.uint16]
    behavior: NDArray[np.uint16]
    connections: tuple[Connection, ...]

    @property
    def width(self) -> int:
        return self.blocks.shape[1]

    @property
    def height(self) -> int:
        return self.blocks.shape[0]

    @property
    def blocked(self) -> NDArray[np.bool_]:
        return (self.blocks & COLLISION_MASK != 0) | (self.blocks == MAPGRID_UNDEFINED)

    @property
    def ledge(self) -> NDArray[np.bool_]:
        return np.isin(self.behavior, LEDGES)

    @property
    def water(self) -> NDArray[np.bool_]:
        return np.isin(self.behavior, WATER)

    @property
    def walkable(self) -> NDArray[np.bool_]:
        # Water has collision 0; the player needs Surf. Ledges already carry collision 1.
        return ~self.blocked & ~self.water


def load_maps(root: Path = MAP_DATA) -> dict[tuple[int, int], MapGrid]:
    """Every map in map_groups.json, keyed by (map_group, map_num)."""
    marker = root / MARKER
    if not marker.is_file() or marker.read_text() != LAYOUT_COMMIT:
        raise FileNotFoundError(
            f"No map data from {LAYOUT_COMMIT} in {root}: run uv run python scripts/fetch_maps.py"
        )
    groups = json.loads((root / "data/maps/map_groups.json").read_text())
    layouts = {
        layout["id"]: layout
        for layout in json.loads((root / "data/layouts/layouts.json").read_text())["layouts"]
        if layout
    }
    headers = (root / "src/data/tilesets/headers.h").read_text()
    metatiles = (root / "src/data/tilesets/metatiles.h").read_text()
    attribute_files = dict(re.findall(r'u32 (\w+)\[\] = INCBIN_U32\("([^"]+)"\)', metatiles))
    tilesets = {
        name: root / attribute_files[attributes]
        for name, attributes in re.findall(
            r"struct Tileset (\w+) =\s*\{[^}]*\.metatileAttributes = (\w+),", headers
        )
    }
    headers_by_id = {}
    for group, group_name in enumerate(groups["group_order"]):
        for num, folder in enumerate(groups[group_name]):
            header = json.loads((root / "data/maps" / folder / "map.json").read_text())
            headers_by_id[(group, num)] = header
    ids = {header["id"]: map_id for map_id, header in headers_by_id.items()}
    maps = {}
    for map_id, header in headers_by_id.items():
        layout = layouts[header["layout"]]
        blocks = np.fromfile(root / layout["blockdata_filepath"], dtype="<u2")
        blocks = blocks.reshape(layout["height"], layout["width"])
        primary = np.fromfile(tilesets[layout["primary_tileset"]], dtype="<u4")
        if primary.size != NUM_METATILES_IN_PRIMARY:
            raise ValueError(f"{layout['primary_tileset']} has {primary.size} metatiles, not 640")
        secondary = np.fromfile(tilesets[layout["secondary_tileset"]], dtype="<u4")
        attributes = np.concatenate([primary, secondary])
        behavior = (attributes[blocks & METATILE_ID_MASK] & BEHAVIOR_MASK).astype(np.uint16)
        connections = tuple(
            Connection(connection["direction"], connection["offset"], ids[connection["map"]])
            for connection in header["connections"] or ()
        )
        maps[map_id] = MapGrid(header["id"], blocks, behavior, connections)
    return maps


def place(maps: dict[tuple[int, int], MapGrid]) -> dict[tuple[int, int], tuple[int, int]]:
    """Global offset of each map connected to Pallet Town: global (x, y) = local (x, y) + offset.

    A breadth-first walk over connections from Pallet Town at (0, 0). When two paths disagree, the
    first placement stands. Maps the walk does not reach, such as gyms and Viridian Forest, are not
    in the result.
    """
    offsets = {PALLET_TOWN: (0, 0)}
    queue = deque([PALLET_TOWN])
    while queue:
        map_id = queue.popleft()
        x, y = offsets[map_id]
        here = maps[map_id]
        for connection in here.connections:
            if connection.map_id in offsets:
                continue
            there = maps[connection.map_id]
            # FillNorthConnection and the others in src/fieldmap.c: the offset runs along the edge.
            offsets[connection.map_id] = {
                "up": (x + connection.offset, y - there.height),
                "down": (x + connection.offset, y + here.height),
                "left": (x - there.width, y + connection.offset),
                "right": (x + here.width, y + connection.offset),
            }[connection.direction]
            queue.append(connection.map_id)
    return offsets
