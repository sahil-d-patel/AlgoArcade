import hashlib
import struct
from pathlib import Path

import numpy as np
import pytest

from algoarcade.backends.mgba import MgbaBackend
from algoarcade.games.firered.addresses import ADDRESSES
from algoarcade.games.firered.manifest import MANIFEST, ROM_SHA1, StartState
from algoarcade.games.firered.maps import (
    COLLISION_MASK,
    MAP_DATA,
    MAPGRID_UNDEFINED,
    MARKER,
    Connection,
    MapGrid,
    load_maps,
    place,
)
from algoarcade.games.firered.reader import FireRedReader
from algoarcade.rom import identify

ROOT = Path(__file__).resolve().parents[1]

PALLET_TOWN = (3, 0)
ROUTE1 = (3, 19)
VIRIDIAN_CITY = (3, 1)
ROUTE2 = (3, 20)
PEWTER_CITY = (3, 2)
VIRIDIAN_FOREST = (1, 0)
PEWTER_CITY_GYM = (6, 2)


def grid(width: int, height: int, *connections: Connection) -> MapGrid:
    blocks = np.zeros((height, width), dtype=np.uint16)
    return MapGrid("MAP_TEST", blocks, blocks.copy(), connections)


def test_offset_shifts_the_neighbour_along_the_shared_edge() -> None:
    maps = {
        PALLET_TOWN: grid(
            10,
            8,
            Connection("up", 2, (0, 1)),
            Connection("down", -3, (0, 2)),
            Connection("left", 4, (0, 3)),
            Connection("right", -1, (0, 4)),
        ),
        (0, 1): grid(5, 6),
        (0, 2): grid(5, 6),
        (0, 3): grid(5, 6),
        (0, 4): grid(5, 6),
        (0, 5): grid(5, 6),
    }
    assert place(maps) == {
        PALLET_TOWN: (0, 0),
        (0, 1): (2, -6),
        (0, 2): (-3, 8),
        (0, 3): (-5, 4),
        (0, 4): (10, -1),
    }


def test_blocks_decode_collision_and_behaviour() -> None:
    # Metatile 1, collision 1, elevation 1; then 0x3FF; then open water; then an open ledge.
    blocks = np.array([[0x1401, 0x03FF, 0x1002, 0x0003]], dtype=np.uint16)
    behavior = np.array([[0x00, 0x00, 0x10, 0x3B]], dtype=np.uint16)
    tiles = MapGrid("MAP_TEST", blocks, behavior, ())
    assert tiles.blocked.tolist() == [[True, True, False, False]]
    assert tiles.water.tolist() == [[False, False, True, False]]
    assert tiles.ledge.tolist() == [[False, False, False, True]]
    assert tiles.walkable.tolist() == [[False, False, False, True]]


@pytest.fixture(scope="module")
def maps() -> dict[tuple[int, int], MapGrid]:
    if not (MAP_DATA / MARKER).is_file():
        pytest.skip("Run uv run python scripts/fetch_maps.py to fetch the map data")
    return load_maps()


@pytest.fixture(scope="module")
def offsets(maps: dict[tuple[int, int], MapGrid]) -> dict[tuple[int, int], tuple[int, int]]:
    return place(maps)


def test_origins_relative_to_pallet(offsets: dict[tuple[int, int], tuple[int, int]]) -> None:
    assert offsets[PALLET_TOWN] == (0, 0)
    assert offsets[ROUTE1] == (0, -40)
    assert offsets[VIRIDIAN_CITY] == (-12, -80)
    assert offsets[ROUTE2] == (0, -160)
    assert offsets[PEWTER_CITY] == (-12, -200)


def test_no_two_placed_maps_share_a_tile(
    maps: dict[tuple[int, int], MapGrid], offsets: dict[tuple[int, int], tuple[int, int]]
) -> None:
    owner: dict[tuple[int, int], tuple[int, int]] = {}
    for map_id, (dx, dy) in offsets.items():
        for y in range(maps[map_id].height):
            for x in range(maps[map_id].width):
                tile = (x + dx, y + dy)
                assert tile not in owner, f"{map_id} and {owner[tile]} share {tile}"
                owner[tile] = map_id


def test_both_sides_of_each_connection_agree(
    maps: dict[tuple[int, int], MapGrid], offsets: dict[tuple[int, int], tuple[int, int]]
) -> None:
    opposite = {"up": "down", "down": "up", "left": "right", "right": "left"}
    disagree = []
    for map_id in offsets:
        for connection in maps[map_id].connections:
            back = Connection(opposite[connection.direction], -connection.offset, map_id)
            if back not in maps[connection.map_id].connections:
                disagree.append((maps[map_id].name, maps[connection.map_id].name))
    # Route 6 has MAP_SAFFRON_CITY_CONNECTION at offset 0, and that map has Route 6 at 12. The walk
    # places Route 6 from the city side, which Vermilion City's edge to Route 6 agrees with.
    assert disagree == [
        ("MAP_SAFFRON_CITY_CONNECTION", "MAP_ROUTE6"),
        ("MAP_ROUTE6", "MAP_SAFFRON_CITY_CONNECTION"),
    ]


def test_warp_only_maps_are_not_placed(
    maps: dict[tuple[int, int], MapGrid], offsets: dict[tuple[int, int], tuple[int, int]]
) -> None:
    assert maps[VIRIDIAN_FOREST].name == "MAP_VIRIDIAN_FOREST"
    assert maps[PEWTER_CITY_GYM].name == "MAP_PEWTER_CITY_GYM"
    assert VIRIDIAN_FOREST not in offsets
    assert PEWTER_CITY_GYM not in offsets


def test_grid_sizes(maps: dict[tuple[int, int], MapGrid]) -> None:
    assert (maps[PALLET_TOWN].width, maps[PALLET_TOWN].height) == (24, 20)
    assert (maps[ROUTE1].width, maps[ROUTE1].height) == (24, 40)
    assert (maps[VIRIDIAN_FOREST].width, maps[VIRIDIAN_FOREST].height) == (54, 69)


def test_a_ledge_and_a_water_tile_found_in_the_game(maps: dict[tuple[int, int], MapGrid]) -> None:
    # On rev 1 from seg_04_route1, the player walked to Route 1 (2, 4) with the ledge in view
    # below, held down and came out at (2, 7); from (2, 6), holding up did not move him.
    route1 = maps[ROUTE1]
    assert route1.ledge[5, 2]
    assert route1.behavior[5, 2] == 0x3B  # MB_JUMP_SOUTH
    assert route1.walkable[4, 2]
    assert not route1.walkable[5, 2]
    # From seg_02_pallet, the player stood at Pallet Town (7, 16) above the pond and held down
    # without moving.
    pallet = maps[PALLET_TOWN]
    assert pallet.water[17, 7]
    assert pallet.walkable[16, 7]
    assert not pallet.walkable[17, 7]


@pytest.mark.rom
@pytest.mark.parametrize("entry", MANIFEST, ids=lambda entry: entry.name)
def test_offline_collision_matches_the_live_map_buffer(
    maps: dict[tuple[int, int], MapGrid], entry: StartState
) -> None:
    rom = ROOT / "data/roms/firered.gba"
    path = ROOT / "data/states" / f"{entry.name}.state"
    if not rom.is_file() or not path.is_file():
        pytest.skip(f"Needs the local FireRed ROM and {path.name}")
    if identify(rom).sha1 != ROM_SHA1:
        pytest.skip("The start states were captured on rev 1")
    pytest.importorskip(
        "mgba.core", reason="mGBA binding did not load; see docs/setup.md", exc_type=ImportError
    )
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == entry.sha256
    backend = MgbaBackend(rom, render=False)
    backend.load_state(data)
    state = FireRedReader(ROM_SHA1).read(backend)
    tiles = maps[(state.map_group, state.map_num)]
    # struct BackupMapLayout in include/global.fieldmap.h: Xsize, Ysize, then the map pointer.
    # InitMapLayoutData (src/fieldmap.c) copies the map in with a MAP_OFFSET (7) border.
    xsize, ysize, pointer = struct.unpack(
        "<iiI", backend.read(ADDRESSES[ROM_SHA1]["VMap"].address, 12)
    )
    assert pointer == ADDRESSES[ROM_SHA1]["gBackupMapData"].address
    assert (xsize, ysize) == (tiles.width + 15, tiles.height + 14)
    cells = np.frombuffer(backend.read(pointer, xsize * ysize * 2), dtype="<u2")
    live = cells.reshape(ysize, xsize)[7 : 7 + tiles.height, 7 : 7 + tiles.width]
    backend.close()
    live_blocked = (live & COLLISION_MASK != 0) | (live == MAPGRID_UNDEFINED)
    assert (live_blocked == tiles.blocked).mean() >= 0.99
