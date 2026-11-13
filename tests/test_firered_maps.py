import numpy as np
import pytest

from algoarcade.games.firered.maps import MAP_DATA, MARKER, Connection, MapGrid, load_maps, place

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
    # reaches the city side first, so Route 6's edge is the one left out.
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
