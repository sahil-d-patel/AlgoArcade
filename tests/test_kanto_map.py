import numpy as np

from algoarcade.games.firered.maps import MapGrid
from algoarcade.viz.kanto_map import terrain


def test_terrain_preserves_blockers_and_distinguishes_special_tiles() -> None:
    grid = MapGrid(
        "MAP_TEST",
        np.array([[0, 0x1401, 0, 0, 0x0403, 0x03FF, 0x0401, 0]], dtype=np.uint16),
        np.array([[0, 0, 2, 0x15, 0x3B, 0, 2, 0xD1]], dtype=np.uint16),
        (),
    )
    assert terrain(grid).tolist() == [[0, 1, 2, 3, 4, 1, 1, 2]]
    assert grid.blocks[0, 1] & 0x03FF == 1
    assert (grid.blocks[0, 1] >> 10) & 3 == 1
    assert grid.blocks[0, 1] >> 12 == 1
