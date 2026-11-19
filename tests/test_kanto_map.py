import json
from pathlib import Path

import numpy as np

from algoarcade.games.firered.addresses import LAYOUT_COMMIT
from algoarcade.games.firered.maps import MARKER, MapGrid, load_maps
from algoarcade.viz.kanto_map import terrain


def test_terrain_preserves_blockers_and_distinguishes_special_tiles() -> None:
    grid = MapGrid(
        "MAP_TEST",
        np.array([[0, 0x1401, 0, 0, 0x0403, 0x03FF, 0x0401, 0]], dtype=np.uint16),
        np.array([[0, 0, 2, 0x15, 0x3B, 0, 2, 0xD1]], dtype=np.uint16),
        (),
    )
    assert terrain(grid).tolist() == [[0, 1, 2, 3, 4, 1, 1, 2]]


def test_synthetic_map_bin_cell_0x1401(tmp_path: Path) -> None:
    files = {
        "data/maps/map_groups.json": json.dumps({"group_order": ["g"], "g": ["A"]}),
        "data/layouts/layouts.json": json.dumps(
            {
                "layouts": [
                    {
                        "id": "LAYOUT_A",
                        "width": 1,
                        "height": 1,
                        "blockdata_filepath": "data/layouts/A/map.bin",
                        "primary_tileset": "gTileset_P",
                        "secondary_tileset": "gTileset_S",
                    }
                ]
            }
        ),
        "src/data/tilesets/headers.h": (
            "struct Tileset gTileset_P =\n{\n    .metatileAttributes = gMetatileAttributes_P,\n};\n"
            "struct Tileset gTileset_S =\n{\n    .metatileAttributes = gMetatileAttributes_S,\n};\n"
        ),
        "src/data/tilesets/metatiles.h": (
            'const u32 gMetatileAttributes_P[] = INCBIN_U32("p.bin");\n'
            'const u32 gMetatileAttributes_S[] = INCBIN_U32("s.bin");\n'
        ),
        "data/maps/A/map.json": json.dumps(
            {"id": "MAP_A", "layout": "LAYOUT_A", "connections": None}
        ),
        MARKER: LAYOUT_COMMIT,
    }
    for path, text in files.items():
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text(text)
    (tmp_path / "data/layouts/A").mkdir(parents=True)
    (tmp_path / "data/layouts/A/map.bin").write_bytes(np.array([0x1401], "<u2").tobytes())
    primary = np.zeros(640, "<u4")
    primary[1] = 0x02  # MB_TALL_GRASS, only on metatile 1
    primary.tofile(tmp_path / "p.bin")
    np.zeros(1, "<u4").tofile(tmp_path / "s.bin")

    grid = load_maps(tmp_path)[(0, 0)]
    cell = int(grid.blocks[0, 0])
    assert grid.behavior[0, 0] == 0x02  # metatile 1: collision and elevation bits masked off
    assert (cell >> 10) & 3 == 1
    assert grid.blocked[0, 0]
    assert cell >> 12 == 1
    assert terrain(grid)[0, 0] == 1
