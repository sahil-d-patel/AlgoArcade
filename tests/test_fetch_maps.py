import importlib.util
import json
from pathlib import Path

import pytest

from algoarcade.games.firered.addresses import LAYOUT_COMMIT

SCRIPT = Path(__file__).parents[1] / "scripts" / "fetch_maps.py"
spec = importlib.util.spec_from_file_location("fetch_maps", SCRIPT)
assert spec is not None
assert spec.loader is not None
fetch_maps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch_maps)

# A one-map decomp: the index files point at one map.json, one map.bin and one attributes file.
FILES = {
    "data/maps/map_groups.json": json.dumps({"group_order": ["gMapGroup_A"], "gMapGroup_A": ["A"]}),
    "data/layouts/layouts.json": json.dumps(
        {"layouts": [{}, {"id": "LAYOUT_A", "blockdata_filepath": "data/layouts/A/map.bin"}]}
    ),
    "src/data/tilesets/headers.h": "",
    "src/data/tilesets/metatiles.h": (
        'const u16 gMetatiles_A[] = INCBIN_U16("data/tilesets/primary/a/metatiles.bin");\n'
        'const u32 gMetatileAttributes_A[] = INCBIN_U32("data/tilesets/primary/a/metatile_attributes.bin");\n'
    ),
    "data/maps/A/map.json": "{}",
    "data/layouts/A/map.bin": "\0\0",
    "data/tilesets/primary/a/metatile_attributes.bin": "\0\0\0\0",
}


def test_commit_is_the_layout_commit() -> None:
    assert fetch_maps.COMMIT == LAYOUT_COMMIT


def test_fetches_every_file_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    downloads = []

    def fake_download(path: str) -> bytes:
        downloads.append(path)
        return FILES[path].encode()

    monkeypatch.setattr(fetch_maps, "download", fake_download)
    monkeypatch.setattr(fetch_maps, "DESTINATION", tmp_path)
    fetch_maps.main()
    assert sorted(downloads) == sorted(FILES)
    assert all((tmp_path / path).is_file() for path in FILES)
    fetch_maps.main()
    assert len(downloads) == len(FILES)
