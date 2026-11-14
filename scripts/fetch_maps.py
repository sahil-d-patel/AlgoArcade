"""Download the pokefirered map data that algoarcade.games.firered.maps reads.

Run with: uv run python scripts/fetch_maps.py
Files land in data/pokefirered/ under their path in the decomp. Contract: DATA.md, source register.
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

COMMIT = "e93eb12a336fb97df420b0961753d8a3713c9fcc"
DESTINATION = Path(__file__).resolve().parents[1] / "data" / "pokefirered"
# headers.h and metatiles.h are the only link from a layout's tileset name to its attributes file.
INDEX = (
    "data/maps/map_groups.json",
    "data/layouts/layouts.json",
    "src/data/tilesets/headers.h",
    "src/data/tilesets/metatiles.h",
)
# Written last, so an interrupted fetch runs again in full.
MARKER = "maps.commit"


def download(path: str) -> bytes:
    url = f"https://raw.githubusercontent.com/pret/pokefirered/{COMMIT}/{path}"
    with urlopen(url, timeout=30) as response:
        return response.read()


def fetch(paths: list[str]) -> None:
    with ThreadPoolExecutor(max_workers=16) as pool:
        for path, content in zip(paths, pool.map(download, paths), strict=True):
            target = DESTINATION / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)


def main() -> None:
    marker = DESTINATION / MARKER
    if marker.is_file() and marker.read_text() == COMMIT:
        print(f"Map data from {COMMIT} is already in {DESTINATION}")
        return
    fetch(list(INDEX))
    groups = json.loads((DESTINATION / INDEX[0]).read_text())
    layouts = json.loads((DESTINATION / INDEX[1]).read_text())["layouts"]
    metatiles = (DESTINATION / INDEX[3]).read_text()
    files = [
        *(
            f"data/maps/{name}/map.json"
            for group in groups["group_order"]
            for name in groups[group]
        ),
        # layouts.json pads the table with empty entries where a layout id is unused.
        *(layout["blockdata_filepath"] for layout in layouts if layout),
        *re.findall(r'INCBIN_U32\("([^"]+/metatile_attributes\.bin)"\)', metatiles),
    ]
    fetch(files)
    marker.write_text(COMMIT)
    print(f"Saved {len(INDEX) + len(files)} files from {COMMIT} in {DESTINATION}")


if __name__ == "__main__":
    main()
