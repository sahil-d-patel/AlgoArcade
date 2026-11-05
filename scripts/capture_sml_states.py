"""Regenerate the Super Mario Land start states in data/states/ from power-on.

    uv run python scripts/capture_sml_states.py             # writes data/states/level_1_1.state
    uv run python scripts/capture_sml_states.py --out DIR   # writes DIR instead

level_1_1 is the first frame of play after pressing START on the title screen.
"""

import argparse
import hashlib
from pathlib import Path

from algoarcade.backends.base import Button, read_u8
from algoarcade.backends.pyboy import PyBoyBackend
from algoarcade.games.super_mario_land.manifest import ROM_SHA1, STATES
from algoarcade.games.super_mario_land.reader import GAME_STATE, MENU, SuperMarioLandReader
from algoarcade.rom import identify

ROM = Path(__file__).resolve().parents[1] / "data" / "roms" / "super_mario_land.gb"
# The title screen accepts START by frame 120 after power-on.
TITLE_FRAMES = 120


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=STATES)
    out = parser.parse_args().out
    if identify(ROM).sha1 != ROM_SHA1:
        parser.error(f"{ROM} is not the dump the manifest was captured on")
    backend = PyBoyBackend(ROM, render=False)
    backend.step(TITLE_FRAMES)
    backend.press([Button.START], 8, 16)
    while read_u8(backend, GAME_STATE) or read_u8(backend, MENU):
        backend.step(1)
    state = SuperMarioLandReader().read(backend)
    data = backend.save_state()
    backend.close()
    out.mkdir(parents=True, exist_ok=True)
    (out / "level_1_1.state").write_bytes(data)
    print(f"level_1_1 {hashlib.sha256(data).hexdigest()} {state.location} lives {state.lives_left}")


if __name__ == "__main__":
    main()
