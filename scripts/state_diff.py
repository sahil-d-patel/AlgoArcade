"""Print the EWRAM bytes that differ between two save states.

uv run python scripts/state_diff.py A.state B.state
"""

import argparse
from pathlib import Path

from algoarcade.backends.base import GameBackend
from algoarcade.backends.mgba import EWRAM_BASE, EWRAM_SIZE, MgbaBackend

ROM = Path(__file__).resolve().parents[1] / "data" / "roms" / "firered.gba"


def compare_states(backend: GameBackend, state_a: bytes, state_b: bytes) -> None:
    """Print changed memory bytes and restore the backend's original state."""
    original = backend.save_state()
    try:
        backend.load_state(state_a)
        before = backend.read(EWRAM_BASE, EWRAM_SIZE)
        backend.load_state(state_b)
        after = backend.read(EWRAM_BASE, EWRAM_SIZE)
        for offset, (old, new) in enumerate(zip(before, after, strict=True)):
            if old != new:
                print(f"0x{EWRAM_BASE + offset:08x}: {old:02x} -> {new:02x}")
    finally:
        backend.load_state(original)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("state_a", type=Path)
    parser.add_argument("state_b", type=Path)
    args = parser.parse_args(argv)
    backend = MgbaBackend(ROM, render=False)
    try:
        compare_states(backend, args.state_a.read_bytes(), args.state_b.read_bytes())
    finally:
        backend.close()


if __name__ == "__main__":
    main()
