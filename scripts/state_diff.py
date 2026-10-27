"""Compare EWRAM through a supplied GameBackend; CLI wiring awaits ENV-04."""

from algoarcade.backends.base import GameBackend

# GBA hardware memory map: https://mgba-emu.github.io/gbatek/#gbamemorymap
EWRAM_BASE = 0x02000000
EWRAM_SIZE = 0x40000


def compare_states(backend: GameBackend, state_a: bytes, state_b: bytes) -> None:
    """Print changed memory bytes and restore the backend's original state."""
    original = backend.save_state()
    try:
        backend.load_state(state_a)
        before = backend.read(EWRAM_BASE, EWRAM_SIZE)
        backend.load_state(state_b)
        after = backend.read(EWRAM_BASE, EWRAM_SIZE)
        if len(before) != EWRAM_SIZE or len(after) != EWRAM_SIZE:
            raise ValueError("Backend must return the full EWRAM region")
        for offset, (old, new) in enumerate(zip(before, after, strict=True)):
            if old != new:
                print(f"0x{EWRAM_BASE + offset:08x}: {old:02x} -> {new:02x}")
    finally:
        backend.load_state(original)


if __name__ == "__main__":
    raise SystemExit(
        "Save-state file comparison awaits the ENV-04 mGBA backend. "
        "Use compare_states(backend, state_a, state_b) with an existing GameBackend."
    )
