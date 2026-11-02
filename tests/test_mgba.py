import hashlib
from pathlib import Path

import numpy as np
import pytest

from algoarcade.backends.base import Button
from algoarcade.backends.mgba import EWRAM_BASE, EWRAM_SIZE, IWRAM_BASE, IWRAM_SIZE, MgbaBackend
from algoarcade.rom import UnknownRomError

ROM = Path(__file__).resolve().parents[1] / "data" / "roms" / "firered.gba"


@pytest.fixture
def mgba_backend() -> MgbaBackend:
    if not ROM.is_file():
        pytest.skip("Place your FireRed ROM in data/roms/firered.gba")
    return MgbaBackend(ROM)


def test_unknown_rom_is_rejected_before_importing_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected_import(name: str) -> None:
        pytest.fail("The ROM guard must run before importing mGBA")

    monkeypatch.setattr("algoarcade.backends.mgba.import_module", unexpected_import)
    path = tmp_path / "unknown.gba"
    path.write_bytes(b"not a supported ROM")
    with pytest.raises(UnknownRomError):
        MgbaBackend(path)


@pytest.mark.rom
def test_ram_views_survive_restore_and_1000_frame_replay(mgba_backend: MgbaBackend) -> None:
    backend = mgba_backend
    try:
        backend.step(60)
        state = backend.save_state()
        hashes = []
        for _ in range(2):
            backend.load_state(state)
            trace = []
            for index in range(100):
                backend.press([Button.A if index % 2 else Button.START], hold=2, release=8)
                ram = backend.read(EWRAM_BASE, EWRAM_SIZE) + backend.read(IWRAM_BASE, IWRAM_SIZE)
                trace.append(hashlib.sha256(ram).hexdigest())
            hashes.append(trace)
        assert hashes[0] == hashes[1]
        assert len(set(hashes[0])) > 1
        print(f"1000-frame EWRAM+IWRAM SHA-256: {hashes[0][-1]}")  # noqa: T201
    finally:
        backend.close()


@pytest.mark.rom
def test_input_release_memory_boundaries_and_bad_states(mgba_backend: MgbaBackend) -> None:
    backend = mgba_backend
    try:
        backend.step(30)
        state = backend.save_state()
        backend.press([Button.A, Button.START], hold=2, release=0)
        backend.step(8)
        expected = backend.read(EWRAM_BASE, EWRAM_SIZE) + backend.read(IWRAM_BASE, IWRAM_SIZE)
        backend.load_state(state)
        backend.press([Button.A, Button.START], hold=2, release=8)
        assert (
            backend.read(EWRAM_BASE, EWRAM_SIZE) + backend.read(IWRAM_BASE, IWRAM_SIZE) == expected
        )
        for base, size in [(EWRAM_BASE, EWRAM_SIZE), (IWRAM_BASE, IWRAM_SIZE)]:
            assert len(backend.read(base + size - 1, 1)) == 1
            for address, length in [(base - 1, 1), (base + size, 1), (base, -1), (base, size + 1)]:
                with pytest.raises(ValueError, match="outside RAM"):
                    backend.read(address, length)
        with pytest.raises(ValueError, match="rejected"):
            backend.load_state(b"invalid")
        with pytest.raises(ValueError, match="rejected"):
            backend.load_state(bytes(len(state)))
        for call in [lambda: backend.step(-1), lambda: backend.press([], hold=-1)]:
            with pytest.raises(ValueError, match="non-negative"):
                call()
    finally:
        backend.close()


@pytest.mark.rom
def test_rendering_can_be_disabled() -> None:
    if not ROM.is_file():
        pytest.skip("Place your FireRed ROM in data/roms/firered.gba")
    backend = MgbaBackend(ROM, render=False)
    try:
        backend.step(1000)
        state = backend.save_state()
        ram = backend.read(EWRAM_BASE, EWRAM_SIZE) + backend.read(IWRAM_BASE, IWRAM_SIZE)
        backend.step(1)
        backend.load_state(state)
        assert backend.read(EWRAM_BASE, EWRAM_SIZE) + backend.read(IWRAM_BASE, IWRAM_SIZE) == ram
        with pytest.raises(RuntimeError, match="Rendering is disabled"):
            backend.screen()
    finally:
        backend.close()
        backend.close()


@pytest.mark.rom
def test_restore_includes_screen_and_flash_without_advancing(mgba_backend: MgbaBackend) -> None:
    backend = mgba_backend
    try:
        backend.step(100)
        state = backend.save_state()
        screen = backend.screen()
        ram = backend.read(EWRAM_BASE, EWRAM_SIZE) + backend.read(IWRAM_BASE, IWRAM_SIZE)
        core = backend._core
        frame = core.frame_counter
        flash = core._native.memory.savedata.data
        original = flash[0]
        backend.step(300)
        assert not np.array_equal(backend.screen(), screen)
        flash[0] ^= 0xFF
        backend.load_state(state)
        assert core.frame_counter == frame
        np.testing.assert_array_equal(backend.screen(), screen)
        assert core._native.memory.savedata.data[0] == original
        assert backend.read(EWRAM_BASE, EWRAM_SIZE) + backend.read(IWRAM_BASE, IWRAM_SIZE) == ram
    finally:
        backend.close()


@pytest.mark.rom
def test_loading_headless_state_never_returns_stale_pixels(mgba_backend: MgbaBackend) -> None:
    if not ROM.is_file():
        pytest.skip("Place your FireRed ROM in data/roms/firered.gba")
    headless = MgbaBackend(ROM, render=False)
    try:
        headless.step(100)
        state = headless.save_state()
        mgba_backend.step(400)
        mgba_backend.load_state(state)
        with pytest.raises(RuntimeError, match="no screenshot"):
            mgba_backend.screen()
        mgba_backend.step()
        assert mgba_backend.screen().shape == (160, 240, 3)
    finally:
        headless.close()
        mgba_backend.close()
