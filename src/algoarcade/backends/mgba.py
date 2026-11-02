from collections.abc import Iterable
from importlib import import_module
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from algoarcade.backends.base import Button
from algoarcade.rom import identify

EWRAM_BASE = 0x02000000
EWRAM_SIZE = 0x40000
IWRAM_BASE = 0x03000000
IWRAM_SIZE = 0x8000
# mGBA 0.10.5 include/mgba/core/serialize.h; these macros are absent from the binding.
SAVESTATE_SCREENSHOT = 1
SAVESTATE_SAVEDATA = 2
SAVESTATE_RTC = 8


class MgbaBackend:
    def __init__(self, rom_path: Path, *, render: bool = True) -> None:
        self.rom = identify(rom_path)
        # Import after identifying the ROM so the guard also works without the binding.
        core_module = import_module("mgba.core")
        binding = import_module("mgba._pylib")
        self._ffi, self._lib = binding.ffi, binding.lib
        self._vfile = import_module("mgba.vfs").VFile
        # Native debug output must not enter an MCP server's stdout.
        import_module("mgba.log").silence()
        self._core = core_module.load_path(str(rom_path))
        if self._core is None:
            raise ValueError(f"mGBA could not load {rom_path}")
        self._image = None
        self._pixels = None
        if render:
            self._image = import_module("mgba.image").Image(240, 160)
            self._core.set_video_buffer(self._image)
            self._pixels = np.frombuffer(
                self._ffi.buffer(self._image.buffer), dtype=np.uint8
            ).reshape(160, 240, 4)
        self._core.reset()
        self._screen_ready = True
        # libmgba-py 0.2.0-2 exposes neither RAM pointers nor state flags. These two lines are
        # the only reads of its private fields: the mCore handle and the GBA memory struct.
        self._handle = self._core._core
        memory = self._core._native.memory
        self._regions: tuple[tuple[int, memoryview], ...] = (
            (EWRAM_BASE, memoryview(self._ffi.buffer(memory.wram, EWRAM_SIZE))),
            (IWRAM_BASE, memoryview(self._ffi.buffer(memory.iwram, IWRAM_SIZE))),
        )

    def _advance(self, frames: int) -> None:
        for _ in range(frames):
            self._core.run_frame()
            self._screen_ready = True

    def step(self, frames: int = 1) -> None:
        if frames < 0:
            raise ValueError("frames must be non-negative")
        self._core.set_keys()
        self._advance(frames)

    def press(self, buttons: Iterable[Button], hold: int = 8, release: int = 16) -> None:
        if hold < 0 or release < 0:
            raise ValueError("hold and release must be non-negative")
        self._core.set_keys(*buttons)
        try:
            self._advance(hold)
        finally:
            self._core.set_keys()
        self.step(release)

    def screen(self) -> NDArray[np.uint8]:
        if self._pixels is None:
            raise RuntimeError("Rendering is disabled")
        if not self._screen_ready:
            raise RuntimeError("Loaded state has no screenshot; advance a frame before screen()")
        return self._pixels[:, :, :3].copy()

    def read(self, address: int, size: int) -> bytes:
        for base, view in self._regions:
            offset = address - base
            if 0 <= offset <= len(view) and 0 <= size <= len(view) - offset:
                return bytes(view[offset : offset + size])
        raise ValueError(f"read of {size} bytes at {address:#010x} is outside RAM")

    def save_state(self) -> bytes:
        flags = SAVESTATE_SAVEDATA | SAVESTATE_RTC
        if self._pixels is not None and self._screen_ready:
            flags |= SAVESTATE_SCREENSHOT
        file = self._vfile.fromEmpty()
        try:
            if not self._lib.mCoreSaveStateNamed(self._handle, file.handle, flags):
                raise RuntimeError("mGBA could not save state")
            file.seek(0, 0)
            return file.read_all()
        finally:
            file.close()

    def load_state(self, state: bytes) -> None:
        file = self._vfile.fromEmpty()
        try:
            file.write(state, len(state))
            file.seek(0, 0)
            flags = SAVESTATE_SCREENSHOT | SAVESTATE_SAVEDATA | SAVESTATE_RTC
            if not self._lib.mCoreLoadStateNamed(self._handle, file.handle, flags):
                raise ValueError("mGBA rejected the save state")
            self._screen_ready = state.startswith(b"\x89PNG\r\n\x1a\n")
        finally:
            file.close()

    def close(self) -> None:
        if self._core is not None:
            self._regions = ()
            self._ffi.release(self._handle)
            self._core = None
            self._pixels = None
            self._image = None
