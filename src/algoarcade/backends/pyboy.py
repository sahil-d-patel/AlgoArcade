"""Game Boy on PyBoy 2.8. Runs only the Super Mario Land dump pinned in algoarcade.rom."""

import io
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pyboy import PyBoy

from algoarcade.backends.base import Button
from algoarcade.rom import UnknownRomError, identify

ADDRESS_SPACE = 0x10000
# The Game Boy has no R or L, so those buttons are dropped.
BUTTON_NAMES = {
    Button.A: "a",
    Button.B: "b",
    Button.SELECT: "select",
    Button.START: "start",
    Button.RIGHT: "right",
    Button.LEFT: "left",
    Button.UP: "up",
    Button.DOWN: "down",
}


class PyBoyBackend:
    def __init__(self, rom: Path, state: bytes | None = None) -> None:
        game = identify(rom).game
        if game != "super_mario_land":
            raise UnknownRomError(f"{rom}: PyBoyBackend runs Super Mario Land, not {game}")
        self._pyboy = PyBoy(str(rom), window="null", sound_emulated=False)
        if state is not None:
            self.load_state(state)

    def step(self, frames: int = 1) -> None:
        self._pyboy.tick(frames)

    def press(self, buttons: Iterable[Button], hold: int = 8, release: int = 16) -> None:
        names = [BUTTON_NAMES[button] for button in buttons if button in BUTTON_NAMES]
        for name in names:
            self._pyboy.button_press(name)
        self._pyboy.tick(hold)
        for name in names:
            self._pyboy.button_release(name)
        self._pyboy.tick(release)

    def screen(self) -> NDArray[np.uint8]:
        return self._pyboy.screen.ndarray[:, :, :3].copy()

    def read(self, address: int, size: int) -> bytes:
        if address < 0 or size < 0 or address + size > ADDRESS_SPACE:
            raise ValueError(
                f"read of {size} bytes at {address:#06x} is outside the Game Boy address space"
            )
        return bytes(self._pyboy.memory[address : address + size])

    def save_state(self) -> bytes:
        buffer = io.BytesIO()
        self._pyboy.save_state(buffer)
        return buffer.getvalue()

    def load_state(self, state: bytes) -> None:
        self._pyboy.load_state(io.BytesIO(state))

    def close(self) -> None:
        # save=False: PyBoy would otherwise write a .ram file next to the ROM in data/roms/.
        self._pyboy.stop(save=False)
