"""A toy console that needs no ROM: two rooms, a door, walls and one chest.

Tests and CI run on it. All game state lives in a small RAM block, so readers, rewards and
the conformance suite exercise the same read path they use on a real emulator.
"""

import struct
from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

from algoarcade.backends.base import Button

RAM_BASE = 0x0200_0000
RAM_SIZE = 15
# Offsets from RAM_BASE. FRAME and RNG are little-endian u32, the rest are one byte.
PLAYER_X = 0x00
PLAYER_Y = 0x01
ROOM = 0x02
FLAGS = 0x03
FRAME = 0x04
RNG = 0x08
LEVEL_SUM = 0x0C
BADGES = 0x0D
MILESTONES = 0x0E

FLAG_CHEST_OPENED = 0x01
TILE_PIXELS = 8

# '#' wall, '.' floor, 'C' chest, 'D' door to the same row of the other room.
ROOMS = (
    (
        "##########",
        "#..C.....#",
        "#........#",
        "#........D",
        "#........#",
        "##########",
    ),
    (
        "##########",
        "#........#",
        "#........#",
        "D........#",
        "#........#",
        "##########",
    ),
)
WIDTH = len(ROOMS[0][0])

MOVES = {Button.RIGHT: (1, 0), Button.LEFT: (-1, 0), Button.UP: (0, -1), Button.DOWN: (0, 1)}
COLOURS = {"#": (40, 44, 60), ".": (208, 204, 180), "C": (224, 176, 32), "D": (120, 80, 40)}
OPEN_CHEST_COLOUR = (128, 104, 40)
PLAYER_COLOUR = (216, 40, 40)


class FakeBackend:
    def __init__(self) -> None:
        self._ram = bytearray(RAM_SIZE)
        self._ram[PLAYER_X] = 1
        self._ram[PLAYER_Y] = 1
        self._ram[LEVEL_SUM] = 5
        struct.pack_into("<I", self._ram, RNG, 0x2545F491)

    def step(self, frames: int = 1) -> None:
        frame, rng = struct.unpack_from("<II", self._ram, FRAME)
        for _ in range(frames):
            rng = (rng * 1664525 + 1013904223) & 0xFFFFFFFF
        struct.pack_into("<II", self._ram, FRAME, (frame + frames) & 0xFFFFFFFF, rng)

    def press(self, buttons: Iterable[Button], hold: int = 8, release: int = 16) -> None:
        ram = self._ram
        for button in buttons:
            if button in MOVES:
                self._move(*MOVES[button])
            elif button is Button.A and ROOMS[ram[ROOM]][ram[PLAYER_Y]][ram[PLAYER_X]] == "C":
                ram[FLAGS] |= FLAG_CHEST_OPENED
        self.step(hold + release)

    def _move(self, dx: int, dy: int) -> None:
        ram = self._ram
        x, y = ram[PLAYER_X] + dx, ram[PLAYER_Y] + dy
        tile = ROOMS[ram[ROOM]][y][x]
        if tile == "#":
            return
        if tile == "D":
            ram[ROOM] ^= 1
            x = 1 if x else WIDTH - 2
        ram[PLAYER_X], ram[PLAYER_Y] = x, y

    def screen(self) -> NDArray[np.uint8]:
        ram = self._ram
        colours = COLOURS
        if ram[FLAGS] & FLAG_CHEST_OPENED:
            colours = COLOURS | {"C": OPEN_CHEST_COLOUR}
        tiles = np.array([[colours[t] for t in row] for row in ROOMS[ram[ROOM]]], dtype=np.uint8)
        tiles[ram[PLAYER_Y], ram[PLAYER_X]] = PLAYER_COLOUR
        return tiles.repeat(TILE_PIXELS, axis=0).repeat(TILE_PIXELS, axis=1)

    def read(self, address: int, size: int) -> bytes:
        offset = address - RAM_BASE
        if offset < 0 or size < 0 or offset + size > RAM_SIZE:
            raise ValueError(
                f"read of {size} bytes at {address:#010x} is outside RAM "
                f"({RAM_BASE:#010x} to {RAM_BASE + RAM_SIZE - 1:#010x})"
            )
        return bytes(self._ram[offset : offset + size])

    def save_state(self) -> bytes:
        return bytes(self._ram)

    def load_state(self, state: bytes) -> None:
        if len(state) != RAM_SIZE:
            raise ValueError(f"state is {len(state)} bytes, expected {RAM_SIZE}")
        self._ram[:] = state

    def close(self) -> None:
        pass
