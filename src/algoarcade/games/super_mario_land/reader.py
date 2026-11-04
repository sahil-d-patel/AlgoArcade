"""State reader for Super Mario Land (World) rev A, SHA-1 418203621b887caa090215d97e3f509b79affd3e.

Score and coins are left out because PyBoy's wrapper reads them from screen tiles, so there is
no oracle to test them against. The disassembly has them in RAM: wScore at 0xC0A0 (3 bytes,
wram.asm) and hCoins at 0xFFFA (hram.asm).
"""

from dataclasses import dataclass

from algoarcade.backends.base import GameBackend, read_u8
from algoarcade.games.base import StateNotReady

# Labels are from the kaspermeerts/supermarioland disassembly at commit
# 618d00ed6c330928e106719533c6e294ae5d5726, whose rom.sha1 is our dump. PyBoy 2.8.1
# (pyboy/plugins/game_wrapper_super_mario_land.py) reads the same addresses.
TIME_LEFT = 0xDA01  # wGameTimer + 1, wram.asm. BCD: seconds 0-99, then hundreds.
LIVES_LEFT = 0xDA15  # wLives, wram.asm. BCD.
WORLD_AND_LEVEL = 0xFFB4  # hWorldAndLevel, hram.asm. World in the high nibble, level in the low.
# hScrollX, hram.asm. LCDStatus in bank0.asm copies it to rSCX after scanline 16. The wrapper
# never reads it; the ROM oracle test checks it against the scroll PyBoy drew.
SCROLL_X = 0xFFA4
# No label in the disassembly. LEVEL_BLOCK and MARIO_X: PyBoy v2.8.1, commit
# f97d75162df5dafa82db50850d04497013cf215b, pyboy/plugins/game_wrapper_super_mario_land.py
# lines 360-361, in post_tick. MARIO_Y: the disassembly's comments only. Logged as exceptions
# in docs/decisions.md.
LEVEL_BLOCK = 0xC0AB  # Counts 16-pixel blocks scrolled; bank0.asm calls it "progress" in level.
MARIO_X = 0xC202  # Mario's x on screen. bank0.asm: "Mario X pos".
MARIO_Y = 0xC201  # Mario's y on screen. bank0.asm: "Mario Y pos". Not checked against the game.
# No label. Call_2198 (bank0.asm lines 5098-5113) loads the next 8-pixel tile column each time
# bit 3 of hScrollX reaches this byte, then flips it between 0 and 8, and every second flip it
# also adds 1 to LEVEL_BLOCK. Every write in the disassembly is 0 or that flip.
COLUMN_BIT = 0xFFA3


@dataclass(frozen=True)
class SuperMarioLandState:
    lives_left: int
    time_left: int
    world: tuple[int, int]
    level_progress: int
    mario_y: int
    x_tile: int

    @property
    def location(self) -> tuple[int, int, int, int]:
        return (*self.world, self.x_tile, self.mario_y // 8)


def bcd(byte: int) -> int:
    return (byte >> 4) * 10 + (byte & 0x0F)


class SuperMarioLandReader:
    def read(self, backend: GameBackend) -> SuperMarioLandState:
        seconds, hundreds = backend.read(TIME_LEFT, 2)
        world_and_level = read_u8(backend, WORLD_AND_LEVEL)
        world, level = world_and_level >> 4, world_and_level & 0x0F
        # Init (bank0.asm) clears HRAM, then sets 0x11. Every later write is world 1-4, level 1-3.
        if not (1 <= world <= 4 and 1 <= level <= 3):
            raise StateNotReady(f"hWorldAndLevel is {world_and_level:#04x}: Init has not run")
        level_block = read_u8(backend, LEVEL_BLOCK)
        mario_x = read_u8(backend, MARIO_X)
        # The wrapper's formula. It takes the scroll from the line PyBoy drew, which lags hScrollX
        # by one frame while the screen scrolls, so the two can differ on such a frame. It also
        # reads 15 low while hScrollX is 7 mod 16: the term wraps at 7, but Call_2198 adds to
        # LEVEL_BLOCK only at 8. So the location uses x_tile instead.
        level_progress = level_block * 16 + (read_u8(backend, SCROLL_X) - 7) % 16 + mario_x
        # Columns loaded plus Mario's screen tile. Call_2198 writes both bytes in one call, and
        # 2 * LEVEL_BLOCK + COLUMN_BIT // 8 only rises within a level, so the scroll part never
        # dips; it can lag hScrollX by a column for a frame.
        x_tile = 2 * level_block + read_u8(backend, COLUMN_BIT) // 8 + mario_x // 8
        return SuperMarioLandState(
            lives_left=bcd(read_u8(backend, LIVES_LEFT)),
            time_left=bcd(seconds) + bcd(hundreds) * 100,
            world=(world, level),
            level_progress=level_progress,
            mario_y=read_u8(backend, MARIO_Y),
            x_tile=x_tile,
        )
