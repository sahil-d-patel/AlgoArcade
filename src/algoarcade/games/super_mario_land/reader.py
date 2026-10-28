"""State reader for Super Mario Land (World) rev A, SHA-1 418203621b887caa090215d97e3f509b79affd3e.

Score and coins are left out on purpose: the game keeps them only as tiles on screen, not in RAM.
"""

from dataclasses import dataclass

from algoarcade.backends.base import GameBackend, read_u8

# Labels are from the kaspermeerts/supermarioland disassembly at commit
# 618d00ed6c330928e106719533c6e294ae5d5726, whose rom.sha1 is our dump. PyBoy 2.8.1
# (pyboy/plugins/game_wrapper_super_mario_land.py) reads the same addresses.
TIME_LEFT = 0xDA01  # wGameTimer + 1, wram.asm. BCD: seconds 0-99, then hundreds.
LIVES_LEFT = 0xDA15  # wLives, wram.asm. BCD.
WORLD_AND_LEVEL = 0xFFB4  # hWorldAndLevel, hram.asm. World in the high nibble, level in the low.
SCROLL_X = 0xFFA4  # hScrollX, hram.asm. The game copies it to rSCX at scanline 16.
# No label in the disassembly. Source: PyBoy 2.8.1 game_wrapper_super_mario_land.py, which reads
# them in post_tick. Logged as an exception in docs/decisions.md.
LEVEL_BLOCK = 0xC0AB  # Counts 16-pixel blocks scrolled; bank0.asm calls it "progress" in level.
MARIO_X = 0xC202  # Mario's x on screen.


@dataclass(frozen=True)
class SuperMarioLandState:
    lives_left: int
    time_left: int
    world: tuple[int, int]
    level_progress: int

    @property
    def location(self) -> tuple[int, int, int]:
        return (*self.world, self.level_progress)


def bcd(byte: int) -> int:
    return (byte >> 4) * 10 + (byte & 0x0F)


class SuperMarioLandReader:
    def read(self, backend: GameBackend) -> SuperMarioLandState:
        seconds, hundreds = backend.read(TIME_LEFT, 2)
        world_and_level = read_u8(backend, WORLD_AND_LEVEL)
        # The wrapper's formula. It takes the scroll from the line PyBoy drew, which lags hScrollX
        # by one frame while the screen scrolls, so the two can differ on such a frame.
        level_progress = (
            read_u8(backend, LEVEL_BLOCK) * 16
            + (read_u8(backend, SCROLL_X) - 7) % 16
            + read_u8(backend, MARIO_X)
        )
        return SuperMarioLandState(
            lives_left=bcd(read_u8(backend, LIVES_LEFT)),
            time_left=bcd(seconds) + bcd(hundreds) * 100,
            world=(world_and_level >> 4, world_and_level & 0x0F),
            level_progress=level_progress,
        )
