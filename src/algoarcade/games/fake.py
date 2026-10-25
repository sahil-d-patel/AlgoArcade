"""State reader for the toy game in algoarcade.backends.fake."""

from dataclasses import dataclass

from algoarcade.backends import fake
from algoarcade.backends.base import GameBackend, read_u8, read_u32


@dataclass(frozen=True)
class FakeState:
    room: int
    x: int
    y: int
    chest_opened: bool
    frame: int
    story_flags: int
    party_level_sum: int
    badge_count: int
    milestone_count: int

    @property
    def location(self) -> tuple[int, int, int]:
        return (self.room, self.x, self.y)


class FakeReader:
    def read(self, backend: GameBackend) -> FakeState:
        flags = read_u8(backend, fake.RAM_BASE + fake.FLAGS)
        return FakeState(
            room=read_u8(backend, fake.RAM_BASE + fake.ROOM),
            x=read_u8(backend, fake.RAM_BASE + fake.PLAYER_X),
            y=read_u8(backend, fake.RAM_BASE + fake.PLAYER_Y),
            chest_opened=bool(flags & fake.FLAG_CHEST_OPENED),
            frame=read_u32(backend, fake.RAM_BASE + fake.FRAME),
            story_flags=flags.bit_count(),
            party_level_sum=read_u8(backend, fake.RAM_BASE + fake.LEVEL_SUM),
            badge_count=read_u8(backend, fake.RAM_BASE + fake.BADGES).bit_count(),
            milestone_count=read_u8(backend, fake.RAM_BASE + fake.MILESTONES),
        )
