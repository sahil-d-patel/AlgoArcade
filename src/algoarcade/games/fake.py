"""State reader for the toy game in algoarcade.backends.fake."""

from collections.abc import Callable
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
    party_count: int
    party_hp_frac: float
    text_box_open: bool
    text_progress: int

    @property
    def location(self) -> tuple[int, int, int]:
        return (self.room, self.x, self.y)


class FakeReader:
    def read(self, backend: GameBackend) -> FakeState:
        flags = read_u8(backend, fake.RAM_BASE + fake.FLAGS)
        text = read_u8(backend, fake.RAM_BASE + fake.TEXT)
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
            # The fake has a level sum but no party; it stands for one Pokemon.
            party_count=1,
            party_hp_frac=read_u8(backend, fake.RAM_BASE + fake.HP_PERCENT) / 100,
            text_box_open=text != 0,
            text_progress=text,
        )


# The observation vector for AlgoArcadeEnv, each value scaled into [0, 1].
FEATURES: dict[str, Callable[[FakeState], float]] = {
    "room": lambda s: s.room,
    "x": lambda s: s.x / (fake.WIDTH - 1),
    "y": lambda s: s.y / (len(fake.ROOMS[0]) - 1),
    "chest_opened": lambda s: s.chest_opened,
}
