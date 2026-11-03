"""Regenerate the ENV-06 start states in data/states/ from power-on.

    uv run python scripts/capture_states.py               # every stage, from power-on
    uv run python scripts/capture_states.py --from s1     # resume from data/states/s1.state

Each stage loads the state the previous stage saved, replays recorded inputs, checks the
reader's location and the options, and saves. mGBA is deterministic, so the same inputs on the
same ROM give the same bytes.

Inputs, separated by spaces; a suffix xN repeats one input N times:
    A B S s U D L R   press A, B, START, SELECT or a direction (4 frames held, 20 released)
    wN                wait N frames
    tU tD tL tR       walk one tile, then press A through any dialogue or battle it starts
    aa                press A until the player can walk again
    gN                from the Viridian Pokemon Center door, fight on Route 1 until level N
    ab                press A until a battle starts
    ff                press A until the battle ends: the first move every turn
    bb                press B until the player can walk again: it answers NO to a question
"""

import argparse
import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from algoarcade.backends.base import Button, GameBackend, read_u8, read_u32
from algoarcade.backends.mgba import MgbaBackend
from algoarcade.games.firered.addresses import ADDRESSES
from algoarcade.games.firered.reader import FireRedReader

ROOT = Path(__file__).resolve().parents[1]
ROM = ROOT / "data" / "roms" / "firered.gba"
STATES = ROOT / "data" / "states"

KEYS = {
    "A": Button.A,
    "B": Button.B,
    "S": Button.START,
    "s": Button.SELECT,
    "U": Button.UP,
    "D": Button.DOWN,
    "L": Button.LEFT,
    "R": Button.RIGHT,
}
STEP = {"U": (0, -1), "D": (0, 1), "L": (-1, 0), "R": (1, 0)}


@dataclass(frozen=True)
class Stage:
    saves: tuple[str, ...]
    source: str | None
    inputs: str
    location: tuple[int, int, int, int]


# Route 1 grass at (16, 9) and the Viridian Pokemon Center door at (26, 27), both ways. The
# second heals: walk in, talk to the nurse, walk out.
CENTER_TO_GRASS = "tDx2 tLx4 tDx10 tRx2 tD tDx4 tRx4 tDx5"
GRASS_TO_CENTER = "tUx7 tLx3 tUx2 tL tU tUx8 tLx2 tUx4 tRx4 tU tUx4 A aa tDx4 tD"

# Names match the manifest. A source of None means power-on.
STAGES = (
    Stage(
        saves=("s0", "seg_01_house"),
        source=None,
        # Title screen, the controls pages, Oak's speech, BOY, name RED on the keyboard,
        # rival GREEN from the list, then START > OPTION: text speed FAST, battle scene OFF.
        inputs="w1800 S w120 w240 "
        + "A w60 " * 8
        + "w120 "
        + "A w60 " * 8
        + "A w40 " * 8
        + "w100 "
        + "A w100 " * 8
        + "A w100 A w100 A w60 Dx2 Rx5 A Ux2 L A L A w10 S w20 A w100 w60 "
        + "A w100 " * 7
        + "D w10 "
        + "A w100 " * 7
        + "A w100 A w100 A w200 w600 S w30 Dx3 A w60 R D R w10 B w60 B w60",
        location=(4, 1, 6, 6),
    ),
    Stage(
        saves=("seg_01_house_goal",),
        source="s0",
        inputs="tRx4 tUx4 tL tDx6 tLx6 tD",
        location=(3, 0, 6, 8),
    ),
    Stage(
        saves=("seg_03_rival",),
        source="seg_01_house_goal",
        # Oak stops the player at the edge of town and walks him to the lab. The middle ball is
        # Squirtle: A to read it, A to go on, A for YES, then B declines the nickname.
        inputs="tRx6 tUx7 tD tRx3 U A w60 A w60 A w60 bb",
        location=(4, 3, 9, 5),
    ),
    Stage(
        saves=("seg_03_rival_goal",),
        source="seg_03_rival",
        # The rival stops the player at the door. A picks FIGHT, then Tackle, every turn.
        inputs="tDx2 tLx3 tD",
        location=(4, 3, 6, 8),
    ),
    Stage(
        saves=("seg_02_pallet",),
        source="seg_03_rival_goal",
        inputs="tDx4 tD w120",
        location=(3, 0, 16, 14),
    ),
    Stage(
        saves=("seg_02_pallet_goal", "seg_04_route1"),
        source="seg_02_pallet",
        inputs="tLx4 tUx15",
        location=(3, 19, 12, 39),
    ),
    Stage(
        saves=("seg_04_route1_goal",),
        source="seg_04_route1",
        inputs="tUx7 tLx4 tUx5 tRx4 tUx6 tLx2 tUx4 tRx6 tUx15 tLx3 tUx2 tL tU",
        location=(3, 1, 24, 39),
    ),
    Stage(
        saves=("s1",),
        source="seg_04_route1_goal",
        # The Mart clerk hands over the parcel on entry. Heal at the Pokemon Center, walk back
        # down Route 1 and talk to Oak.
        inputs="tUx8 tLx2 tUx11 tRx14 tU tDx4 tD tDx7 tLx10 tU tUx4 A aa tDx4 tD "
        "tDx2 tLx4 tDx10 tRx2 tD tDx4 tRx4 tDx15 tLx6 tDx5 tRx2 tDx6 tLx4 tDx5 tRx4 tDx4 tD "
        "tDx14 tRx4 tU tUx8 A w60 aa",
        location=(4, 3, 6, 4),
    ),
    Stage(
        saves=("seg_05_forest",),
        source="s1",
        # Back to Viridian, heal, grind Squirtle to level 13 (Water Gun) so the forest's wild
        # Pokemon and trainers cannot knock it out, then north. The old man at (21, 8) blocks
        # the road, so the path steps on his tutorial trigger at (20, 8).
        inputs="tDx8 tD tLx4 tUx15 "
        "tUx7 tLx4 tUx5 tRx4 tUx6 tLx2 tUx4 tRx6 tUx15 tLx3 tUx2 tL tU "
        "tUx8 tLx2 tUx4 tRx4 tU tUx4 A aa tDx4 tD g13 "
        "tLx3 tUx8 tL tUx7 tL tUx3 tL tU tUx8 tU "
        "tUx8 tR tUx4 tLx4 tUx11 tRx6 tUx3 tLx4 tU tL tU tUx8 tU w60",
        location=(1, 0, 29, 62),
    ),
    Stage(
        saves=("seg_05_forest_goal",),
        source="seg_05_forest",
        inputs="tUx3 tR tU tR tUx3 tRx8 tUx36 tLx7 tDx4 tLx9 tUx13 tLx8 tDx17 tLx8 "
        "tUx4 tL tUx10 tL tUx4",
        location=(15, 3, 7, 10),
    ),
    Stage(
        saves=("gym_door",),
        source="seg_05_forest_goal",
        # Out of the gate, up Route 2, heal at the Pewter Pokemon Center, stop below the gym door.
        inputs="tUx9 tUx11 tRx3 tUx2 tR tU tUx10 tL tUx3 tLx3 tU tUx4 A aa tDx4 tD "
        "tRx5 tUx14 tLx14 tDx5 tRx7",
        location=(3, 2, 15, 17),
    ),
    Stage(
        saves=("brock",),
        source="gym_door",
        # Column 9 and 8 stay outside the camper's four-tile sight line on row 8. The state is
        # the first "What will SQUIRTLE do?" menu against Geodude.
        inputs="tU tUx4 tRx3 tUx2 tL tUx2 tLx2 U ab w300 A w420",
        location=(6, 2, 6, 6),
    ),
    Stage(
        saves=("brock_goal",),
        source="brock",
        # Water Gun replaced Tackle in the first move slot at level 13.
        inputs="ff",
        location=(6, 2, 6, 6),
    ),
)


class Player:
    def __init__(self, backend: MgbaBackend) -> None:
        self.backend = backend
        self.reader = FireRedReader(backend.rom.sha1)
        addresses = ADDRESSES[backend.rom.sha1]
        self._main = addresses["gMain"].address
        self._lock = addresses["sLockFieldControls"].address
        self._message = addresses["sMessageBoxType"].address

    def location(self) -> tuple[int, int, int, int]:
        return self.reader.read(self.backend).location

    def _callback2(self) -> int:
        # gMain.callback2 (include/main.h): the overworld callback while the player walks.
        return read_u32(self.backend, self._main + 4)

    def _busy(self, overworld: int) -> bool:
        return bool(
            read_u8(self.backend, self._lock)
            or read_u8(self.backend, self._message)
            or self._in_battle()
            or self._callback2() != overworld
        )

    def _settle(self, overworld: int) -> None:
        for _ in range(3000):
            if not self._busy(overworld):
                return
            party = self.reader.read(self.backend).party
            if party and not any(pokemon.hp for pokemon in party):
                raise RuntimeError(f"the party fainted at {self.location()}")
            if self._callback2() == overworld or self._in_battle():
                self.backend.press([Button.A], hold=2, release=28)
            else:
                # A warp or a battle transition: wait it out.
                self.backend.step(4)
        raise RuntimeError(f"still busy at {self.location()}")

    def walk(self, direction: str) -> None:
        overworld = self._callback2()
        start = self.location()
        dx, dy = STEP[direction]
        target = (start[0], start[1], start[2] + dx, start[3] + dy)
        for _ in range(3):
            for _ in range(60):
                self.backend.press([KEYS[direction]], hold=1, release=0)
                now = self.location()
                if now != start or self._busy(overworld):
                    break
            self.backend.step(16)
            # A wild battle can begin a few frames after the step lands.
            for _ in range(8):
                if self._busy(overworld):
                    break
                self.backend.step(4)
            self._settle(overworld)
            now = self.location()
            if now[:2] != start[:2] or now == target:
                return
            if now != start:
                raise RuntimeError(f"walking {direction} from {start} ended at {now}")
        raise RuntimeError(f"could not walk {direction} from {start}")

    def grind(self, level: int) -> None:
        """From the Viridian Pokemon Center door, fight on Route 1 until the lead has `level`."""
        self.run(CENTER_TO_GRASS)
        while (lead := self.reader.read(self.backend).party[0]).level < level:
            if lead.hp * 3 < lead.max_hp:
                self.run(f"{GRASS_TO_CENTER} {CENTER_TO_GRASS}")
            else:
                self.run("tR tL")
        self.run(GRASS_TO_CENTER)

    def run(self, inputs: str) -> None:
        for token in inputs.split():
            name, _, count = token.partition("x")
            for _ in range(int(count or 1)):
                self._do(name)

    def _in_battle(self) -> bool:
        return bool(read_u8(self.backend, self._main + 0x439) & 2)

    def _has_control(self) -> bool:
        return not read_u8(self.backend, self._lock) and not read_u8(self.backend, self._message)

    def _press_until(self, button: Button, done: Callable[[], bool]) -> None:
        for _ in range(3000):
            if done():
                return
            self.backend.press([button], hold=2, release=28)
        raise RuntimeError(f"pressed {button.name} 3000 times at {self.location()}")

    def _do(self, token: str) -> None:
        if token == "aa":
            self._press_until(Button.A, self._has_control)
        elif token == "bb":
            self._press_until(Button.B, self._has_control)
        elif token == "ab":
            self._press_until(Button.A, self._in_battle)
        elif token == "ff":
            self._press_until(Button.A, lambda: not self._in_battle())
            self._press_until(Button.A, self._has_control)
        elif token[0] == "g":
            self.grind(int(token[1:]))
        elif token[0] == "w":
            self.backend.step(int(token[1:]))
        elif token[0] == "t":
            self.walk(token[1])
        else:
            self.backend.press([KEYS[token]], hold=4, release=20)


def options_ok(backend: GameBackend, rom_sha1: str) -> bool:
    """Text speed fast and battle scene off, read from SaveBlock2 through gSaveBlock2Ptr."""
    addresses = ADDRESSES[rom_sha1]
    pointer = read_u32(backend, addresses["gSaveBlock2Ptr"].address)
    base = addresses["gSaveBlock2"].address
    # SetSaveBlocksPointers moves SaveBlock2 by the same 0 to 0x7C offset as SaveBlock1.
    if not 0 <= pointer - base <= 0x7C:
        return False
    # struct SaveBlock2 in include/global.h: optionsTextSpeed is bits 0-2 of +0x14
    # (OPTIONS_TEXT_SPEED_FAST is 2) and optionsBattleSceneOff is bit 2 of +0x15.
    return read_u8(backend, pointer + 0x14) & 7 == 2 and bool(read_u8(backend, pointer + 0x15) & 4)


def play(backend: MgbaBackend, stage: Stage) -> bytes:
    if stage.source is not None:
        backend.load_state((STATES / f"{stage.source}.state").read_bytes())
    player = Player(backend)
    player.run(stage.inputs)
    location = player.location()
    if location != stage.location:
        raise RuntimeError(f"{stage.saves[0]}: expected {stage.location}, got {location}")
    if not options_ok(backend, backend.rom.sha1):
        raise RuntimeError(f"{stage.saves[0]}: options are not text fast and battle scene off")
    return backend.save_state()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Regenerate the start states from power-on.")
    parser.add_argument("--from", dest="start", help="first stage to play, by saved name")
    args = parser.parse_args(argv)
    names = [stage.saves[0] for stage in STAGES]
    first = names.index(args.start) if args.start else 0
    STATES.mkdir(parents=True, exist_ok=True)
    for stage in STAGES[first:]:
        backend = MgbaBackend(ROM)
        try:
            state = play(backend, stage)
            location = Player(backend).location()
        finally:
            backend.close()
        for name in stage.saves:
            (STATES / f"{name}.state").write_bytes(state)
            digest = hashlib.sha256(state).hexdigest()
            print(f"{name}  {digest}  {location}")


if __name__ == "__main__":
    main()
