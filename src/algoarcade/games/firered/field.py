"""The overworld memory the agent's L2 tools read. Contract: docs/harness-spec.md, FireRed
memory map. Layouts are from pret/pokefirered at LAYOUT_COMMIT."""

import struct

import numpy as np
from numpy.typing import NDArray

from algoarcade.backends.base import Button, GameBackend
from algoarcade.games.base import StateNotReady
from algoarcade.games.firered.addresses import ADDRESSES
from algoarcade.games.firered.maps import MapGrid
from algoarcade.games.firered.reader import FireRedState

# include/fieldmap.h: VMap holds the map with a 7-tile border, and object coordinates carry the
# same offset (MAP_OFFSET). The right border is one wider: MAP_OFFSET_W is MAP_OFFSET * 2 + 1.
MAP_OFFSET = 7
# The pinned .sym gives gBackupMapData 0x5000 bytes, gStringVar4 0x3E8 and
# gDisplayedStringBattle 0x12C.
BACKUP_MAP_BYTES = 0x5000
STRING_VAR4_BYTES = 0x3E8
BATTLE_STRING_BYTES = 0x12C
# include/global.fieldmap.h: a block is metatile id (bits 0-9), collision (10-11) and elevation
# (12-15); an undefined block is 0x03FF.
COLLISION_MASK = 0x0C00
UNDEFINED_BLOCK = 0x03FF
# struct ObjectEvent: 16 of 0x24 bytes. active is bit 0 and isPlayer bit 16 of the flags word,
# currentElevation the low nibble of +0x0B, currentCoords +0x10, previousCoords +0x14.
OBJECT_EVENTS = 16
OBJECT_EVENT_BYTES = 0x24
# include/task.h: struct Task is func (4), isActive, prev, next, priority, then 16 s16 of data.
TASKS = 16
TASK_BYTES = 0x28
# The input tasks of the script menus (src/script_menu.c) a dialogue can open: yes/no,
# multichoice and the multichoice grid.
MENU_TASKS = (
    "Task_YesNoMenu_HandleInput",
    "Task_MultichoiceMenu_HandleInput",
    "Hask_MultichoiceGridMenu_HandleInput",
)

# charmap.txt at LAYOUT_COMMIT, the characters English text uses. 0xFA, 0xFB and 0xFE end a
# line or a page.
CHARACTERS = {
    0x00: " ",
    0x1B: "é",
    0x2D: "&",
    0x2E: "+",
    0x35: "=",
    0x36: ";",
    0x5B: "%",
    0x5C: "(",
    0x5D: ")",
    0x85: "<",
    0x86: ">",
    0xF0: ":",
    0xFA: "\n",
    0xFB: "\n",
    0xFE: "\n",
} | dict(
    zip(
        range(0xA1, 0xEF),
        "0123456789!?.-·…“”\u2018\u2019♂♀¥,\u00d7/ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
        strict=True,
    )
)
# MB_JUMP_EAST to MB_JUMP_SOUTH (include/constants/metatile_behaviors.h): the direction a ledge
# can be jumped.
JUMPS = {0x38: Button.RIGHT, 0x39: Button.LEFT, 0x3A: Button.UP, 0x3B: Button.DOWN}
EOS = 0xFF
EXT_CTRL_CODE = 0xFC
# Characters followed by one argument byte: a placeholder, a dynamic character, a keypad icon
# and an extra symbol (include/characters.h).
ONE_ARGUMENT = (0xFD, 0xF7, 0xF8, 0xF9)
# GetExtCtrlCodeLength in src/string_util.c: bytes after 0xFC, the code included.
EXT_CTRL_CODE_LENGTHS = (1, 2, 2, 2, 4, 2, 2, 1, 2, 1, 1, 3, 2, 2, 2, 1, 3, 2, 2, 2, 2, 1, 1, 1, 1)


def decode(raw: bytes) -> str:
    """A game string up to its terminator, without control codes or unknown characters."""
    text = []
    index = 0
    while index < len(raw) and raw[index] != EOS:
        byte = raw[index]
        if byte == EXT_CTRL_CODE:
            code = raw[index + 1] if index + 1 < len(raw) else 0
            index += 1 + (EXT_CTRL_CODE_LENGTHS[code] if code < len(EXT_CTRL_CODE_LENGTHS) else 0)
            continue
        if byte in ONE_ARGUMENT:
            index += 2
            continue
        text.append(CHARACTERS.get(byte, ""))
        index += 1
    return "".join(text)


class FireRedField:
    """`maps` is ENV-10's offline map data (`maps.load_maps`). Without it no ledge is known, and
    a ledge is only a wall: its block carries collision 1."""

    def __init__(self, rom_sha1: str, maps: dict[tuple[int, int], MapGrid] | None = None) -> None:
        self._addresses = {name: entry.address for name, entry in ADDRESSES[rom_sha1].items()}
        self._menus = {self._addresses[name] for name in MENU_TASKS}
        self._maps = maps or {}

    def walkable(self, backend: GameBackend) -> NDArray[np.bool_]:
        """The current map as [y, x] in map-local tiles: True where collision is 0, the elevation
        matches the player's as IsElevationMismatchAt allows, and no other object stands now or
        stood before its last step (DoesObjectCollideWithObjectAt). Ledges and other one-way
        tiles are not marked."""
        width, height, pointer = struct.unpack("<iiI", backend.read(self._addresses["VMap"], 12))
        if pointer != self._addresses["gBackupMapData"] or not (
            0 < width * height * 2 <= BACKUP_MAP_BYTES
        ):
            raise StateNotReady(f"VMap is {width}x{height} at {pointer:#010x}")
        blocks = np.frombuffer(backend.read(pointer, width * height * 2), "<u2")
        blocks = blocks.reshape(height, width)[
            MAP_OFFSET : height - MAP_OFFSET, MAP_OFFSET : width - MAP_OFFSET - 1
        ]
        elevations = blocks >> 12
        objects = backend.read(self._addresses["gObjectEvents"], OBJECT_EVENTS * OBJECT_EVENT_BYTES)
        occupied = []
        player_elevation = 0
        for index in range(OBJECT_EVENTS):
            entry = objects[index * OBJECT_EVENT_BYTES : (index + 1) * OBJECT_EVENT_BYTES]
            flags = int.from_bytes(entry[:4], "little")
            if not flags & 1:
                continue
            if flags >> 16 & 1:
                player_elevation = entry[0x0B] & 0x0F
                continue
            x, y, last_x, last_y = struct.unpack_from("<hhhh", entry, 0x10)
            occupied += [
                (x - MAP_OFFSET, y - MAP_OFFSET),
                (last_x - MAP_OFFSET, last_y - MAP_OFFSET),
            ]
        walkable = (blocks & COLLISION_MASK == 0) & (blocks != UNDEFINED_BLOCK)
        if player_elevation:
            walkable &= (elevations == 0) | (elevations == 15) | (elevations == player_elevation)
        rows, columns = walkable.shape
        for x, y in occupied:
            if 0 <= x < columns and 0 <= y < rows:
                walkable[y, x] = False
        return walkable

    def text(self, backend: GameBackend, state: FireRedState) -> str:
        """The message the text box holds: gStringVar4 in the field, where ShowFieldMessage
        expands every message, and gDisplayedStringBattle in battle."""
        if state.in_battle:
            raw = backend.read(self._addresses["gDisplayedStringBattle"], BATTLE_STRING_BYTES)
        else:
            raw = backend.read(self._addresses["gStringVar4"], STRING_VAR4_BYTES)
        return decode(raw)

    def choice(self, backend: GameBackend) -> bool:
        """A script menu waits for an answer: one of its input tasks is active in gTasks."""
        tasks = backend.read(self._addresses["gTasks"], TASKS * TASK_BYTES)
        for index in range(TASKS):
            function, active = struct.unpack_from("<IB", tasks, index * TASK_BYTES)
            # A Thumb function pointer has bit 0 set.
            if active and function & ~1 in self._menus:
                return True
        return False

    def battle(self, state: FireRedState) -> bool:
        return state.in_battle

    def ledges(self, state: FireRedState) -> dict[tuple[int, int], Button]:
        grid = self._maps.get((state.map_group, state.map_num))
        if grid is None:
            return {}
        ys, xs = np.nonzero(grid.ledge)
        return {
            (int(x), int(y)): JUMPS[int(grid.behavior[y, x])] for x, y in zip(xs, ys, strict=True)
        }
