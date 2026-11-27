from collections.abc import Iterator
from dataclasses import fields, replace
from itertools import pairwise
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from algoarcade.backends.base import Button, GameBackend, read_u8
from algoarcade.backends.pyboy import PyBoyBackend
from algoarcade.games.base import StateNotReady
from algoarcade.games.super_mario_land.reader import (
    COLUMN_BIT,
    LEVEL_BLOCK,
    LIVES_LEFT,
    MARIO_X,
    MARIO_Y,
    SCROLL_X,
    TIME_LEFT,
    WORLD_AND_LEVEL,
    SuperMarioLandReader,
    SuperMarioLandState,
)
from algoarcade.rewards.explore import Explore

SUPER_MARIO_LAND_ROM = Path(__file__).parents[1] / "data/roms/super_mario_land.gb"
ACTIONS = [Button.UP, Button.DOWN, Button.LEFT, Button.RIGHT, Button.A, Button.B]
READS = (
    TIME_LEFT,
    TIME_LEFT + 1,
    LIVES_LEFT,
    WORLD_AND_LEVEL,
    LEVEL_BLOCK,
    SCROLL_X,
    COLUMN_BIT,
    MARIO_X,
    MARIO_Y,
)
FIXED = {
    TIME_LEFT: 0x45,
    TIME_LEFT + 1: 0x03,
    LIVES_LEFT: 0x12,
    WORLD_AND_LEVEL: 0x23,
    LEVEL_BLOCK: 5,
    SCROLL_X: 3,
    COLUMN_BIT: 8,
    MARIO_X: 40,
    MARIO_Y: 112,
}


class Memory:
    def __init__(self, values: dict[int, int]) -> None:
        self.values = values

    def read(self, address: int, size: int) -> bytes:
        return bytes(self.values[a] for a in range(address, address + size))


def memory(values: dict[int, int]) -> GameBackend:
    return cast(GameBackend, Memory(values))


@pytest.fixture
def backend() -> Iterator[PyBoyBackend]:
    if not SUPER_MARIO_LAND_ROM.exists():
        pytest.skip(f"no ROM at {SUPER_MARIO_LAND_ROM}")
    backend = PyBoyBackend(SUPER_MARIO_LAND_ROM)
    backend._pyboy.game_wrapper.start_game(timer_div=0)
    yield backend
    backend.close()


def test_state_has_no_score_or_coins() -> None:
    assert {field.name for field in fields(SuperMarioLandState)} == {
        "lives_left",
        "time_left",
        "world",
        "level_progress",
        "mario_y",
        "x_tile",
    }


def test_reader_decodes_fixed_bytes() -> None:
    state = SuperMarioLandReader().read(memory(FIXED))
    assert state == SuperMarioLandState(
        lives_left=12,
        time_left=345,
        world=(2, 3),
        level_progress=5 * 16 + 12 + 40,  # (3 - 7) % 16 is 12
        mario_y=112,
        x_tile=2 * 5 + 1 + 40 // 8,
    )
    assert state.location == (2, 3, 16, 14)


@pytest.mark.parametrize("world_and_level", [0x00, 0x10, 0x14, 0x01, 0x51])
def test_reader_is_not_ready_before_init(world_and_level: int) -> None:
    with pytest.raises(StateNotReady):
        SuperMarioLandReader().read(memory(FIXED | {WORLD_AND_LEVEL: world_and_level}))


def test_x_tile_never_dips_while_the_screen_scrolls() -> None:
    # Call_2198 (bank0.asm lines 5098-5113) over 64 pixels of scrolling. Each frame moves
    # hScrollX by one and is read twice: before the call, a torn frame, and after it.
    values = FIXED | {SCROLL_X: 0, COLUMN_BIT: 8, LEVEL_BLOCK: 0}
    x_tiles, level_progress = [], []
    for _ in range(64):
        values[SCROLL_X] += 1
        for torn in (True, False):
            if not torn and values[SCROLL_X] & 8 == values[COLUMN_BIT]:
                values[COLUMN_BIT] ^= 8
                if values[COLUMN_BIT] == 0:
                    values[LEVEL_BLOCK] += 1
            state = SuperMarioLandReader().read(memory(values))
            x_tiles.append(state.x_tile)
            level_progress.append(state.level_progress)
    assert all(b - a in (0, 1) for a, b in pairwise(x_tiles))
    assert x_tiles[-1] - x_tiles[0] == 8
    assert any(b - a == -15 for a, b in pairwise(level_progress))


def test_jumping_in_place_pays_explore_once_per_y_tile() -> None:
    standing = SuperMarioLandReader().read(memory(FIXED | {MARIO_Y: 0x78}))
    arc = [*range(0x78, 0x48, -4), *range(0x48, 0x79, 4)]
    jump = [replace(standing, mario_y=y) for y in arc]
    explore = Explore()
    explore.reset(standing)
    paid = [explore.score(prev, curr) for prev, curr in pairwise([standing, *jump, *jump])]
    assert sum(paid) == len({y // 8 for y in arc}) - 1 == 6
    assert sum(paid[len(jump) :]) == 0


@pytest.mark.rom
def test_mario_y_changes_when_mario_jumps(backend: PyBoyBackend) -> None:
    reader = SuperMarioLandReader()
    standing = reader.read(backend).mario_y
    backend.press([Button.A], hold=8, release=0)
    assert reader.read(backend).mario_y != standing


@pytest.mark.rom
def test_reader_matches_the_pyboy_wrapper_over_1000_random_steps(backend: PyBoyBackend) -> None:
    pyboy = backend._pyboy
    wrapper = pyboy.game_wrapper
    reader = SuperMarioLandReader()
    rng = np.random.default_rng(0)
    states = []
    mismatches = []
    still_scrolls = []
    scroll = read_u8(backend, SCROLL_X)
    for step in range(1000):
        backend.press([ACTIONS[rng.integers(len(ACTIONS))]])
        state = reader.read(backend)
        states.append(state)
        drawn_scroll = pyboy.screen.tilemap_position_list[16][0]
        # hScrollX moves only while the screen scrolls. It wraps at 256 and resets on a death or
        # a new level; coming back to the same value within 24 frames would take a wrap, over 10
        # pixels a frame. So if it reads the same before and after a step, the step ended without
        # scrolling and LCDStatus copied it to rSCX on the last frame: PyBoy drew this value.
        previous, scroll = scroll, read_u8(backend, SCROLL_X)
        if scroll == previous:
            still_scrolls.append(scroll)
            if scroll != drawn_scroll:
                mismatches.append((step, "hScrollX", scroll, drawn_scroll))
        mismatches.extend(
            (step, field, getattr(state, field), getattr(wrapper, field))
            for field in ("lives_left", "time_left", "world")
            if getattr(state, field) != getattr(wrapper, field)
        )
        # The wrapper takes the scroll PyBoy drew at scanline 16. While the screen scrolls, that
        # is one frame behind hScrollX, and no RAM byte holds it. So the reader runs on this
        # step's bytes with the drawn scroll in place of hScrollX, and must equal the wrapper.
        drawn = memory({a: read_u8(backend, a) for a in READS} | {SCROLL_X: drawn_scroll})
        level_progress = reader.read(drawn).level_progress
        if level_progress != wrapper.level_progress:
            mismatches.append((step, "level_progress", level_progress, wrapper.level_progress))
        # Derived from Call_2198, not observed: x_tile is within one tile of the wrapper's.
        if abs(state.x_tile - wrapper.level_progress // 8) > 1:
            mismatches.append((step, "x_tile", state.x_tile, wrapper.level_progress // 8))
    assert len(set(states)) > 1, "the state never changed, so this run proves nothing"
    assert len({state.mario_y for state in states}) > 1, "mario_y never changed"
    # 500 is a guess: in the df5dc67 run the screen scrolled only in steps 939-968.
    assert len(still_scrolls) >= 500, f"only {len(still_scrolls)} steps compared hScrollX"
    assert any(still_scrolls), "hScrollX was only compared at 0"
    assert not mismatches, f"(step, field, reader, wrapper): {mismatches}"
