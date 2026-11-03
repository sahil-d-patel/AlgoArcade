from collections.abc import Iterator
from dataclasses import fields
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from algoarcade.backends.base import Button, GameBackend, read_u8
from algoarcade.backends.pyboy import PyBoyBackend
from algoarcade.games.super_mario_land.reader import (
    LEVEL_BLOCK,
    LIVES_LEFT,
    MARIO_X,
    SCROLL_X,
    TIME_LEFT,
    WORLD_AND_LEVEL,
    SuperMarioLandReader,
    SuperMarioLandState,
)

SUPER_MARIO_LAND_ROM = Path(__file__).parents[1] / "data/roms/super_mario_land.gb"
ACTIONS = [Button.UP, Button.DOWN, Button.LEFT, Button.RIGHT, Button.A, Button.B]


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
    }


def test_reader_decodes_fixed_bytes() -> None:
    state = SuperMarioLandReader().read(
        memory(
            {
                TIME_LEFT: 0x45,
                TIME_LEFT + 1: 0x03,
                LIVES_LEFT: 0x12,
                WORLD_AND_LEVEL: 0x23,
                LEVEL_BLOCK: 5,
                SCROLL_X: 3,
                MARIO_X: 40,
            }
        )
    )
    assert state == SuperMarioLandState(
        lives_left=12,
        time_left=345,
        world=(2, 3),
        level_progress=5 * 16 + 12 + 40,  # (3 - 7) % 16 is 12
    )
    assert state.location == (2, 3, 132)


@pytest.mark.rom
def test_reader_matches_the_pyboy_wrapper_over_1000_random_steps(backend: PyBoyBackend) -> None:
    pyboy = backend._pyboy
    wrapper = pyboy.game_wrapper
    reader = SuperMarioLandReader()
    rng = np.random.default_rng(0)
    states = []
    mismatches = []
    for step in range(1000):
        backend.press([ACTIONS[rng.integers(len(ACTIONS))]])
        state = reader.read(backend)
        states.append(state)
        mismatches.extend(
            (step, field, getattr(state, field), getattr(wrapper, field))
            for field in ("lives_left", "time_left", "world")
            if getattr(state, field) != getattr(wrapper, field)
        )
        # The wrapper takes the scroll PyBoy drew at scanline 16. While the screen scrolls, that
        # is one frame behind hScrollX, and no RAM byte holds it. So the reader runs on this
        # step's bytes with the drawn scroll in place of hScrollX, and must equal the wrapper.
        drawn = memory(
            {
                address: read_u8(backend, address)
                for address in (
                    TIME_LEFT,
                    TIME_LEFT + 1,
                    LIVES_LEFT,
                    WORLD_AND_LEVEL,
                    LEVEL_BLOCK,
                    MARIO_X,
                )
            }
            | {SCROLL_X: pyboy.screen.tilemap_position_list[16][0]}
        )
        level_progress = reader.read(drawn).level_progress
        if level_progress != wrapper.level_progress:
            mismatches.append((step, "level_progress", level_progress, wrapper.level_progress))
    assert len(set(states)) > 1, "the state never changed, so this run proves nothing"
    assert not mismatches, f"(step, field, reader, wrapper): {mismatches}"
