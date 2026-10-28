from collections.abc import Iterator
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from algoarcade.backends.base import Button, read_u8
from algoarcade.backends.pyboy import PyBoyBackend
from algoarcade.games.super_mario_land.reader import (
    LEVEL_BLOCK,
    MARIO_X,
    SuperMarioLandReader,
    SuperMarioLandState,
)

SUPER_MARIO_LAND_ROM = Path(__file__).parents[1] / "data/roms/super_mario_land.gb"
ACTIONS = [Button.UP, Button.DOWN, Button.LEFT, Button.RIGHT, Button.A, Button.B]


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


@pytest.mark.rom
def test_reader_matches_the_pyboy_wrapper_over_1000_random_steps(backend: PyBoyBackend) -> None:
    pyboy = backend._pyboy
    wrapper = pyboy.game_wrapper
    reader = SuperMarioLandReader()
    rng = np.random.default_rng(0)
    states = []
    other_mismatches = []
    explained = []
    unexplained = []
    for step in range(1000):
        backend.press([ACTIONS[rng.integers(len(ACTIONS))]])
        state = reader.read(backend)
        states.append(state)
        other_mismatches.extend(
            (step, field, getattr(state, field), getattr(wrapper, field))
            for field in ("lives_left", "time_left", "world")
            if getattr(state, field) != getattr(wrapper, field)
        )
        if state.level_progress != wrapper.level_progress:
            # The wrapper takes the scroll PyBoy drew at scanline 16. While the screen scrolls,
            # that is one frame behind hScrollX (0xFFA4), and no RAM byte holds it. A difference
            # is allowed only if the reader's formula with the drawn scroll gives the wrapper's value.
            drawn_scroll = pyboy.screen.tilemap_position_list[16][0]
            with_drawn_scroll = (
                read_u8(backend, LEVEL_BLOCK) * 16
                + (drawn_scroll - 7) % 16
                + read_u8(backend, MARIO_X)
            )
            lagging = with_drawn_scroll == wrapper.level_progress
            (explained if lagging else unexplained).append(
                (step, state.level_progress, wrapper.level_progress)
            )
    print(  # noqa: T201
        f"level_progress differences: {len(explained) + len(unexplained)}, "
        f"explained by scroll lag: {len(explained)}, unexplained: {len(unexplained)}"
    )
    assert len(set(states)) > 1, "the state never changed, so this run proves nothing"
    assert not other_mismatches, f"lives, time or world differ: {other_mismatches}"
    assert not unexplained, f"level_progress differs with no scroll lag: {unexplained}"
