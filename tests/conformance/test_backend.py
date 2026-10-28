"""Every backend passes this suite. Contract: docs/harness-spec.md, Conformance tests.

To add a backend, add one entry to BACKENDS. A factory that needs a ROM calls pytest.skip when
the file is absent, and its entry is wrapped in pytest.param(..., marks=pytest.mark.rom).
"""

import hashlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from algoarcade.backends import fake
from algoarcade.backends.base import Button, GameBackend
from algoarcade.backends.pyboy import ADDRESS_SPACE, PyBoyBackend

SUPER_MARIO_LAND_ROM = Path(__file__).parents[2] / "data/roms/super_mario_land.gb"


def make_pyboy() -> GameBackend:
    if not SUPER_MARIO_LAND_ROM.exists():
        pytest.skip(f"no ROM at {SUPER_MARIO_LAND_ROM}")
    backend = PyBoyBackend(SUPER_MARIO_LAND_ROM)
    backend.step(120)  # WRAM stays zero until the game initialises it, about frame 96
    return backend


@dataclass(frozen=True)
class BackendCase:
    make: Callable[[], GameBackend]
    ram: tuple[int, int]  # address and size of a block that holds the game's state
    unmapped: int  # an address read() must reject


BACKENDS = {
    "fake": BackendCase(
        make=fake.FakeBackend,
        ram=(fake.RAM_BASE, fake.RAM_SIZE),
        unmapped=fake.RAM_BASE + fake.RAM_SIZE,
    ),
    "pyboy": pytest.param(
        BackendCase(make=make_pyboy, ram=(0xC000, 0x2000), unmapped=ADDRESS_SPACE),
        marks=pytest.mark.rom,
    ),
}

INPUTS: list[list[Button]] = [
    [Button.RIGHT],
    [Button.DOWN],
    [],
    [Button.A],
    [Button.DOWN, Button.B],
    [Button.LEFT],
    [Button.START],
    [Button.UP],
]


@pytest.fixture(params=list(BACKENDS.values()), ids=list(BACKENDS))
def case(request: pytest.FixtureRequest) -> BackendCase:
    return request.param


@pytest.fixture
def backend(case: BackendCase) -> Iterator[GameBackend]:
    backend = case.make()
    yield backend
    backend.close()


def ram_hashes(backend: GameBackend, case: BackendCase) -> list[str]:
    hashes = []
    for buttons in INPUTS:
        backend.press(buttons)
        hashes.append(hashlib.sha1(backend.read(*case.ram)).hexdigest())
    return hashes


def test_same_state_and_inputs_give_the_same_ram_at_every_step(
    backend: GameBackend, case: BackendCase
) -> None:
    backend.step(7)
    state = backend.save_state()
    first = ram_hashes(backend, case)
    backend.load_state(state)
    second = ram_hashes(backend, case)
    assert first == second
    assert len(set(first)) > 1, "RAM never changed, so this run proves nothing"


def test_save_then_load_restores_ram(backend: GameBackend, case: BackendCase) -> None:
    backend.press([Button.RIGHT])
    state = backend.save_state()
    saved_ram = backend.read(*case.ram)
    backend.press([Button.DOWN])
    backend.step(5)
    assert backend.read(*case.ram) != saved_ram
    backend.load_state(state)
    assert backend.read(*case.ram) == saved_ram


def test_screen_is_rgb_uint8_and_a_copy(backend: GameBackend) -> None:
    screen = backend.screen()
    assert screen.ndim == 3
    assert screen.shape[2] == 3
    assert screen.dtype == np.uint8
    original = screen.copy()
    screen += 1
    np.testing.assert_array_equal(backend.screen(), original)


def test_read_outside_memory_raises(backend: GameBackend, case: BackendCase) -> None:
    with pytest.raises(ValueError):  # noqa: PT011 the message is the backend's own
        backend.read(case.unmapped, 1)
