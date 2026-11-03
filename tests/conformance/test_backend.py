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
from algoarcade.backends.mgba import EWRAM_BASE, EWRAM_SIZE, MgbaBackend


def make_mgba() -> GameBackend:
    path = Path(__file__).resolve().parents[2] / "data" / "roms" / "firered.gba"
    if not path.is_file():
        pytest.skip("Place your FireRed ROM in data/roms/firered.gba")
    pytest.importorskip(
        "mgba.core", reason="mGBA binding did not load; see docs/setup.md", exc_type=ImportError
    )
    return MgbaBackend(path)


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
    "mgba": BackendCase(
        make=make_mgba,
        ram=(EWRAM_BASE, EWRAM_SIZE),
        unmapped=EWRAM_BASE + EWRAM_SIZE,
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


@pytest.fixture(
    params=[
        pytest.param(value, id=name, marks=pytest.mark.rom if name == "mgba" else ())
        for name, value in BACKENDS.items()
    ]
)
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
