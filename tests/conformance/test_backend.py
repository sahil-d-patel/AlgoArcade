"""Every backend passes this suite. Contract: docs/harness-spec.md, Conformance tests.

To add a backend, add one entry to BACKENDS. A factory that needs a ROM calls pytest.skip when
the file is absent, and its entry is wrapped in pytest.param(..., marks=pytest.mark.rom).
"""

import hashlib
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from algoarcade import rom
from algoarcade.backends import fake
from algoarcade.backends.base import Button, GameBackend
from algoarcade.backends.mgba import EWRAM_BASE, EWRAM_SIZE, MgbaBackend
from algoarcade.backends.pyboy import ADDRESS_SPACE, PyBoyBackend


def make_mgba() -> GameBackend:
    path = Path(__file__).resolve().parents[2] / "data" / "roms" / "firered.gba"
    if not path.is_file():
        pytest.skip("Place your FireRed ROM in data/roms/firered.gba")
    pytest.importorskip(
        "mgba.core", reason="mGBA binding did not load; see docs/setup.md", exc_type=ImportError
    )
    return MgbaBackend(path)


def make_pyboy() -> GameBackend:
    path = Path(__file__).resolve().parents[2] / "data" / "roms" / "super_mario_land.gb"
    if not path.is_file():
        pytest.skip("Place your Super Mario Land ROM in data/roms/super_mario_land.gb")
    backend = PyBoyBackend(path)
    backend.step(120)  # WRAM stays zero until the game initialises it, about frame 96
    return backend


# A 15-byte Game Boy program of our own, with no Nintendo code, logo or data, so CI can run
# PyBoyBackend without a ROM. It loops forever: select the d-pad, count at 0xC000, copy the
# d-pad to 0xC001.
PROBE_CODE = bytes.fromhex("3e20e0002100c034f000ea01c018f1")


def make_pyboy_probe() -> GameBackend:
    image = bytearray(0x8000)
    image[0x100:0x104] = bytes.fromhex("00c35001")  # nop; jp 0x150
    image[0x150 : 0x150 + len(PROBE_CODE)] = PROBE_CODE
    image[0x14D] = (-sum(image[0x134:0x14D]) - 25) & 0xFF  # header checksum; PyBoy checks it
    sha1 = hashlib.sha1(image).hexdigest()
    with tempfile.TemporaryDirectory() as folder, pytest.MonkeyPatch.context() as patch:
        path = Path(folder) / "probe.gb"
        path.write_bytes(image)
        # PyBoyBackend runs only Super Mario Land, so the probe borrows that game name.
        patch.setitem(rom.KNOWN_ROMS, sha1, rom.Rom(sha1, "super_mario_land", "probe"))
        backend = PyBoyBackend(path)
    backend.step(120)  # PyBoy's boot ROM runs first; the probe starts at frame 65
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
    "mgba": BackendCase(
        make=make_mgba,
        ram=(EWRAM_BASE, EWRAM_SIZE),
        unmapped=EWRAM_BASE + EWRAM_SIZE,
    ),
    "pyboy": BackendCase(make=make_pyboy, ram=(0xC000, 0x2000), unmapped=ADDRESS_SPACE),
    "pyboy_probe": BackendCase(make=make_pyboy_probe, ram=(0xC000, 0x2000), unmapped=ADDRESS_SPACE),
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
        pytest.param(value, id=name, marks=pytest.mark.rom if name in ("mgba", "pyboy") else ())
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
