import hashlib
from collections.abc import Iterator
from pathlib import Path

import pytest

from algoarcade.backends.mgba import MgbaBackend
from algoarcade.games.firered.reader import FireRedReader, FireRedState
from algoarcade.rom import identify


@pytest.fixture
def trainer_card_state() -> Iterator[FireRedState]:
    root = Path(__file__).resolve().parents[1]
    rom = root / "data" / "roms" / "firered.gba"
    state = root / "data" / "states" / "env08-money-3000.state"
    if not rom.is_file() or not state.is_file():
        pytest.skip("Needs the local FireRed ROM and ENV-08 trainer-card capture")
    if identify(rom).sha1 != "41cb23d8dccc8ebd7c649cd8fbb58eeace6e2fdc":
        pytest.skip("This capture was recorded on FireRed USA rev 0")
    data = state.read_bytes()
    assert hashlib.sha256(data).hexdigest() == (
        "37ad9fefd6813f5f68362247ee276d6205dd2b4a0823403c3a69f41b325ae093"
    )
    backend = MgbaBackend(rom)
    try:
        backend.load_state(data)
        yield FireRedReader(backend.rom.sha1).read(backend)
    finally:
        backend.close()


@pytest.mark.rom
def test_firered_fields_money(trainer_card_state: FireRedState) -> None:
    # Trainer card captured November 2: MONEY ₽3000.
    assert trainer_card_state.money == 3000


@pytest.mark.rom
def test_firered_fields_badges(trainer_card_state: FireRedState) -> None:
    # The same trainer card shows all eight badge slots empty.
    assert trainer_card_state.badge_flags == 0
    assert trainer_card_state.badge_count == 0
