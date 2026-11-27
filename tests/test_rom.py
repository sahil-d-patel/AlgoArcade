import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from algoarcade import rom
from algoarcade.backends.pyboy import PyBoyBackend

FIRERED_REV0 = "41cb23d8dccc8ebd7c649cd8fbb58eeace6e2fdc"
FIRERED_REV1 = "dd5945db9b930750cb39d00c84da8571feebf417"
SUPER_MARIO_LAND = "418203621b887caa090215d97e3f509b79affd3e"


@pytest.fixture
def dump(tmp_path: Path) -> Path:
    path = tmp_path / "dump.gba"
    path.write_bytes(b"not a real ROM")
    return path


def test_table_holds_only_the_verified_hashes() -> None:
    assert set(rom.KNOWN_ROMS) == {FIRERED_REV0, FIRERED_REV1, SUPER_MARIO_LAND}
    assert all(sha1 == entry.sha1 for sha1, entry in rom.KNOWN_ROMS.items())


def test_unknown_rom_error_names_expected_and_actual_hashes(dump: Path) -> None:
    actual = hashlib.sha1(dump.read_bytes()).hexdigest()
    with pytest.raises(rom.UnknownRomError) as raised:
        rom.identify(dump)
    message = str(raised.value)
    assert actual in message
    assert FIRERED_REV0 in message
    assert FIRERED_REV1 in message


def test_known_rom_is_identified(dump: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sha1 = hashlib.sha1(dump.read_bytes()).hexdigest()
    entry = rom.Rom(sha1, "toy", "Toy dump")
    monkeypatch.setitem(rom.KNOWN_ROMS, sha1, entry)
    assert rom.identify(dump) == entry


def test_main_prints_the_identification(
    dump: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    sha1 = hashlib.sha1(dump.read_bytes()).hexdigest()
    monkeypatch.setitem(rom.KNOWN_ROMS, sha1, rom.Rom(sha1, "toy", "Toy dump"))
    rom.main([str(dump)])
    assert capsys.readouterr().out == f"{sha1}  Toy dump\n"


@pytest.mark.parametrize("name", ["dump.gba", "missing.gba"])
def test_module_entry_point_exits_non_zero_on_a_bad_path(dump: Path, name: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "algoarcade.rom", str(dump.parent / name)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert result.stdout == ""
    assert name in result.stderr


@pytest.mark.parametrize("game", [None, "firered"])
def test_pyboy_backend_refuses_any_rom_but_super_mario_land(
    dump: Path, monkeypatch: pytest.MonkeyPatch, game: str | None
) -> None:
    if game:
        sha1 = hashlib.sha1(dump.read_bytes()).hexdigest()
        monkeypatch.setitem(rom.KNOWN_ROMS, sha1, rom.Rom(sha1, game, "Another known game"))
    with pytest.raises(rom.UnknownRomError):
        PyBoyBackend(dump)
