from pathlib import Path

import pytest

from algoarcade.games.firered.addresses import ADDRESSES, SOURCE_COMMIT
from algoarcade.rom import KNOWN_ROMS


def parse_symbol_line(line: str) -> tuple[str, int]:
    address, _scope, _size, name = line.split()
    base, separator, suffix = name.rpartition(".")
    if separator and suffix.isdecimal():
        name = base
    return name, int(address, 16)


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("02024029 g 00000001 gPlayerPartyCount", ("gPlayerPartyCount", 0x02024029)),
        ("03005008 g 00000004 gSaveBlock1Ptr", ("gSaveBlock1Ptr", 0x03005008)),
        ("02000010 l 00000001 sExample.182", ("sExample", 0x02000010)),
    ],
)
def test_parse_symbol_line(line: str, expected: tuple[str, int]) -> None:
    assert parse_symbol_line(line) == expected


def test_table_covers_supported_firered_roms() -> None:
    assert set(ADDRESSES) == {sha1 for sha1, rom in KNOWN_ROMS.items() if rom.game == "firered"}
    for entries in ADDRESSES.values():
        assert len(entries) == 10
        assert all(entry.source_commit == SOURCE_COMMIT for entry in entries.values())


@pytest.mark.parametrize(
    ("sha1", "filename"),
    [
        ("41cb23d8dccc8ebd7c649cd8fbb58eeace6e2fdc", "pokefirered.sym"),
        ("dd5945db9b930750cb39d00c84da8571feebf417", "pokefirered_rev1.sym"),
    ],
)
def test_addresses_match_downloaded_symbols(sha1: str, filename: str) -> None:
    path = Path(__file__).resolve().parents[1] / "data" / "pokefirered" / filename
    if not path.exists():
        pytest.skip("Run uv run python scripts/fetch_symbols.py to fetch symbol files")
    entries = ADDRESSES[sha1]
    found: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        name, address = parse_symbol_line(line)
        if name in entries:
            assert name not in found, f"Ambiguous symbol: {name}"
            found[name] = address
    assert found == {name: entry.address for name, entry in entries.items()}
