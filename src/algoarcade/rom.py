"""Known ROMs, keyed by SHA-1. Address tables are keyed by the same hash.

Check a file with: uv run python -m algoarcade.rom PATH
"""

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Rom:
    sha1: str
    game: str
    name: str


# Add a hash only after two members have confirmed the same dump.
KNOWN_ROMS: dict[str, Rom] = {
    rom.sha1: rom
    for rom in (
        Rom("41cb23d8dccc8ebd7c649cd8fbb58eeace6e2fdc", "firered", "Pokemon FireRed (USA) rev 0"),
        Rom("dd5945db9b930750cb39d00c84da8571feebf417", "firered", "Pokemon FireRed (USA) rev 1"),
        Rom(
            "418203621b887caa090215d97e3f509b79affd3e",
            "super_mario_land",
            "Super Mario Land (World) rev A",
        ),
    )
}


class UnknownRomError(ValueError):
    pass


def identify(path: Path) -> Rom:
    sha1 = hashlib.sha1(path.read_bytes()).hexdigest()
    if sha1 not in KNOWN_ROMS:
        expected = "\n".join(f"  {rom.sha1}  {rom.name}" for rom in KNOWN_ROMS.values())
        raise UnknownRomError(
            f"{path}: not a known ROM.\n  actual SHA-1: {sha1}\nexpected one of:\n{expected}"
        )
    return KNOWN_ROMS[sha1]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m algoarcade.rom", description="Identify a ROM.")
    parser.add_argument("path", type=Path)
    path = parser.parse_args(argv).path
    try:
        rom = identify(path)
    except (OSError, UnknownRomError) as error:
        raise SystemExit(str(error)) from error
    print(f"{rom.sha1}  {rom.name}")  # noqa: T201


if __name__ == "__main__":
    main()
