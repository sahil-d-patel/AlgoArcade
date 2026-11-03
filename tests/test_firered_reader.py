import struct
from pathlib import Path

import pytest

from algoarcade.backends.fake import FakeBackend
from algoarcade.backends.mgba import MgbaBackend
from algoarcade.games.base import StateNotReady
from algoarcade.games.firered.addresses import ADDRESSES
from algoarcade.games.firered.reader import FireRedReader, FireRedState, PartyPokemon
from algoarcade.rom import identify

DATA = Path(__file__).resolve().parents[1] / "data"
ROM = DATA / "roms" / "firered.gba"
STATES_ROM_SHA1 = "dd5945db9b930750cb39d00c84da8571feebf417"  # rev 1, as captured by ENV-06

# Written down from the screen of each state. Map ids are (group, number) in
# data/maps/map_groups.json of pret/pokefirered at e93eb12a336fb97df420b0961753d8a3713c9fcc, and x
# and y share the frame of the event coordinates in each map's map.json there.
SCREEN = {
    # Player's bedroom, MAP_PALLET_TOWN_PLAYERS_HOUSE_2F. The player faces up, and A prints "RED
    # played with the NES.": the bg event at (6, 5). The START menu has no POKéMON entry.
    "s0": FireRedState(map_group=4, map_num=1, x=6, y=6, party=(), badge_flags=0),
    # Oak's lab, MAP_PALLET_TOWN_PROFESSOR_OAKS_LAB, after the starter and the parcel. The last
    # item ball, (10, 4) in map.json, sits four tiles right on the player's row, with Oak at
    # (6, 3) directly above. Party screen: SQUIRTLE Lv7, HP 14/ 25, no status icon.
    "s1": FireRedState(
        map_group=4,
        map_num=3,
        x=6,
        y=4,
        party=(PartyPokemon(level=7, hp=14, max_hp=25, status=0),),
        badge_flags=0,
    ),
    # All three come before Brock, so badge_flags is 0. MAP_VIRIDIAN_FOREST, on the middle tile of the south entrance warps (28 to 30, 62). The sign
    # at (31, 60) is two tiles right and two up. Party screen: SQUIRTLE Lv13, HP 37/ 37, no status
    # icon.
    "seg_05_forest": FireRedState(
        map_group=1,
        map_num=0,
        x=29,
        y=62,
        party=(PartyPokemon(level=13, hp=37, max_hp=37, status=0),),
        badge_flags=0,
    ),
}


class MemoryBackend(FakeBackend):
    def __init__(self) -> None:
        super().__init__()
        self.memory: dict[int, int] = {}

    def read(self, address: int, size: int) -> bytes:
        return bytes(self.memory[address + offset] for offset in range(size))

    def write(self, address: int, data: bytes) -> None:
        self.memory.update((address + offset, value) for offset, value in enumerate(data))


def write_save_block1(
    backend: MemoryBackend, address: int, position: bytes, badge_flags: int = 0
) -> None:
    backend.write(address, position)
    backend.write(address + 0xFE4, bytes([badge_flags]))


@pytest.mark.parametrize("sha1", ADDRESSES)
@pytest.mark.parametrize("offset", range(0, 0x80, 4))
def test_position_follows_moving_pointer(sha1: str, offset: int) -> None:
    addresses = ADDRESSES[sha1]
    backend = MemoryBackend()
    reader = FireRedReader(sha1)
    base = addresses["gSaveBlock1"].address
    pointer_address = addresses["gSaveBlock1Ptr"].address
    backend.write(pointer_address, base.to_bytes(4, "little"))
    write_save_block1(backend, base, struct.pack("<hhbb", 12, 19, 3, 4))
    backend.write(addresses["gPlayerPartyCount"].address, b"\x00")
    first = reader.read(backend)
    assert first.location == (3, 4, 12, 19)
    assert first.party == ()
    assert first.party_count == 0

    backend.write(pointer_address, (base + offset).to_bytes(4, "little"))
    write_save_block1(backend, base + offset, struct.pack("<hhbb", 27, 31, 5, 6))
    assert reader.read(backend).location == (5, 6, 27, 31)
    assert first.location == (3, 4, 12, 19)


@pytest.mark.parametrize("sha1", ADDRESSES)
@pytest.mark.parametrize("offset", [None, -4, 0x80, 1, 0x7B, 0x100000])
def test_invalid_pointer_is_not_ready(sha1: str, offset: int | None) -> None:
    addresses = ADDRESSES[sha1]
    backend = MemoryBackend()
    pointer = 0 if offset is None else addresses["gSaveBlock1"].address + offset
    backend.write(addresses["gSaveBlock1Ptr"].address, pointer.to_bytes(4, "little"))
    with pytest.raises(StateNotReady, match="gSaveBlock1Ptr"):
        FireRedReader(sha1).read(backend)


@pytest.mark.parametrize("sha1", ADDRESSES)
def test_empty_save_block_is_not_ready(sha1: str) -> None:
    addresses = ADDRESSES[sha1]
    backend = MemoryBackend()
    base = addresses["gSaveBlock1"].address
    backend.write(addresses["gSaveBlock1Ptr"].address, base.to_bytes(4, "little"))
    backend.write(base, bytes(6))
    backend.write(addresses["gPlayerPartyCount"].address, bytes(1))
    with pytest.raises(StateNotReady, match="empty"):
        FireRedReader(sha1).read(backend)


@pytest.mark.parametrize("sha1", ADDRESSES)
@pytest.mark.parametrize("count", range(1, 7))
def test_party_uses_live_entries_and_refreshes_each_read(sha1: str, count: int) -> None:
    addresses = ADDRESSES[sha1]
    backend = MemoryBackend()
    base = addresses["gSaveBlock1"].address
    backend.write(addresses["gSaveBlock1Ptr"].address, base.to_bytes(4, "little"))
    write_save_block1(backend, base, struct.pack("<hhbb", 12, 19, 3, 4))
    backend.write(addresses["gPlayerPartyCount"].address, bytes([count]))
    expected = []
    for index in range(count):
        entry = bytearray(100)
        status, level, hp, max_hp = 0x12340000 + index, 5 + index, 256 + index, 512 + index
        struct.pack_into("<I", entry, 0x50, status)
        entry[0x54] = level
        entry[0x55] = 0xFF
        struct.pack_into("<HH", entry, 0x56, hp, max_hp)
        backend.write(addresses["gPlayerParty"].address + index * 100, bytes(entry))
        expected.append(PartyPokemon(level=level, hp=hp, max_hp=max_hp, status=status))

    reader = FireRedReader(sha1)
    first = reader.read(backend)
    assert first.party == tuple(expected)
    assert first.party_count == count

    backend.write(addresses["gPlayerParty"].address + 0x56, b"\x00\x00")
    assert reader.read(backend).party[0].hp == 0
    assert first.party[0].hp == 256
    backend.write(addresses["gPlayerPartyCount"].address, b"\x00")
    assert reader.read(backend).party == ()


@pytest.mark.parametrize("sha1", ADDRESSES)
@pytest.mark.parametrize("count", [7, 255])
def test_invalid_party_count_is_not_ready(sha1: str, count: int) -> None:
    addresses = ADDRESSES[sha1]
    backend = MemoryBackend()
    base = addresses["gSaveBlock1"].address
    backend.write(addresses["gSaveBlock1Ptr"].address, base.to_bytes(4, "little"))
    write_save_block1(backend, base, struct.pack("<hhbb", 12, 19, 3, 4))
    backend.write(addresses["gPlayerPartyCount"].address, bytes([count]))
    with pytest.raises(StateNotReady, match="gPlayerPartyCount"):
        FireRedReader(sha1).read(backend)


@pytest.mark.rom
@pytest.mark.parametrize("name", SCREEN)
def test_captured_state_matches_the_screen(name: str) -> None:
    if not ROM.is_file():
        pytest.skip("Place your FireRed ROM in data/roms/firered.gba")
    if identify(ROM).sha1 != STATES_ROM_SHA1:
        pytest.skip("The states were captured on rev 1")
    state = DATA / "states" / f"{name}.state"
    if not state.is_file():
        pytest.skip(f"Missing {state.name}; scripts/capture_states.py (ENV-06) regenerates it")
    pytest.importorskip(
        "mgba.core", reason="mGBA binding did not load; see docs/setup.md", exc_type=ImportError
    )
    backend = MgbaBackend(ROM, render=False)
    try:
        backend.load_state(state.read_bytes())
        assert FireRedReader(STATES_ROM_SHA1).read(backend) == SCREEN[name]
    finally:
        backend.close()


@pytest.mark.parametrize("sha1", ADDRESSES)
def test_badge_flags_follow_moving_pointer(sha1: str) -> None:
    addresses = ADDRESSES[sha1]
    backend = MemoryBackend()
    reader = FireRedReader(sha1)
    base = addresses["gSaveBlock1"].address
    pointer_address = addresses["gSaveBlock1Ptr"].address
    backend.write(pointer_address, base.to_bytes(4, "little"))
    backend.write(addresses["gPlayerPartyCount"].address, b"\x00")
    write_save_block1(backend, base, struct.pack("<hhbb", 6, 6, 4, 1), badge_flags=0x01)
    assert reader.read(backend).badge_flags == 0x01

    backend.write(pointer_address, (base + 0x40).to_bytes(4, "little"))
    write_save_block1(backend, base + 0x40, struct.pack("<hhbb", 6, 6, 4, 1), badge_flags=0x03)
    assert reader.read(backend).badge_flags == 0x03
