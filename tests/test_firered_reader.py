import struct

import pytest

from algoarcade.backends.fake import FakeBackend
from algoarcade.games.base import StateNotReady
from algoarcade.games.firered.addresses import ADDRESSES
from algoarcade.games.firered.reader import FireRedReader, PartyPokemon


class MemoryBackend(FakeBackend):
    def __init__(self) -> None:
        super().__init__()
        self.memory: dict[int, int] = {}

    def read(self, address: int, size: int) -> bytes:
        return bytes(self.memory[address + offset] for offset in range(size))

    def write(self, address: int, data: bytes) -> None:
        self.memory.update((address + offset, value) for offset, value in enumerate(data))


@pytest.mark.parametrize("sha1", ADDRESSES)
@pytest.mark.parametrize("offset", range(0, 0x80, 4))
def test_position_follows_moving_pointer(sha1: str, offset: int) -> None:
    addresses = ADDRESSES[sha1]
    backend = MemoryBackend()
    reader = FireRedReader(sha1)
    base = addresses["gSaveBlock1"].address
    pointer_address = addresses["gSaveBlock1Ptr"].address
    backend.write(pointer_address, base.to_bytes(4, "little"))
    backend.write(base, struct.pack("<hhbb", 12, 19, 3, 4))
    backend.write(addresses["gPlayerPartyCount"].address, b"\x00")
    first = reader.read(backend)
    assert first.location == (3, 4, 12, 19)
    assert first.party == ()
    assert first.party_count == 0

    backend.write(pointer_address, (base + offset).to_bytes(4, "little"))
    backend.write(base + offset, struct.pack("<hhbb", 27, 31, 5, 6))
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
@pytest.mark.parametrize("count", range(1, 7))
def test_party_uses_live_entries_and_refreshes_each_read(sha1: str, count: int) -> None:
    addresses = ADDRESSES[sha1]
    backend = MemoryBackend()
    base = addresses["gSaveBlock1"].address
    backend.write(addresses["gSaveBlock1Ptr"].address, base.to_bytes(4, "little"))
    backend.write(base, struct.pack("<hhbb", 12, 19, 3, 4))
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
    backend.write(base, struct.pack("<hhbb", 12, 19, 3, 4))
    backend.write(addresses["gPlayerPartyCount"].address, bytes([count]))
    with pytest.raises(StateNotReady, match="gPlayerPartyCount"):
        FireRedReader(sha1).read(backend)
