import struct

import pytest

from algoarcade.backends.fake import FakeBackend
from algoarcade.games.base import StateNotReady
from algoarcade.games.firered.addresses import ADDRESSES
from algoarcade.games.firered.reader import FireRedReader


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
    first = reader.read(backend)
    assert first.location == (3, 4, 12, 19)

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
