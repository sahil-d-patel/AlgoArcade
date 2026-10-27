import struct
from dataclasses import dataclass

from algoarcade.backends.base import GameBackend, read_u8, read_u32
from algoarcade.games.base import StateNotReady
from algoarcade.games.firered.addresses import ADDRESSES


@dataclass(frozen=True)
class PartyPokemon:
    level: int
    hp: int
    max_hp: int
    status: int


@dataclass(frozen=True)
class FireRedState:
    map_group: int
    map_num: int
    x: int
    y: int
    party: tuple[PartyPokemon, ...]

    @property
    def party_count(self) -> int:
        return len(self.party)

    @property
    def location(self) -> tuple[int, int, int, int]:
        return (self.map_group, self.map_num, self.x, self.y)


class FireRedReader:
    def __init__(self, rom_sha1: str) -> None:
        self._addresses = ADDRESSES[rom_sha1]

    def read(self, backend: GameBackend) -> FireRedState:
        pointer = read_u32(backend, self._addresses["gSaveBlock1Ptr"].address)
        base = self._addresses["gSaveBlock1"].address
        # SetSaveBlocksPointers: src/load_save.c at ce6406d8c2eaf46d543c8a2606de0c3cb1951982.
        if not (0 <= pointer - base <= 0x7C) or (pointer - base) % 4:
            raise StateNotReady(f"Invalid gSaveBlock1Ptr: {pointer:#010x}")
        # SaveBlock1.pos/location: include/global.h at the same source commit.
        x, y, map_group, map_num = struct.unpack("<hhbb", backend.read(pointer, 6))
        count = read_u8(backend, self._addresses["gPlayerPartyCount"].address)
        if count > 6:
            raise StateNotReady(f"Invalid gPlayerPartyCount: {count}")
        party = []
        for index in range(count):
            # struct Pokemon: include/pokemon.h at the same source commit; live party only.
            address = self._addresses["gPlayerParty"].address + index * 100
            status, level, hp, max_hp = struct.unpack("<IBxHH", backend.read(address + 0x50, 10))
            party.append(PartyPokemon(level=level, hp=hp, max_hp=max_hp, status=status))
        return FireRedState(map_group=map_group, map_num=map_num, x=x, y=y, party=tuple(party))
