import struct
from dataclasses import replace
from itertools import permutations

import pytest

from algoarcade.backends.fake import FakeBackend
from algoarcade.games.base import StateNotReady
from algoarcade.games.firered.addresses import ADDRESSES
from algoarcade.games.firered.reader import STORY_FLAGS, FireRedReader, PartyPokemon


class MemoryBackend(FakeBackend):
    def __init__(self) -> None:
        super().__init__()
        self.memory: dict[int, int] = {}
        addresses = next(iter(ADDRESSES.values()))
        self.write(addresses["gSaveBlock1"].address, bytes(0x3D68 + 0x7C))
        self.write(addresses["gSaveBlock2"].address, bytes(0xF24 + 0x7C))
        self.write(
            addresses["gSaveBlock2Ptr"].address,
            addresses["gSaveBlock2"].address.to_bytes(4, "little"),
        )
        self.write(addresses["gMain"].address + 0x439, b"\x00")
        self.write(addresses["sMessageBoxType"].address, b"\x00")
        self.write(addresses["sGlobalScriptContextStatus"].address, b"\x02")
        self.write(addresses["sLockFieldControls"].address, b"\x00")

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
    assert first.party_hp_frac == 0

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
        expected.append(
            PartyPokemon(
                level=level, hp=hp, max_hp=max_hp, status=status, species=0, moves=(0, 0, 0, 0)
            )
        )

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


@pytest.mark.parametrize("sha1", ADDRESSES)
@pytest.mark.parametrize("offset", range(0, 0x80, 4))
def test_money_follows_both_pointers_and_changed_key(sha1: str, offset: int) -> None:
    addresses = ADDRESSES[sha1]
    backend = MemoryBackend()
    reader = FireRedReader(sha1)
    backend.write(addresses["gPlayerPartyCount"].address, b"\x00")
    for shift, money, key in [(0, 3000, 0x12345678), (offset, 999999, 0xFEDCBA98)]:
        block1 = addresses["gSaveBlock1"].address + shift
        block2 = addresses["gSaveBlock2"].address + shift
        backend.write(addresses["gSaveBlock1Ptr"].address, block1.to_bytes(4, "little"))
        backend.write(addresses["gSaveBlock2Ptr"].address, block2.to_bytes(4, "little"))
        backend.write(block1, struct.pack("<hhbb", 12, 19, 3, 4))
        backend.write(block1 + 0x290, (money ^ key).to_bytes(4, "little"))
        backend.write(block2 + 0xF20, key.to_bytes(4, "little"))
        assert reader.read(backend).money == money


@pytest.mark.parametrize("sha1", ADDRESSES)
@pytest.mark.parametrize("offset", [None, -4, 0x80, 1, 0x7B, 0x100000])
def test_invalid_second_save_pointer_is_not_ready(sha1: str, offset: int | None) -> None:
    addresses = ADDRESSES[sha1]
    backend = MemoryBackend()
    backend.write(
        addresses["gSaveBlock1Ptr"].address, addresses["gSaveBlock1"].address.to_bytes(4, "little")
    )
    pointer = 0 if offset is None else addresses["gSaveBlock2"].address + offset
    backend.write(addresses["gSaveBlock2Ptr"].address, pointer.to_bytes(4, "little"))
    with pytest.raises(StateNotReady, match="gSaveBlock2Ptr"):
        FireRedReader(sha1).read(backend)


@pytest.mark.parametrize("sha1", ADDRESSES)
@pytest.mark.parametrize(("personality", "order"), list(enumerate(permutations(range(4)))))
def test_party_decryption_and_checksum(sha1: str, personality: int, order: tuple[int, ...]) -> None:
    addresses = ADDRESSES[sha1]
    backend = MemoryBackend()
    base = addresses["gSaveBlock1"].address
    backend.write(addresses["gSaveBlock1Ptr"].address, base.to_bytes(4, "little"))
    backend.write(base, struct.pack("<hhbb", 12, 19, 3, 4))
    backend.write(addresses["gPlayerPartyCount"].address, b"\x01")
    growth = struct.pack("<HHIBBH", 7, 13, 125, 0, 70, 0)
    attacks = struct.pack("<4H4B", 33, 39, 55, 0, 35, 30, 25, 0)
    blocks = (growth, attacks, bytes([5] * 12), bytes([11] * 12))
    plain = b"".join(blocks[index] for index in order)
    entry = bytearray(100)
    trainer_id = 0xFEDCBA98
    struct.pack_into("<II", entry, 0, personality, trainer_id)
    struct.pack_into("<H", entry, 0x1C, sum(struct.unpack("<24H", plain)) & 0xFFFF)
    key = personality ^ trainer_id
    struct.pack_into("<12I", entry, 0x20, *(word ^ key for word in struct.unpack("<12I", plain)))
    struct.pack_into("<IBxHH", entry, 0x50, 0, 5, 19, 20)
    address = addresses["gPlayerParty"].address
    backend.write(address, bytes(entry))
    reader = FireRedReader(sha1)
    pokemon = reader.read(backend).party[0]
    assert pokemon.species == 7
    assert pokemon.moves == (33, 39, 55, 0)
    assert pokemon.hp == 19
    entry[0x25] ^= 1
    backend.write(address, bytes(entry))
    with pytest.raises(StateNotReady, match="checksum in slot 0"):
        reader.read(backend)


@pytest.mark.parametrize("sha1", ADDRESSES)
def test_progress_context_and_pokedex_follow_current_memory(sha1: str) -> None:
    addresses = ADDRESSES[sha1]
    backend = MemoryBackend()
    base1 = addresses["gSaveBlock1"].address
    base2 = addresses["gSaveBlock2"].address
    backend.write(addresses["gSaveBlock1Ptr"].address, base1.to_bytes(4, "little"))
    backend.write(addresses["gPlayerPartyCount"].address, b"\x00")
    reader = FireRedReader(sha1)
    first = reader.read(backend)
    assert first.badge_count == first.trainers_defeated == 0
    assert first.pokedex_seen == first.pokedex_owned == 0
    assert not any([first.in_battle, first.text_active, first.script_active, first.controls_locked])

    backend.write(base1 + 0xFE4, b"\x85")
    backend.write(base1 + 0xF80, bytes([0xFF] * 96))
    for offset, block in [(0x5F8, base1), (0x3A18, base1), (0x5C, base2), (0x28, base2)]:
        backend.write(block + offset, ((1 << 392) - 1).to_bytes(49, "little"))
    backend.write(addresses["gMain"].address + 0x439, b"\x03")
    backend.write(addresses["sMessageBoxType"].address, b"\x02")
    backend.write(addresses["sGlobalScriptContextStatus"].address, b"\x00")
    backend.write(addresses["sLockFieldControls"].address, b"\x01")
    second = reader.read(backend)
    assert second.badge_flags == 0x85
    assert second.badge_count == 3
    assert second.trainers_defeated == 768
    assert second.pokedex_seen == second.pokedex_owned == 386
    assert all([second.in_battle, second.text_active, second.script_active, second.controls_locked])
    backend.write(base1 + 0x5F8, b"\x00")
    third = reader.read(backend)
    assert third.pokedex_seen == third.pokedex_owned == 378
    backend.write(base2 + 0x28, b"\x00\x00")
    assert reader.read(backend).pokedex_owned == 370
    backend.write(addresses["sGlobalScriptContextStatus"].address, b"\x01")
    assert not reader.read(backend).script_active


@pytest.mark.parametrize("sha1", ADDRESSES)
def test_temporary_and_hide_flags_do_not_count_as_progress(sha1: str) -> None:
    addresses = ADDRESSES[sha1]
    backend = MemoryBackend()
    base = addresses["gSaveBlock1"].address
    backend.write(addresses["gSaveBlock1Ptr"].address, base.to_bytes(4, "little"))
    backend.write(addresses["gPlayerPartyCount"].address, b"\x00")
    backend.write(base + 0xEE0, bytes([0xFF] * (0x230 // 8)))
    state = FireRedReader(sha1).read(backend)
    assert state.badge_count == 0
    assert state.trainers_defeated == 0
    assert state.story_flags == state.world_map_flags == 0


@pytest.mark.parametrize("sha1", ADDRESSES)
def test_story_whitelist_isolated_from_other_flags(sha1: str) -> None:
    assert STORY_FLAGS == {
        "FLAG_SYS_POKEMON_GET": 0x828,
        "FLAG_BEAT_RIVAL_IN_OAKS_LAB": 0x258,
        "FLAG_SYS_POKEDEX_GET": 0x829,
        "FLAG_GOT_POTION_ON_ROUTE_1": 0x230,
        "FLAG_DEFEATED_BROCK": 0x4B0,
        "FLAG_GOT_TM39_FROM_BROCK": 0x254,
    }
    assert min(STORY_FLAGS.values()) >= 0x230
    addresses = ADDRESSES[sha1]
    backend = MemoryBackend()
    base = addresses["gSaveBlock1"].address
    backend.write(addresses["gSaveBlock1Ptr"].address, base.to_bytes(4, "little"))
    backend.write(addresses["gPlayerPartyCount"].address, b"\x00")
    reader = FireRedReader(sha1)
    for count, flag in enumerate([0x828, 0x258, 0x829, 0x230, 0x4B0, 0x254], start=1):
        address = base + 0xEE0 + flag // 8
        backend.write(address, bytes([backend.read(address, 1)[0] | (1 << (flag % 8))]))
        assert reader.read(backend).story_flags == count
    backend.write(base + 0xEE0, bytes([0xFF] * (0x230 // 8)))
    assert reader.read(backend).story_flags == 6
    backend.write(base + 0xEE0 + 0x231 // 8, b"\xff")
    assert reader.read(backend).story_flags == 6


@pytest.mark.parametrize("sha1", ADDRESSES)
@pytest.mark.parametrize(
    ("speed", "scene_off"), [(0, False), (1, False), (2, False), (1, True), (2, True)]
)
def test_world_flags_and_options_follow_relocated_save_blocks(
    sha1: str, speed: int, scene_off: bool
) -> None:
    addresses = ADDRESSES[sha1]
    backend = MemoryBackend()
    base1 = addresses["gSaveBlock1"].address + 0x7C
    base2 = addresses["gSaveBlock2"].address + 0x7C
    backend.write(addresses["gSaveBlock1Ptr"].address, base1.to_bytes(4, "little"))
    backend.write(addresses["gSaveBlock2Ptr"].address, base2.to_bytes(4, "little"))
    backend.write(addresses["gPlayerPartyCount"].address, b"\x00")
    backend.write(base1 + 0xFF2, bytes([0xFF] * 7))
    backend.write(base2 + 0x14, bytes([0xF8 | speed, 0xFB | (4 if scene_off else 0)]))
    reader = FireRedReader(sha1)
    state = reader.read(backend)
    assert state.world_map_flags == 51
    assert state.text_speed == speed
    assert state.battle_scene_off == scene_off
    if speed == 2 and scene_off:
        reader.check_options(backend)
    else:
        with pytest.raises(ValueError, match="fast text speed and battle scene off"):
            reader.check_options(backend)
    backend.write(base1 + 0xFF2, b"\x01\x00\x00\x00\x00\x00\xfc")
    assert reader.read(backend).world_map_flags == 2


def test_party_hp_fraction_uses_total_hp() -> None:
    addresses = next(iter(ADDRESSES.values()))
    backend = MemoryBackend()
    backend.write(
        addresses["gSaveBlock1Ptr"].address, addresses["gSaveBlock1"].address.to_bytes(4, "little")
    )
    backend.write(addresses["gPlayerPartyCount"].address, b"\x00")
    state = FireRedReader(next(iter(ADDRESSES))).read(backend)
    party = (
        PartyPokemon(level=5, hp=10, max_hp=20, status=0, species=7, moves=(33, 39, 0, 0)),
        PartyPokemon(level=5, hp=0, max_hp=80, status=0, species=1, moves=(33, 45, 0, 0)),
    )
    assert replace(state, party=party).party_hp_frac == 0.1
