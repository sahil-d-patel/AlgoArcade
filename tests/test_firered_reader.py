import struct
from dataclasses import replace
from itertools import permutations
from pathlib import Path

import pytest

from algoarcade.backends.fake import FakeBackend
from algoarcade.backends.mgba import MgbaBackend
from algoarcade.games.base import StateNotReady
from algoarcade.games.firered.addresses import ADDRESSES, IDS
from algoarcade.games.firered.reader import (
    STORY_FLAGS,
    BattlePokemon,
    FireRedReader,
    PartyPokemon,
)
from algoarcade.rewards.penalties import AntiSpam
from algoarcade.rom import identify

DATA = Path(__file__).resolve().parents[1] / "data"
ROM = DATA / "roms" / "firered.gba"
STATES_ROM_SHA1 = "dd5945db9b930750cb39d00c84da8571feebf417"  # rev 1, as captured by ENV-06

# Location and (level, hp, max_hp, status) per party slot, written down from the screen of each
# state. Map ids are (group, number) in data/maps/map_groups.json of pret/pokefirered at
# e93eb12a336fb97df420b0961753d8a3713c9fcc, and x and y share the frame of the event coordinates
# in each map's map.json there.
SCREEN = {
    # Player's bedroom, MAP_PALLET_TOWN_PLAYERS_HOUSE_2F. The player faces up, and A prints "RED
    # played with the NES.": the bg event at (6, 5). The START menu has no POKéMON entry.
    "s0": ((4, 1, 6, 6), ()),
    # Oak's lab, MAP_PALLET_TOWN_PROFESSOR_OAKS_LAB, after the starter and the parcel. The last
    # item ball, (10, 4) in map.json, sits four tiles right on the player's row, with Oak at
    # (6, 3) directly above. Party screen: SQUIRTLE Lv7, HP 14/ 25, no status icon.
    "s1": ((4, 3, 6, 4), ((7, 14, 25, 0),)),
    # MAP_VIRIDIAN_FOREST, on the middle tile of the south entrance warps (28 to 30, 62). The sign
    # at (31, 60) is two tiles right and two up. Party screen: SQUIRTLE Lv13, HP 37/ 37, no status
    # icon.
    "seg_05_forest": ((1, 0, 29, 62), ((13, 37, 37, 0),)),
}


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
        self.write(addresses["sTextPrinters"].address, bytes(25 * 0x24))
        self.write(addresses["sGlobalScriptContext"].address, bytes(8))
        self.write(addresses["gBattleMons"].address, bytes(2 * 0x58))
        self.write(addresses["gBattlerPartyIndexes"].address, bytes(8))
        self.write(addresses["gEnemyParty"].address, bytes(600))
        self.write(addresses["gPlayerParty"].address, bytes(600))

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
    backend.write(
        addresses["gSaveBlock2Ptr"].address,
        (addresses["gSaveBlock2"].address + offset).to_bytes(4, "little"),
    )
    backend.write(base + offset, struct.pack("<hhbb", 27, 31, 5, 6))
    backend.write(base + offset + 0xFE4, b"\x03")
    moved = reader.read(backend)
    assert moved.location == (5, 6, 27, 31)
    assert moved.badge_flags == 3
    assert first.location == (3, 4, 12, 19)
    assert first.badge_flags == 0


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
                level=level,
                hp=hp,
                max_hp=max_hp,
                status=status,
                species=0,
                moves=(0, 0, 0, 0),
                is_egg=False,
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
@pytest.mark.parametrize(("offset1", "offset2"), [(0, 4), (4, 0), (0x40, 0x7C)])
def test_save_blocks_at_different_offsets_are_not_ready(
    sha1: str, offset1: int, offset2: int
) -> None:
    addresses = ADDRESSES[sha1]
    backend = MemoryBackend()
    block1 = addresses["gSaveBlock1"].address + offset1
    block2 = addresses["gSaveBlock2"].address + offset2
    backend.write(addresses["gSaveBlock1Ptr"].address, block1.to_bytes(4, "little"))
    backend.write(addresses["gSaveBlock2Ptr"].address, block2.to_bytes(4, "little"))
    with pytest.raises(StateNotReady, match="different offsets"):
        FireRedReader(sha1).read(backend)


def party_entry(personality: int, species: int, level: int, hp: int, *, egg: bool = False) -> bytes:
    """A gPlayerParty entry with growth, attacks, EVs and misc in the order personality % 24 picks."""
    growth = struct.pack("<HHIBBH", species, 13, 125, 0, 70, 0)
    attacks = struct.pack("<4H4B", 33, 39, 55, 0, 35, 30, 25, 0)
    misc = struct.pack("<BBHII", 0, 0, 0, (1 << 30) * egg | 0x0B0B0B0B, 0)
    blocks = (growth, attacks, bytes([5] * 12), misc)
    plain = b"".join(blocks[index] for index in list(permutations(range(4)))[personality % 24])
    entry = bytearray(100)
    trainer_id = 0xFEDCBA98
    struct.pack_into("<II", entry, 0, personality, trainer_id)
    struct.pack_into("<H", entry, 0x1C, sum(struct.unpack("<24H", plain)) & 0xFFFF)
    key = personality ^ trainer_id
    struct.pack_into("<12I", entry, 0x20, *(word ^ key for word in struct.unpack("<12I", plain)))
    struct.pack_into("<IBxHH", entry, 0x50, 0, level, hp, 20)
    return bytes(entry)


@pytest.mark.parametrize("sha1", ADDRESSES)
@pytest.mark.parametrize("personality", range(24))
def test_party_decryption_and_checksum(sha1: str, personality: int) -> None:
    addresses = ADDRESSES[sha1]
    backend = MemoryBackend()
    base = addresses["gSaveBlock1"].address
    backend.write(addresses["gSaveBlock1Ptr"].address, base.to_bytes(4, "little"))
    backend.write(base, struct.pack("<hhbb", 12, 19, 3, 4))
    backend.write(addresses["gPlayerPartyCount"].address, b"\x01")
    entry = bytearray(party_entry(personality, 1 + personality, 5, 19))
    address = addresses["gPlayerParty"].address
    backend.write(address, bytes(entry))
    reader = FireRedReader(sha1)
    pokemon = reader.read(backend).party[0]
    assert pokemon.species == 1 + personality
    assert pokemon.moves == (33, 39, 55, 0)
    assert pokemon.hp == 19
    assert not pokemon.is_egg
    entry[0x25] ^= 1
    backend.write(address, bytes(entry))
    with pytest.raises(StateNotReady, match="checksum in slot 0"):
        reader.read(backend)


@pytest.mark.parametrize("sha1", ADDRESSES)
@pytest.mark.parametrize("personality", range(24))
def test_eggs_are_left_out_of_party_totals(sha1: str, personality: int) -> None:
    addresses = ADDRESSES[sha1]
    backend = MemoryBackend()
    base = addresses["gSaveBlock1"].address
    backend.write(addresses["gSaveBlock1Ptr"].address, base.to_bytes(4, "little"))
    backend.write(base, struct.pack("<hhbb", 12, 19, 3, 4))
    backend.write(addresses["gPlayerPartyCount"].address, b"\x02")
    party = addresses["gPlayerParty"].address
    backend.write(party, party_entry(personality, 7, 5, 10))
    backend.write(party + 100, party_entry(personality, 1, 5, 20, egg=True))
    state = FireRedReader(sha1).read(backend)
    assert [mon.is_egg for mon in state.party] == [False, True]
    assert state.party_count == 2
    assert state.party_level_sum == 5
    assert state.party_hp_frac == 0.5


@pytest.mark.parametrize("sha1", ADDRESSES)
def test_progress_context_and_pokedex_follow_current_memory(sha1: str) -> None:
    addresses = ADDRESSES[sha1]
    backend = MemoryBackend()
    base1 = addresses["gSaveBlock1"].address
    base2 = addresses["gSaveBlock2"].address
    backend.write(addresses["gSaveBlock1Ptr"].address, base1.to_bytes(4, "little"))
    backend.write(addresses["gPlayerPartyCount"].address, b"\x00")
    backend.write(base1, struct.pack("<hhbb", 12, 19, 3, 4))
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
    backend.write(base, struct.pack("<hhbb", 12, 19, 3, 4))
    backend.write(base + 0xEE0, bytes([0xFF] * (0x230 // 8)))
    state = FireRedReader(sha1).read(backend)
    assert state.badge_count == 0
    assert state.trainers_defeated == 0
    assert state.story_flags == state.world_map_flags == 0


@pytest.mark.parametrize("sha1", ADDRESSES)
def test_story_whitelist_isolated_from_other_flags(sha1: str) -> None:
    assert min(STORY_FLAGS.values()) >= 0x230
    addresses = ADDRESSES[sha1]
    backend = MemoryBackend()
    base = addresses["gSaveBlock1"].address
    backend.write(addresses["gSaveBlock1Ptr"].address, base.to_bytes(4, "little"))
    backend.write(addresses["gPlayerPartyCount"].address, b"\x00")
    backend.write(base, struct.pack("<hhbb", 12, 19, 3, 4))
    reader = FireRedReader(sha1)
    # The Pallet-to-Pewter list approved in docs/decisions.md, 2026-11-03.
    approved = [0x2CF, 0x828, 0x258, 0x230, 0x829, 0x4B0, 0x254]
    for count, flag in enumerate(approved, start=1):
        address = base + 0xEE0 + flag // 8
        backend.write(address, bytes([backend.read(address, 1)[0] | (1 << (flag % 8))]))
        assert reader.read(backend).story_flags == count
    assert reader.read(backend).story_flags == len(STORY_FLAGS)
    backend.write(base + 0xEE0, bytes([0xFF] * (0x230 // 8)))
    assert reader.read(backend).story_flags == 7
    backend.write(base + 0xEE0 + 0x231 // 8, b"\xff")
    assert reader.read(backend).story_flags == 7


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
    backend.write(base1, struct.pack("<hhbb", 12, 19, 3, 4))
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
    backend.write(addresses["gSaveBlock1"].address, struct.pack("<hhbb", 12, 19, 3, 4))
    state = FireRedReader(next(iter(ADDRESSES))).read(backend)
    party = (
        PartyPokemon(
            level=5, hp=10, max_hp=20, status=0, species=7, moves=(33, 39, 0, 0), is_egg=False
        ),
        PartyPokemon(
            level=5, hp=0, max_hp=80, status=0, species=1, moves=(33, 45, 0, 0), is_egg=False
        ),
    )
    assert replace(state, party=party).party_hp_frac == 0.1


@pytest.mark.parametrize("sha1", ADDRESSES)
def test_text_progress_reads_printer_zero_and_drives_anti_spam(sha1: str) -> None:
    addresses = ADDRESSES[sha1]
    backend = MemoryBackend()
    backend.write(
        addresses["gSaveBlock1Ptr"].address, addresses["gSaveBlock1"].address.to_bytes(4, "little")
    )
    backend.write(addresses["gPlayerPartyCount"].address, b"\x00")
    backend.write(addresses["gSaveBlock1"].address, struct.pack("<hhbb", 12, 19, 3, 4))
    printer = addresses["sTextPrinters"].address
    backend.write(printer + 0x1B, b"\x01")
    reader = FireRedReader(sha1)
    backend.write(printer, (0x02021D45).to_bytes(4, "little"))
    first = reader.read(backend)
    assert first.text_progress == 0x02021D45
    backend.write(printer + 0x24, (0x02021D60).to_bytes(4, "little"))
    same = reader.read(backend)
    backend.write(printer, (0x02021D60).to_bytes(4, "little"))
    advanced = reader.read(backend)
    assert advanced.text_progress == 0x02021D60
    anti_spam = AntiSpam()
    anti_spam.reset(first)
    assert anti_spam.score(first, same) == 1.0
    assert anti_spam.score(first, advanced) == 0.0


def overworld(sha1: str) -> tuple[MemoryBackend, int]:
    addresses = ADDRESSES[sha1]
    backend = MemoryBackend()
    base = addresses["gSaveBlock1"].address
    backend.write(addresses["gSaveBlock1Ptr"].address, base.to_bytes(4, "little"))
    backend.write(addresses["gPlayerPartyCount"].address, b"\x00")
    backend.write(base, struct.pack("<hhbb", 12, 19, 3, 4))
    return backend, base


def battler(species: int, hp: int, level: int, max_hp: int, personality: int) -> bytes:
    entry = bytearray(0x58)
    struct.pack_into("<H", entry, 0, species)
    struct.pack_into("<4H", entry, 0x0C, 33, 45, 0, 0)
    entry[0x24:0x28] = bytes([35, 40, 0, 0])
    struct.pack_into("<HBxH", entry, 0x28, hp, level, max_hp)
    struct.pack_into("<I", entry, 0x48, personality)
    return bytes(entry)


def in_battle(sha1: str) -> MemoryBackend:
    """Squirtle 14/ 20 in party slot 0 against Bulbasaur 13/ 19 in enemy slot 1."""
    addresses = ADDRESSES[sha1]
    backend, _ = overworld(sha1)
    mons = addresses["gBattleMons"].address
    backend.write(mons, battler(7, 14, 5, 20, 0x1111) + battler(1, 13, 5, 19, 0x2222))
    backend.write(addresses["gBattlerPartyIndexes"].address, struct.pack("<4H", 0, 1, 0, 0))
    backend.write(addresses["gPlayerParty"].address, party_entry(0x1111, 7, 5, 14) + bytes(500))
    backend.write(addresses["gEnemyParty"].address + 100, party_entry(0x2222, 1, 5, 13))
    backend.write(addresses["gMain"].address + 0x439, b"\x02")
    return backend


@pytest.mark.parametrize("sha1", ADDRESSES)
def test_battle_fields_read_both_battlers(sha1: str) -> None:
    state = FireRedReader(sha1).read(in_battle(sha1))
    assert state.battle_own == BattlePokemon(14, 20, 5, (33, 45, 0, 0), (35, 40, 0, 0))
    assert state.battle_enemy == BattlePokemon(13, 19, 5, (33, 45, 0, 0), (35, 40, 0, 0))


# Each write leaves gBattleMons behind the party copies, as the game does for some frames.
STALE = {
    # gBattleMons outlives the battle.
    "battle over": ("gMain", 0x439, b"\x00"),
    # A switch: gBattlerPartyIndexes moves before the new Pokemon is copied in.
    "player switch": ("gBattlerPartyIndexes", 0, struct.pack("<H", 1)),
    "opponent switch": ("gBattlerPartyIndexes", 2, struct.pack("<H", 2)),
    # A new opponent, before the battle intro copies it.
    "new opponent": ("gEnemyParty", 100, (0x3333).to_bytes(4, "little")),
    # A rematch after a loss: same personalities, the party healed.
    "rematch": ("gPlayerParty", 0x56, struct.pack("<H", 20)),
    # A level-up reaches the party first.
    "level up": ("gPlayerParty", 0x54, b"\x06"),
    "bad slot": ("gBattlerPartyIndexes", 2, struct.pack("<H", 0xFFFF)),
}


@pytest.mark.parametrize("sha1", ADDRESSES)
@pytest.mark.parametrize("case", STALE)
def test_battle_fields_are_none_while_gbattlemons_is_stale(sha1: str, case: str) -> None:
    backend = in_battle(sha1)
    symbol, offset, data = STALE[case]
    backend.write(ADDRESSES[sha1][symbol].address + offset, data)
    state = FireRedReader(sha1).read(backend)
    assert (state.battle_own, state.battle_enemy) == (None, None)


@pytest.mark.parametrize("sha1", ADDRESSES)
@pytest.mark.parametrize("offset", [0, 0x7C])
def test_event_flags_and_parcel_follow_the_save_block(sha1: str, offset: int) -> None:
    addresses = ADDRESSES[sha1]
    backend, base = overworld(sha1)
    pointer = base + offset
    backend.write(addresses["gSaveBlock1Ptr"].address, pointer.to_bytes(4, "little"))
    backend.write(
        addresses["gSaveBlock2Ptr"].address,
        (addresses["gSaveBlock2"].address + offset).to_bytes(4, "little"),
    )
    backend.write(pointer, struct.pack("<hhbb", 12, 19, 3, 4))
    reader = FireRedReader(sha1)
    first = reader.read(backend)
    assert first.event_flags == frozenset()
    assert not first.parcel_delivered
    for name in ("FLAG_SYS_POKEMON_GET", "FLAG_WORLD_MAP_PEWTER_CITY", "FLAG_BADGE01_GET"):
        flag = IDS[name].value
        address = pointer + 0xEE0 + flag // 8
        backend.write(address, bytes([backend.read(address, 1)[0] | 1 << (flag % 8)]))
    # A neighbour of FLAG_WORLD_MAP_PEWTER_CITY that is not a milestone.
    neighbour = pointer + 0xEE0 + 0x893 // 8
    backend.write(neighbour, bytes([backend.read(neighbour, 1)[0] | 1 << (0x893 % 8)]))
    assert reader.read(backend).event_flags == {
        "FLAG_SYS_POKEMON_GET",
        "FLAG_WORLD_MAP_PEWTER_CITY",
        "FLAG_BADGE01_GET",
    }
    mart = pointer + 0x1000 + 2 * (IDS["VAR_MAP_SCENE_VIRIDIAN_CITY_MART"].value - 0x4000)
    parcel = IDS["ITEM_OAKS_PARCEL"].value
    backend.write(mart, struct.pack("<H", 1))
    backend.write(pointer + 0x3B8 + 4 * 2, struct.pack("<HH", parcel, 1))
    assert not reader.read(backend).parcel_delivered
    backend.write(pointer + 0x3B8 + 4 * 2, bytes(4))
    assert reader.read(backend).parcel_delivered
    # Stored in the PC is not delivered.
    backend.write(pointer + 0x298 + 4 * 29, struct.pack("<HH", parcel, 1))
    assert not reader.read(backend).parcel_delivered
    backend.write(pointer + 0x298 + 4 * 29, bytes(4))
    backend.write(mart, struct.pack("<H", 0))
    assert not reader.read(backend).parcel_delivered


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
        state = FireRedReader(STATES_ROM_SHA1).read(backend)
        party = tuple((mon.level, mon.hp, mon.max_hp, mon.status) for mon in state.party)
        assert (state.location, party) == SCREEN[name]
    finally:
        backend.close()


@pytest.mark.parametrize("sha1", ADDRESSES)
def test_text_box_open_covers_printing_waiting_and_battle_windows(sha1: str) -> None:
    addresses = ADDRESSES[sha1]
    backend = MemoryBackend()
    base = addresses["gSaveBlock1"].address
    backend.write(addresses["gSaveBlock1Ptr"].address, base.to_bytes(4, "little"))
    backend.write(addresses["gPlayerPartyCount"].address, b"\x00")
    backend.write(base, struct.pack("<hhbb", 12, 19, 3, 4))
    printers = addresses["sTextPrinters"].address
    script = addresses["sGlobalScriptContext"].address
    wait = addresses["WaitForAorBPress"].address | 1
    backend.write(printers, (0x02021D45).to_bytes(4, "little"))
    backend.write(printers + 24 * 0x24, (0x02021000).to_bytes(4, "little"))
    backend.write(script, bytes(8))
    reader = FireRedReader(sha1)
    closed = reader.read(backend)
    assert not closed.text_box_open
    assert closed.text_progress == 0x02021D45

    backend.write(printers + 0x1B, b"\x01")
    assert reader.read(backend).text_box_open
    backend.write(printers + 0x1B, b"\x00")
    backend.write(printers + 24 * 0x24 + 0x1B, b"\x01")
    oak = reader.read(backend)
    assert oak.text_box_open
    assert oak.text_progress == 0x02021000
    backend.write(printers + 24 * 0x24 + 0x1B, b"\x00")

    backend.write(script, struct.pack("<BBxxI", 0, 2, wait))
    waiting = reader.read(backend)
    assert waiting.text_box_open
    assert not waiting.text_active
    anti_spam = AntiSpam()
    anti_spam.reset(waiting)
    assert anti_spam.score(waiting, reader.read(backend)) == 1.0
    backend.write(script, struct.pack("<BBxxI", 0, 0, wait))
    assert not reader.read(backend).text_box_open
    backend.write(script, struct.pack("<BBxxI", 0, 2, wait + 0x10))
    assert not reader.read(backend).text_box_open
