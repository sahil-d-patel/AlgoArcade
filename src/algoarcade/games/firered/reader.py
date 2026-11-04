import struct
from dataclasses import dataclass
from itertools import permutations

from algoarcade.backends.base import GameBackend, read_u8, read_u32
from algoarcade.games.base import StateNotReady
from algoarcade.games.firered.addresses import ADDRESSES

# GetSubstruct in src/pokemon.c: physical order of growth, attacks, EVs and miscellaneous.
SUBSTRUCT_ORDERS = tuple(permutations(range(4)))
# include/constants/flags.h at source e93eb12a336fb97df420b0961753d8a3713c9fcc. The list defines
# the logged story_flags column: docs/decisions.md, 2026-11-03.
STORY_FLAGS = {
    "FLAG_VISITED_OAKS_LAB": 0x2CF,
    "FLAG_SYS_POKEMON_GET": 0x828,
    "FLAG_BEAT_RIVAL_IN_OAKS_LAB": 0x258,
    "FLAG_GOT_POTION_ON_ROUTE_1": 0x230,
    "FLAG_SYS_POKEDEX_GET": 0x829,
    "FLAG_DEFEATED_BROCK": 0x4B0,
    "FLAG_GOT_TM39_FROM_BROCK": 0x254,
}


@dataclass(frozen=True)
class PartyPokemon:
    level: int
    hp: int
    max_hp: int
    status: int
    species: int
    moves: tuple[int, int, int, int]


@dataclass(frozen=True)
class FireRedState:
    map_group: int
    map_num: int
    x: int
    y: int
    party: tuple[PartyPokemon, ...]
    money: int
    badge_flags: int
    trainers_defeated: int
    pokedex_owned: int
    pokedex_seen: int
    in_battle: bool
    text_active: bool
    text_progress: int
    script_active: bool
    controls_locked: bool
    story_flags: int
    world_map_flags: int
    text_speed: int
    battle_scene_off: bool

    @property
    def party_count(self) -> int:
        return len(self.party)

    @property
    def location(self) -> tuple[int, int, int, int]:
        return (self.map_group, self.map_num, self.x, self.y)

    @property
    def badge_count(self) -> int:
        return self.badge_flags.bit_count()

    @property
    def party_level_sum(self) -> int:
        return sum(pokemon.level for pokemon in self.party)

    @property
    def party_hp_frac(self) -> float:
        maximum = sum(pokemon.max_hp for pokemon in self.party)
        return sum(pokemon.hp for pokemon in self.party) / maximum if maximum else 0.0


class FireRedReader:
    def __init__(self, rom_sha1: str) -> None:
        self._addresses = ADDRESSES[rom_sha1]

    def read(self, backend: GameBackend) -> FireRedState:
        pointer = read_u32(backend, self._addresses["gSaveBlock1Ptr"].address)
        base = self._addresses["gSaveBlock1"].address
        # SetSaveBlocksPointers: src/load_save.c at e93eb12a336fb97df420b0961753d8a3713c9fcc.
        if not (0 <= pointer - base <= 0x7C) or (pointer - base) % 4:
            raise StateNotReady(f"Invalid gSaveBlock1Ptr: {pointer:#010x}")
        pointer2 = read_u32(backend, self._addresses["gSaveBlock2Ptr"].address)
        base2 = self._addresses["gSaveBlock2"].address
        if not (0 <= pointer2 - base2 <= 0x7C) or (pointer2 - base2) % 4:
            raise StateNotReady(f"Invalid gSaveBlock2Ptr: {pointer2:#010x}")
        if pointer - base != pointer2 - base2:
            raise StateNotReady("gSaveBlock1Ptr and gSaveBlock2Ptr are at different offsets")
        # SaveBlock1.money, SaveBlock2.encryptionKey and GetMoney at the same source commit.
        money = read_u32(backend, pointer + 0x290) ^ read_u32(backend, pointer2 + 0xF20)
        # SaveBlock1.flags is at 0xEE0 and FLAG_BADGE01_GET is 0x820 (include/constants/flags.h),
        # so the eight badges are the byte at 0xFE4, Boulder Badge in bit 0.
        badge_flags = read_u8(backend, pointer + 0xFE4)
        # flags.h trainer ids 0x500..0x7FF; FlagGet indexes SaveBlock1.flags at 0xEE0.
        trainers_defeated = sum(byte.bit_count() for byte in backend.read(pointer + 0xF80, 96))
        story_flags = sum(
            bool(read_u8(backend, pointer + 0xEE0 + flag // 8) & (1 << (flag % 8)))
            for flag in STORY_FLAGS.values()
        )
        # World-map ids 0x890..0x8C2 occupy 51 bits; exclude padding in the final byte.
        world_map_flags = int.from_bytes(backend.read(pointer + 0xFF2, 7), "little")
        world_map_flags &= (1 << 51) - 1
        options = backend.read(pointer2 + 0x14, 2)
        # DexScreen_GetSetPokedexFlag requires matching seen bits in all three copies.
        seen = int.from_bytes(backend.read(pointer2 + 0x5C, 49), "little")
        seen &= int.from_bytes(backend.read(pointer + 0x5F8, 49), "little")
        seen &= int.from_bytes(backend.read(pointer + 0x3A18, 49), "little")
        seen &= (1 << 386) - 1
        owned = int.from_bytes(backend.read(pointer2 + 0x28, 49), "little") & seen
        # Main.inBattle, GetFieldMessageBoxType, ScriptContext_IsEnabled, IsPlayerFieldControlsLocked.
        in_battle = bool(read_u8(backend, self._addresses["gMain"].address + 0x439) & 2)
        text_active = read_u8(backend, self._addresses["sMessageBoxType"].address) != 0
        # currentChar of TextPrinter 0, at offset 0 (include/text.h). AddTextPrinter in
        # src/text_printer.c uses the window id as the printer index; window 0 is the field
        # message box and B_WIN_MSG in battle.
        text_progress = read_u32(backend, self._addresses["sTextPrinters"].address)
        script_active = read_u8(backend, self._addresses["sGlobalScriptContextStatus"].address) == 0
        controls_locked = bool(read_u8(backend, self._addresses["sLockFieldControls"].address))
        # SaveBlock1.pos/location: include/global.h at the same source commit.
        x, y, map_group, map_num = struct.unpack("<hhbb", backend.read(pointer, 6))
        count = read_u8(backend, self._addresses["gPlayerPartyCount"].address)
        if count > 6:
            raise StateNotReady(f"Invalid gPlayerPartyCount: {count}")
        # On the title screen and intro the pointer is valid but the block is still empty.
        if (map_group, map_num, x, y) == (0, 0, 0, 0) and count == 0:
            raise StateNotReady("SaveBlock1 is empty: the game has not reached the overworld")
        party = []
        for index in range(count):
            # struct Pokemon: include/pokemon.h at the same source commit; live party only.
            address = self._addresses["gPlayerParty"].address + index * 100
            entry = backend.read(address, 100)
            status, level, hp, max_hp = struct.unpack_from("<IBxHH", entry, 0x50)
            personality, trainer_id = struct.unpack_from("<II", entry)
            key = personality ^ trainer_id
            decrypted = struct.pack(
                "<12I", *(word ^ key for word in struct.unpack_from("<12I", entry, 0x20))
            )
            checksum = sum(struct.unpack("<24H", decrypted)) & 0xFFFF
            if checksum != struct.unpack_from("<H", entry, 0x1C)[0]:
                raise StateNotReady(f"Invalid party checksum in slot {index}")
            order = SUBSTRUCT_ORDERS[personality % 24]
            species = struct.unpack_from("<H", decrypted, order.index(0) * 12)[0]
            moves = struct.unpack_from("<4H", decrypted, order.index(1) * 12)
            party.append(
                PartyPokemon(
                    level=level, hp=hp, max_hp=max_hp, status=status, species=species, moves=moves
                )
            )
        return FireRedState(
            map_group=map_group,
            map_num=map_num,
            x=x,
            y=y,
            party=tuple(party),
            money=money,
            badge_flags=badge_flags,
            trainers_defeated=trainers_defeated,
            pokedex_owned=owned.bit_count(),
            pokedex_seen=seen.bit_count(),
            in_battle=in_battle,
            text_active=text_active,
            text_progress=text_progress,
            script_active=script_active,
            controls_locked=controls_locked,
            story_flags=story_flags,
            world_map_flags=world_map_flags.bit_count(),
            text_speed=options[0] & 7,
            battle_scene_off=bool(options[1] & 4),
        )

    def check_options(self, backend: GameBackend) -> None:
        state = self.read(backend)
        if state.text_speed != 2 or not state.battle_scene_off:
            raise ValueError("FireRed requires fast text speed and battle scene off")
