---
name: ram-field
description: How a value read from game memory enters GameState in this repo. Use when adding or changing a field in a StateReader, an address table, anything under src/algoarcade/games, or when debugging a memory read that returns wrong or unstable values in FireRed or another game.
paths: src/algoarcade/games/**
---

# Adding a RAM field

A wrong address does not crash. It returns a believable number, the reward function pays on it, and the agent learns the wrong thing for a week. So every field goes through the same five steps.

1. Find the symbol. Search the pokefirered symbol file in `data/pokefirered/` (fetched at the commit pinned in DATA.md) for the variable. Struct offsets come from the decompilation's headers. Never take an address from a wiki, a forum post or memory. If the symbols do not cover it, use `scripts/state_diff.py` (from ENV-05) on two save states that differ by one change, then find the symbol that explains the address.

2. Add it to the address table with its symbol name and the commit it came from. The table is keyed by ROM SHA-1, because ROM addresses differ between revisions. File-local symbols carry a numeric suffix that changes between builds, so match on the base name.

3. Read it the safe way. Anything inside SaveBlock1 or SaveBlock2 is reached through `gSaveBlock1Ptr` or `gSaveBlock2Ptr`, and those blocks move on every warp and every return from a battle or menu. Dereference the pointer on every read and check it lands in its expected range. Never store the dereferenced address on the reader. Money is XORed with a key in SaveBlock2. Species and moves are inside the encrypted party block: decrypt, then validate the checksum, and treat a failed checksum as "not ready".

4. Write the validation test. Capture a save state where you know the true value from the screen, and assert the reader returns exactly that. Mark the test `rom`, and skip inside the test when the ROM, the state file or the binding is missing, or when the state was captured on the other ROM revision, as `load_capture` in `tests/test_firered_fields.py` does. Add a second state where the value differs, so a constant cannot pass.

5. Check the blast radius. A field used by a reward needs that reward's isolation test rerun. A field that should be logged is a schema change: before November 8 add it to the schema, after November 8 stop and ask a PM.

Locations are always `(map_group, map_num, x, y)`. Progress counts use only ranges that go up: the badge byte, trainer flags, world-map flags and the story whitelist. Temporary flags reset on every map load.

Battle fields are `battle_own` and `battle_enemy`, from `gBattleMons`: battler 0 is the player and 1 the opponent in a single battle. gBattleMons keeps the last battle's Pokémon after the battle ends, so read it only while `gMain.inBattle` is set, and only once the opponent's personality matches `gEnemyParty` at its `gBattlerPartyIndexes` slot. For about 30 frames after inBattle goes up, gBattleMons still holds the last battle's Pokémon, or zeros. The enemy box shows only level and an HP bar, so validate enemy HP against the bar width and the rest against `src/data/trainer_parties.h`.

The milestones are the nine predicates in `src/algoarcade/games/firered/milestones.py`, in order: starter_chosen, rival_beaten, parcel_delivered, pokedex_received, forest_exited, pewter_reached, gym_entered, brock_defeated, boulder_badge. `milestone_count` is the position, 1 to 9, of the furthest that holds, or 0. A flag, var or item id goes in `addresses.IDS` with its source file. A map-based milestone stops holding when the player leaves. Prefer a flag nothing clears, such as a world-map flag; otherwise rely on the env's per-episode best. The Oak's Lab rival battle sets its trainer flag and `FLAG_BEAT_RIVAL_IN_OAKS_LAB` on a loss too, so check for a win separately, as `rival_beaten` does.

`FireRedReader` in `src/algoarcade/games/firered/reader.py` returns a `FireRedState`. Each field and the symbol it is read from:

- `map_group`, `map_num`, `x`, `y`: `gSaveBlock1Ptr`. The `location` property is the four as a tuple.
- `party`: `gPlayerPartyCount` entries of `gPlayerParty`, each a `PartyPokemon(level, hp, max_hp, status, species, moves, is_egg)`. `species` and `moves` come from the encrypted block. The properties `party_count`, `party_level_sum` and `party_hp_frac` (total HP over total max HP, 0 when that total is 0) derive from it. `party_level_sum` and `party_hp_frac` leave eggs out (`is_egg` is PokemonSubstruct3's bit, `MON_DATA_IS_EGG`); `party_count` counts them.
- `money`: `gSaveBlock1Ptr`, XORed with the key at `gSaveBlock2Ptr`.
- `badge_flags`: the badge byte in the `gSaveBlock1Ptr` flags. The `badge_count` property is its popcount.
- `trainers_defeated`: popcount of the trainer flags in `gSaveBlock1Ptr`. `CB2_EndTrainerBattle` in `src/battle_setup.c` sets a trainer's flag after a win, and after a loss only in the Oak's Lab rival battle (`TRAINER_BATTLE_EARLY_RIVAL` with `RIVAL_BATTLE_HEAL_AFTER`).
- `story_flags`: how many of the seven `STORY_FLAGS` in reader.py are set. `FLAG_BEAT_RIVAL_IN_OAKS_LAB` (0x258) means the lab rival battle finished, won or lost.
- `world_map_flags`: popcount of the world-map flags in `gSaveBlock1Ptr`.
- `event_flags`: the names in `milestones.FLAGS` whose flag is set in `gSaveBlock1Ptr`, with ids from `addresses.IDS`.
- `parcel_delivered`: `VAR_MAP_SCENE_VIRIDIAN_CITY_MART` in `gSaveBlock1Ptr` is 1 or more and `ITEM_OAKS_PARCEL` is in neither the key items pocket nor the PC.
- `pokedex_owned`, `pokedex_seen`: the `gSaveBlock2Ptr` bits. A seen bit counts only if both seen copies in `gSaveBlock1Ptr` also have it, and an owned bit only if its seen bit counts.
- `text_speed`, `battle_scene_off`: saved options at `gSaveBlock2Ptr`. `check_options` raises `ValueError` unless they are 2 and True.
- `in_battle`: `gMain.inBattle`.
- `battle_own`, `battle_enemy`: `gBattleMons`, checked against `gEnemyParty` at the `gBattlerPartyIndexes` slot. Each is a `BattlePokemon(hp, max_hp, level, moves, pp)`, or None outside battle and until that check passes.
- `text_active`: `sMessageBoxType` is not 0. It means field text is printing, not that a box is open. It turns false when printing ends while the box is still up, and stays false in battle.
- `text_box_open`: printer 0 or 24 in `sTextPrinters` is active, or `sGlobalScriptContext` is in native mode with `nativePtr` at `WaitForAorBPress` (a finished box still up). `anti_spam` reads this. A yes/no menu reads false.
- `text_progress`: `currentChar` of printer 24 (`B_WIN_OAK_OLD_MAN`) while it is active, else of printer 0, in the field and in battle.
- `script_active`: `sGlobalScriptContextStatus` is 0 (running).
- `controls_locked`: `sLockFieldControls` is nonzero.

`read` raises `StateNotReady` when:

- `gSaveBlock1Ptr` or `gSaveBlock2Ptr` is not 0 to 0x7C bytes above its block in steps of 4.
- The two pointers are at different offsets from their blocks.
- `gPlayerPartyCount` is above 6.
- The save block is empty: position (0, 0, 0, 0) with an empty party, as on the title screen and intro.
- A party slot fails its checksum.

AlgoArcadeEnv retries a not-ready read one frame later, up to 8 times. On the frame the save blocks move, Pokédex counts can read 0; consumers need two agreeing reads.

To add a field: add the symbol to `ADDRESSES` in `addresses.py` and raise the entry count in `tests/test_firered_addresses.py`. Add the field to `FireRedState` and read it in `FireRedReader.read`, with a fake-memory test in `tests/test_firered_reader.py`. Put the screen-read test in `tests/test_firered_fields.py`: rows in `test_firered_fields_state`, or in `test_firered_fields_party` for a party slot field, with two captures that read different values on each ROM revision. Each capture is pinned in `CAPTURES` by SHA-256 and the revision it was made on, and runs only on that revision. Rev 1 captures come from `scripts/capture_validation_states.py`, which replays recorded input from power-on. Extend its `SEGMENTS` to reach a new state, or append a `BRANCHES` entry that starts from a saved state. `SCREEN` in `tests/test_firered_reader.py` holds only the ENV-05 location and plaintext party values.

Offsets, layouts and the rules are in `docs/harness-spec.md` under "FireRed memory map". ENV-05 owns this skill.

`uv run python scripts/state_diff.py A.state B.state` loads both states into a headless `MgbaBackend` on `data/roms/firered.gba` and prints each EWRAM byte that differs as `0xADDRESS: old -> new`. A SaveBlock1 field shows up at the address `gSaveBlock1Ptr` held in that state plus the field's offset, not at a fixed address.
