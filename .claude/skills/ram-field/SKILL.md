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

4. Write the validation test. Capture a save state where you know the true value from the screen, and assert the reader returns exactly that. Mark the test `rom`, and skip inside the test when the ROM, the state file or the binding is missing, as `test_captured_state_matches_the_screen` does. Add a second state where the value differs, so a constant cannot pass.

5. Check the blast radius. A field used by a reward needs that reward's isolation test rerun. A field that should be logged is a schema change: before November 1 add it to the schema, after November 1 stop and ask a PM.

Locations are always `(map_group, map_num, x, y)`. Progress counts use only ranges that go up: the badge byte, trainer flags, world-map flags and the story whitelist. Temporary flags reset on every map load.

`FireRedReader` in `src/algoarcade/games/firered/reader.py` returns a `FireRedState` with `map_group`, `map_num`, `x`, `y` and `party`, a tuple of `PartyPokemon(level, hp, max_hp, status)`; `party_count` and `location` derive from them. It raises `StateNotReady` when `gSaveBlock1Ptr` is not 0 to 0x7C bytes above `gSaveBlock1` in steps of 4, when `gPlayerPartyCount` is above 6, or when the save block is empty: position (0, 0, 0, 0) with an empty party, as on the title screen and intro. To add a field: add the symbol to `ADDRESSES` in `addresses.py` and raise the entry count in `tests/test_firered_addresses.py`, add the field to `FireRedState` and read it in `FireRedReader.read`, then add its screen value to every state in `SCREEN` in `tests/test_firered_reader.py`.

The memory map and the rules are in `docs/harness-spec.md` under "FireRed memory map". ENV-05 owns this skill.

`uv run python scripts/state_diff.py A.state B.state` loads both states into a headless `MgbaBackend` on `data/roms/firered.gba` and prints each EWRAM byte that differs as `0xADDRESS: old -> new`. A SaveBlock1 field shows up at the address `gSaveBlock1Ptr` held in that state plus the field's offset, not at a fixed address.
