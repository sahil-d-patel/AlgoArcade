# games

One folder per game: its StateReader, address table, start-state manifest and reward config. `base.py` holds the two protocols every game implements. `fake.py` reads the toy game.

- `firered/` starts with the address table in ENV-02 and the StateReader in ENV-05 (Prince). `manifest.py` lists the start states from ENV-06; `scripts/capture_states.py` regenerates them. `milestones.py` holds the ordered Pallet-to-Brock milestones from ENV-13. `maps.py` builds each map's walkability grid and the Kanto placement offline from the decomp's map files (ENV-10); `scripts/fetch_maps.py` fetches them.
- `super_mario_land/reader.py` reads lives, time, world, level progress and Mario's screen y (ENV-03, Akshaj). Its addresses come from the kaspermeerts/supermarioland disassembly, or from PyBoy's game wrapper where the disassembly has no label.
- `super_mario_land/manifest.py` lists the start state `level_1_1`, which `scripts/capture_sml_states.py` regenerates; `features.py` is its observation vector and `rewards.py` its reward table (TRN-05).

Contract: `docs/harness-spec.md`, StateReader and GameState, and FireRed memory map. No address enters this package without a symbol name and the commit it came from.
