# games

One folder per game: its StateReader, address table, start-state manifest and reward config. `base.py` holds the two protocols every game implements. `fake.py` reads the toy game.

- `firered/` starts with the address table in ENV-02 and the StateReader in ENV-05 (Prince).
- `super_mario_land/reader.py` reads lives, time, world and level progress (ENV-03, Akshaj). Its addresses come from the kaspermeerts/supermarioland disassembly, or from PyBoy's game wrapper where the disassembly has no label.

Contract: `docs/harness-spec.md`, StateReader and GameState, and FireRed memory map. No address enters this package without a symbol name and the commit it came from.
