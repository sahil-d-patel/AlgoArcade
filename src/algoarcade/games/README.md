# games

One folder per game: its StateReader, address table, start-state manifest and reward config. `base.py` holds the two protocols every game implements. `fake.py` reads the toy game.

- `firered/` starts with the address table in ENV-02 and the StateReader in ENV-05 (Prince). `manifest.py` lists the start states from ENV-06; `scripts/capture_states.py` regenerates them.
- `super_mario_land/` arrives with ENV-03 (Akshaj).

Contract: `docs/harness-spec.md`, StateReader and GameState, and FireRed memory map. No address enters this package without a symbol name and the commit it came from.
