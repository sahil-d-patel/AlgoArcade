# Adding a game

A draft, written from the Super Mario Land port ([port diary](port-diary-super-mario-land.md)). DOC-04 finishes it. `<game>` is the game's package name, such as `super_mario_land`. Each step names the file to create or change and the command that checks it. Run commands from the repo root; ROM tests skip without the ROM in `data/roms/`.

1. **Pin the ROM.** Add its SHA-1 and name to `KNOWN_ROMS` in `src/algoarcade/rom.py`, and to the ROM table in `docs/harness-spec.md`, ROM pinning. Put the dump in `data/roms/`; it never enters git. Check: `uv run pytest tests/test_rom.py -q`.

2. **Add a backend, only for a new console.** Create `src/algoarcade/backends/<emulator>.py`, implementing `GameBackend` from `backends/base.py`, and add one entry to the registry in `tests/conformance/test_backend.py`. Super Mario Land needed `backends/pyboy.py`. Check: `uv run pytest tests/conformance -q`.

3. **Write the reader.** Create `src/algoarcade/games/<game>/reader.py`: a frozen GameState with a `location` tuple whose last two elements are tile x and y, and a reader that raises `StateNotReady` until the game has set up its memory. Every address names its symbol and the commit it came from (`.claude/skills/ram-field/SKILL.md`). Test each field against an oracle in `tests/test_<game>.py`. Check: `uv run pytest tests/test_<game>.py -q`.

4. **Capture the start states.** Create `scripts/capture_<game>_states.py`, which writes `data/states/<name>.state` from power-on, and `src/algoarcade/games/<game>/manifest.py`, which lists each state's SHA-256, goal and step budget. Add the state to `docs/harness-spec.md`, Start states and segments. Check: a test that the script reproduces the manifest hash, `uv run pytest tests/training/test_<game>_training.py -k capture -q`.

5. **Define the observation.** Create `src/algoarcade/games/<game>/features.py`: a table of name to a function of the GameState scaled into [0, 1]. Check: Gymnasium's env checker in a test of the env from step 7, `uv run pytest tests/training/test_<game>_training.py -q`.

6. **Pick the reward table.** Create `src/algoarcade/games/<game>/rewards.py` with `COMPONENTS` (the registry components that make sense for the game) and `DEFAULT_WEIGHTS`. A new component follows `.claude/skills/reward-component/SKILL.md`, with an isolation test and an exploit note. Check: `uv run pytest tests/test_registry.py tests/test_exploits.py -q`.

7. **Wire the env.** In `src/algoarcade/training/games.py`, add a `<game>_env` factory that builds the backend, reader, reward registry and `AlgoArcadeEnv`, and an entry in `env_factory`. If a button must be held longer than 8 of the decision's 24 frames, as Super Mario Land's jump is, give the game its default in `game_hold`; a config can still set `hold`. Log 0 for any FireRed count the game lacks. Check: `uv run pytest tests/training/test_<game>_training.py -q`.

8. **Add the preset.** In `src/algoarcade/training/config.py`, add a preset made with `dataclasses.replace(BASELINE, ...)` that sets `env_id`, `start_state` and `reward_weights`, and list it in `PRESETS`. Leave `ppo`, `n_envs` and the normalization switches as they are. Check: `uv run pytest tests/training/test_training_config.py -q`.

9. **Run a smoke and queue the runs.** Train about 20,000 steps locally: `uv run python -m algoarcade.training.runner <game> --seed 0 --total-timesteps 20000 --runs-dir data/runs/smoke`. Then write `experiments/<id>.py` for the run queue. Check: `uv run python -m algoarcade.runqueue submit experiments/<id>.py --dry-run`.

10. **Run the gate.** Update the package READMEs you touched and `docs/decisions.md` for any choice a PM could veto. Check: `uv run poe check`.
