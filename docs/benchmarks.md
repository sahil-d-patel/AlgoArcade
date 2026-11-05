# Throughput benchmarks

Measured with `scripts/benchmark.py` (OPS-02, #13). Each cell runs one emulator per spawned process under a uniform random policy over the six default buttons, for 30 seconds, and reports the median of 3 runs. "State read" is one `read()` over EWRAM and IWRAM per decision. The menu check presses DOWN 20 times from a state with the start menu open and counts presses whose screen differs from a no-input step from the same state.

The project default is 24 frames per decision (hold 8, release 16). The other splits: 8 = 4/4, 16 = 8/8, 48 = 8/40.

## Mac (macOS 26.0 arm64)

Reproduce from the repo root, with `DYLD_FALLBACK_LIBRARY_PATH` set as in docs/setup.md, the ROM at `data/roms/firered.gba`, and the ENV-06 states plus a start-menu state in `data/states/`:

```sh
uv run python scripts/benchmark.py --backend mgba --rom data/roms/firered.gba --state data/states/s1.state --menu-state data/states/start_menu.state
```

`start_menu.state` is `s1.state` after one START press and 40 frames (the menu opens with the cursor on the player's name).

Command: `uv run python scripts/benchmark.py --backend mgba --rom data/roms/firered.gba --state data/states/s1.state --menu-state data/states/start_menu.state`
Commit: `66f2eab`
OS: macOS-26.0-arm64-arm-64bit, arm64, 12 logical CPUs
Python: 3.12.13
Date: 2026-11-03 08:23 UTC
Start state: `data/states/s1.state`

### Emulator frames per second, one process, 24 frames per decision

| Render | No state read | State read |
|---|---|---|
| off | 2,737 | 2,753 |
| on | 2,430 | 2,429 |

### Agent steps per second, random policy, render off, state read on

| Frames per decision | Hold / release | 4 envs | 8 envs | 12 envs | 16 envs |
|---|---|---|---|---|---|
| 8 | 4 / 4 | 1,283 | 2,277 | 2,759 | 2,798 |
| 16 | 8 / 8 | 687 | 1,194 | 1,457 | 1,461 |
| 24 | 8 / 16 | 458 | 802 | 962 | 1,005 |
| 48 | 8 / 40 | 228 | 404 | 491 | 506 |

### Menu presses that moved the cursor, out of 20

| Frames per decision | Hold / release | Moved |
|---|---|---|
| 8 | 4 / 4 | 20 |
| 16 | 8 / 8 | 20 |
| 24 | 8 / 16 | 20 |
| 48 | 8 / 40 | 20 |

Other jobs were running on the Mac during this run, so these are a lower bound. Scaling flattens past 12 environments on 12 logical CPUs.

## Training box (Windows desktop)

Not run yet. On the box, from this branch in PowerShell, with the same files in `data\roms\` and `data\states\`:

```
uv run python scripts/fetch_mgba.py
uv run python scripts/benchmark.py --backend mgba --rom data/roms/firered.gba --state data/states/s1.state --menu-state data/states/start_menu.state
```

Paste the output here as its own section.
