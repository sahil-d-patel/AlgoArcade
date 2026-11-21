"""How long a decision must hold its button for Mario to pass the tall pipe at x_tile 103 in 1-1.

    uv run python scripts/sml_pipe_probe.py

Exploratory: one state. It reaches the pipe from level_1_1 by playing APPROACH, env actions at
hold 16 after reset seed 0's 6 no-op frames, until x_tile first reads 103. From that state it
plays each pattern for 40 decisions of 24 frames at each hold and prints the furthest x_tile in
1-1 (ENV-11, #31; docs/decisions.md, 2026-11-19).
"""

from algoarcade.backends.base import Button, GameBackend
from algoarcade.backends.pyboy import PyBoyBackend
from algoarcade.envs.env import ACTIONS, HOLD, RELEASE
from algoarcade.games.super_mario_land.manifest import STATES
from algoarcade.games.super_mario_land.reader import SuperMarioLandReader
from algoarcade.training.games import SML_ROM

PIPE = 103
APPROACH = "33333343333433334333433333333334333333330403"
PATTERNS: dict[str, list[tuple[Button, ...]]] = {
    "RIGHT": [(Button.RIGHT,)],
    "A": [(Button.A,)],
    "A, RIGHT": [(Button.A,), (Button.RIGHT,)],
    "A+RIGHT": [(Button.A, Button.RIGHT)],
    "A+B+RIGHT": [(Button.A, Button.B, Button.RIGHT)],
    "B+RIGHT, A+RIGHT": [(Button.B, Button.RIGHT), (Button.A, Button.RIGHT)],
}
HOLDS = range(8, 21, 2)


def pipe_state(backend: GameBackend) -> bytes:
    backend.load_state((STATES / "level_1_1.state").read_bytes())
    backend.step(6)
    reader = SuperMarioLandReader()
    for action in APPROACH:
        backend.press(ACTIONS[int(action)], 16, HOLD + RELEASE - 16)
        if reader.read(backend).x_tile == PIPE:
            return backend.save_state()
    raise RuntimeError(f"APPROACH never reached x_tile {PIPE}")


def furthest(
    backend: PyBoyBackend, state: bytes, pattern: list[tuple[Button, ...]], hold: int
) -> int:
    backend.load_state(state)
    reader = SuperMarioLandReader()
    best = 0
    for i in range(40):
        backend.press(pattern[i % len(pattern)], hold, HOLD + RELEASE - hold)
        read = reader.read(backend)
        if read.world == (1, 1):
            best = max(best, read.x_tile)
    return best


if __name__ == "__main__":
    backend = PyBoyBackend(SML_ROM, render=False)
    state = pipe_state(backend)
    print("| Hold / release | " + " | ".join(PATTERNS) + " |")
    print("|---" * (len(PATTERNS) + 1) + "|")
    for hold in HOLDS:
        row = [str(furthest(backend, state, pattern, hold)) for pattern in PATTERNS.values()]
        print(f"| {hold} / {HOLD + RELEASE - hold} | " + " | ".join(row) + " |")
