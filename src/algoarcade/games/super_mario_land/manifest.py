"""Super Mario Land start states. Contract: docs/harness-spec.md, Start states and segments.

The files live in data/states/ as <name>.state; scripts/capture_sml_states.py regenerates them.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from algoarcade.games.super_mario_land.reader import SuperMarioLandState

# Super Mario Land (World) rev A, the only dump in algoarcade.rom.
ROM_SHA1 = "418203621b887caa090215d97e3f509b79affd3e"
STATES = Path(__file__).resolve().parents[4] / "data" / "states"


@dataclass(frozen=True)
class StartState:
    name: str
    sha256: str
    goal: Callable[[SuperMarioLandState], bool]
    step_budget: int


def left_level_1_1(state: SuperMarioLandState) -> bool:
    return state.world != (1, 1)


# level_1_1: the first frame of play after START on the title, three lives and 400 on the timer.
# The timer runs out after about 675 decisions of 24 frames, so 2,048 decisions cover three full
# timers. Running 1-1 to its end takes about 80 decisions in the demo.
MANIFEST = (
    StartState(
        "level_1_1",
        "7e20b73bc50148ca04ba66047450b25f3188c006c3c37a73dab74f720958ae7c",
        left_level_1_1,
        2048,
    ),
)
