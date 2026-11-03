"""Replay recorded input from power-on and write the ENV-08 validation states.

Run `uv run python scripts/capture_validation_states.py` with the rev 1 ROM at
data/roms/firered.gba. It writes data/states/validate_<name>.state, one per segment, and
validate_poisoned.state when data/states/seg_05_forest.state (from scripts/capture_states.py, #69)
is present. The expected values in tests/test_firered_fields.py were read off the screen at these
states.
"""

from pathlib import Path

from algoarcade.backends.base import Button, GameBackend
from algoarcade.backends.mgba import MgbaBackend

ROOT = Path(__file__).resolve().parents[1]
ROM = ROOT / "data" / "roms" / "firered.gba"
STATES = ROOT / "data" / "states"

# Each segment continues from the one before. A button name presses it for 8 frames and
# releases for 16; NAME*N does that N times; wN waits N frames.
SEGMENTS = (
    # New game in the bedroom with the default options; the player is named A, the rival GREEN.
    (
        "bedroom_default",
        "w600 START w200 START w120 START w90 w120 A*10 w60 A*20 w60 A*12 w30 A*20 w30 w60 A*3 "
        "w30 A*2 w60 w150 w120 A w60 A*6 w120 A START A w120 w90 A w60 w60 A*8 w90 A w120 DOWN A "
        "w60 A*14 w300 A*10 w400 w60",
    ),
    # Trainer card, then OPTION: text speed FAST, battle scene OFF.
    ("bedroom", "START w30 DOWN A w90 B w60 DOWN DOWN A w60 RIGHT DOWN RIGHT w10 B w60 B w60"),
    # Out of the house, Oak stops the player on Route 1, Squirtle from the middle ball, no nickname.
    (
        "starter",
        "UP UP UP UP RIGHT RIGHT RIGHT RIGHT RIGHT UP UP UP UP w60 UP w120 LEFT w30 LEFT w150 "
        "DOWN*6 w30 DOWN LEFT*6 DOWN w150 LEFT LEFT DOWN DOWN w150 RIGHT*8 UP*9 w60 B w30 LEFT "
        "LEFT UP*9 w60 B*60 w60 B*40 w60 DOWN DOWN RIGHT RIGHT RIGHT RIGHT UP w20 A w60 A w60 A "
        "w60 B w60 B w90 B w60 B w60 B*30 w60",
    ),
    # The rival's challenge is printing.
    ("rival_text", "DOWN DOWN DOWN w60 LEFT LEFT LEFT LEFT DOWN DOWN w90"),
    # The rival battle has started.
    ("battle", "B*15 w200"),
    # Won with Tackle; back in the lab.
    ("after_battle", "A*60 w60 A*120 w60"),
    # Out the lab door into Pallet Town.
    ("pallet", "DOWN*6 w120"),
    # Route 1 to Viridian City, running from wild battles.
    (
        "viridian",
        "LEFT*6 UP*15 w60 RIGHT RIGHT UP*12 w60 LEFT LEFT LEFT LEFT LEFT UP UP UP UP UP UP RIGHT "
        "RIGHT RIGHT RIGHT RIGHT B B B B B B B DOWN RIGHT A w30 B B B DOWN RIGHT A w30 B B B DOWN "
        "RIGHT A w30 B B B DOWN RIGHT A w30 UP UP UP UP UP UP UP LEFT LEFT LEFT UP UP UP UP UP "
        "RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT UP UP UP B B B B B B B DOWN RIGHT A w30 B B B "
        "DOWN RIGHT A w30 B B B DOWN RIGHT A w30 B B B DOWN RIGHT A w30 UP UP UP UP UP UP UP UP "
        "UP UP UP UP UP LEFT LEFT LEFT LEFT UP UP UP UP",
    ),
    # Into the Viridian Mart, where the clerk hands over Oak's Parcel.
    (
        "parcel",
        "UP UP UP UP UP UP UP UP LEFT LEFT LEFT LEFT UP UP UP UP UP UP UP UP UP UP UP UP RIGHT "
        "RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT UP B "
        "B B w60 B*30 w30",
    ),
    # Back to Oak's Lab; Oak takes the parcel and gives the Pokedex.
    (
        "pokedex",
        "DOWN DOWN DOWN DOWN DOWN DOWN DOWN B UP UP B B DOWN DOWN DOWN DOWN DOWN DOWN DOWN DOWN "
        "DOWN DOWN DOWN DOWN DOWN DOWN DOWN LEFT LEFT LEFT LEFT LEFT LEFT LEFT LEFT LEFT LEFT "
        "LEFT LEFT DOWN DOWN DOWN DOWN DOWN DOWN DOWN DOWN DOWN DOWN DOWN DOWN DOWN DOWN DOWN "
        "DOWN DOWN DOWN DOWN DOWN LEFT LEFT LEFT LEFT LEFT DOWN DOWN DOWN DOWN DOWN DOWN RIGHT "
        "RIGHT RIGHT DOWN DOWN DOWN DOWN DOWN DOWN RIGHT RIGHT DOWN DOWN DOWN DOWN DOWN B B B B B "
        "B B DOWN RIGHT A w30 B B B DOWN RIGHT A w30 B B B DOWN RIGHT A w30 B B B DOWN RIGHT A "
        "w30 DOWN DOWN DOWN DOWN DOWN DOWN DOWN DOWN DOWN DOWN DOWN DOWN DOWN DOWN DOWN DOWN DOWN "
        "DOWN DOWN DOWN DOWN DOWN DOWN DOWN DOWN DOWN RIGHT RIGHT RIGHT RIGHT RIGHT UP B B B w90 "
        "UP*9 w30 A w30 B*80 w60 B*60 w60",
    ),
)

# From seg_05_forest: into the grass, Tail Whip a wild Weedle until Poison Sting lands, then
# Water Gun. The party screen shows SQUIRTLE PSN 30/37.
POISONED = (
    "UP UP LEFT*4 UP*3 LEFT LEFT "
    + "RIGHT RIGHT LEFT LEFT " * 9
    + "w240 B*6 w30 A w30 RIGHT w4 A A*37 B*15 A w30 LEFT A A*40 B*10 w60"
)


def play(backend: GameBackend, tokens: str) -> None:
    for token in tokens.split():
        if token.startswith("w"):
            backend.step(int(token[1:]))
            continue
        name, _, repeat = token.partition("*")
        for _ in range(int(repeat or 1)):
            backend.press([Button[name]], hold=8, release=16)


def main() -> None:
    backend = MgbaBackend(ROM)
    STATES.mkdir(exist_ok=True)
    try:
        for name, tokens in SEGMENTS:
            play(backend, tokens)
            path = STATES / f"validate_{name}.state"
            path.write_bytes(backend.save_state())
            print(path.relative_to(ROOT))
        forest = STATES / "seg_05_forest.state"
        if forest.is_file():
            backend.load_state(forest.read_bytes())
            play(backend, POISONED)
            path = STATES / "validate_poisoned.state"
            path.write_bytes(backend.save_state())
            print(path.relative_to(ROOT))
    finally:
        backend.close()


if __name__ == "__main__":
    main()
