"""The game an MCP server owns, and the L2 tools: navigate_to and advance_dialogue.

Contract: docs/harness-spec.md, MCP tool surface.
"""

import heapq
import json
import time
from dataclasses import dataclass
from typing import Any, Protocol, TextIO

import numpy as np
from numpy.typing import NDArray
from pydantic_core import to_jsonable_python

from algoarcade.backends.base import Button, GameBackend
from algoarcade.envs.env import NOT_READY_FRAMES
from algoarcade.games.base import StateNotReady, StateReader

# Guesses. A path across Route 1 is about 60 presses and the rival's lab speech about 15.
NAVIGATE_PRESSES = 400
DIALOGUE_PRESSES = 60
# Live collision does not show ledges or one-way tiles, so a step can fail. Each failure marks
# the tile and plans again, this many times at most. A guess.
PLANS = 8
STEPS = {Button.UP: (0, -1), Button.DOWN: (0, 1), Button.LEFT: (-1, 0), Button.RIGHT: (1, 0)}


class Field(Protocol):
    """What the tools read from a game's overworld besides its GameState."""

    def walkable(self, backend: GameBackend, /) -> NDArray[np.bool_]:
        """The current map as [y, x] in the location's x and y, False where the player cannot
        step, other characters included."""
        ...

    def text(self, backend: GameBackend, state: Any, /) -> str:
        """The message in the open text box."""
        ...

    def choice(self, backend: GameBackend, /) -> bool:
        """A menu waits for the player to choose."""
        ...

    def battle(self, state: Any, /) -> bool: ...


@dataclass
class Game:
    """One emulator, its reader and its field. With a trace open, every press appends a JSON
    line {"button", "time", "state"}: the button, the wall clock and the state after it, null
    when not ready. The server writes the start state first, with button null."""

    backend: GameBackend
    reader: StateReader
    field: Field
    trace: TextIO | None = None

    def press(self, button: Button) -> None:
        self.backend.press([button])
        self.record(button)

    def record(self, button: Button | None) -> None:
        if self.trace is None:
            return
        try:
            state = to_jsonable_python(self.reader.read(self.backend))
        except StateNotReady:
            state = None
        line = {
            "button": None if button is None else button.name,
            "time": time.time(),
            "state": state,
        }
        self.trace.write(json.dumps(line, separators=(",", ":")) + "\n")
        self.trace.flush()

    def read(self) -> Any:
        """The state, waiting a frame at a time while it is not ready, as AlgoArcadeEnv does."""
        for _ in range(NOT_READY_FRAMES):
            try:
                return self.reader.read(self.backend)
            except StateNotReady:
                self.backend.step(1)
        return self.reader.read(self.backend)


def shortest_path(
    walkable: NDArray[np.bool_], start: tuple[int, int], goal: tuple[int, int]
) -> list[Button] | None:
    """A* over the walkable tiles: the fewest steps, and among those the fewest turns, since
    on FireRed a press in a new direction can only turn the player. None when there is no path."""
    height, width = walkable.shape

    def distance(tile: tuple[int, int]) -> int:
        return abs(goal[0] - tile[0]) + abs(goal[1] - tile[1])

    # (estimate, turns, steps, tile, facing); facing is a Button value, -1 before the first step.
    frontier = [(distance(start), 0, 0, start, -1)]
    best = {(start, -1): (0, 0)}
    came_from: dict[tuple[tuple[int, int], int], tuple[tuple[int, int], int]] = {}
    while frontier:
        _, turns, steps, tile, facing = heapq.heappop(frontier)
        if tile == goal:
            path = []
            key = (tile, facing)
            while key in came_from:
                path.append(Button(key[1]))
                key = came_from[key]
            return path[::-1]
        if best[(tile, facing)] < (steps, turns):
            continue
        for button, (dx, dy) in STEPS.items():
            x, y = tile[0] + dx, tile[1] + dy
            if not (0 <= x < width and 0 <= y < height and walkable[y, x]):
                continue
            cost = (steps + 1, turns + (facing not in (-1, button)))
            key = ((x, y), int(button))
            if key in best and best[key] <= cost:
                continue
            best[key] = cost
            came_from[key] = (tile, facing)
            heapq.heappush(frontier, (cost[0] + distance((x, y)), cost[1], cost[0], (x, y), key[1]))
    return None


@dataclass(frozen=True)
class Navigation:
    reason: str  # arrived, battle, dialogue, warp, no_path or press_cap
    presses: int


def stopped(
    game: Game, state: Any, start_map: tuple[int, ...], goal: tuple[int, int]
) -> str | None:
    if game.field.battle(state):
        return "battle"
    if state.text_box_open:
        return "dialogue"
    if state.location[:-2] != start_map:
        return "warp"
    if state.location[-2:] == goal:
        return "arrived"
    return None


def navigate_to(  # noqa: C901 a loop per plan, per step and per press
    game: Game, x: int, y: int, max_presses: int = NAVIGATE_PRESSES
) -> Navigation:
    """Walk to (x, y) on the current map. Stops on arrival, a battle, a text box, a change of
    map, no path, or after `max_presses` presses."""
    state = game.read()
    start_map = state.location[:-2]
    presses = 0
    failed: set[tuple[int, int]] = set()
    for _ in range(PLANS):
        if reason := stopped(game, state, start_map, (x, y)):
            return Navigation(reason, presses)
        walkable = game.field.walkable(game.backend)
        for fx, fy in failed:
            walkable[fy, fx] = False
        path = shortest_path(walkable, state.location[-2:], (x, y))
        if path is None:
            return Navigation("no_path", presses)
        for button in path:
            here = state.location[-2:]
            # The first press in a new direction may only turn the player, so a step gets two.
            for _ in range(2):
                if reason := stopped(game, state, start_map, (x, y)):
                    return Navigation(reason, presses)
                if presses >= max_presses:
                    return Navigation("press_cap", presses)
                game.press(button)
                presses += 1
                state = game.read()
                if state.location[-2:] != here:
                    break
            ahead = (here[0] + STEPS[button][0], here[1] + STEPS[button][1])
            if state.location[-2:] != ahead:
                if state.location[-2:] == here:
                    failed.add(ahead)
                break
    return Navigation(stopped(game, state, start_map, (x, y)) or "no_path", presses)


@dataclass(frozen=True)
class Dialogue:
    reason: str  # closed, choice or press_cap
    presses: int
    text: str


def advance_dialogue(game: Game, max_presses: int = DIALOGUE_PRESSES) -> Dialogue:
    """Press A until the text box closes or a choice shows, at most `max_presses` times. The
    text is each distinct message the box held, one per line."""
    messages: list[str] = []
    presses = 0
    while True:
        state = game.read()
        choice = game.field.choice(game.backend)
        if choice or state.text_box_open:
            message = game.field.text(game.backend, state)
            if message and message not in messages[-1:]:
                messages.append(message)
        if choice:
            reason = "choice"
        elif not state.text_box_open:
            reason = "closed"
        elif presses >= max_presses:
            reason = "press_cap"
        else:
            game.press(Button.A)
            presses += 1
            continue
        return Dialogue(reason, presses, "\n".join(messages))
