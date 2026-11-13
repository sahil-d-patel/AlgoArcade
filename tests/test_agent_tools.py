import itertools
import json
from collections import deque
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pytest

from algoarcade.agent.tools import Field, Game, advance_dialogue, navigate_to, shortest_path
from algoarcade.backends import fake
from algoarcade.backends.base import Button, GameBackend
from algoarcade.backends.fake import FakeBackend
from algoarcade.games.fake import FakeField, FakeReader, FakeState

# The player starts at (1, 1). Walls force detours to the right half, (1, 3) and (1, 4) are
# walled in, and the door is on row 3 as in the fake's own room.
WALLED = (
    "##########",
    "#...#....#",
    "###.#.##.#",
    "#.#...#..D",
    "#.#####.##",
    "##########",
)
ROOMS = (WALLED, fake.ROOMS[1])
DOOR = (9, 3)


@pytest.fixture
def walled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fake, "ROOMS", ROOMS)


def game(backend: GameBackend | None = None, field: Field | None = None) -> Game:
    return Game(backend or FakeBackend(), FakeReader(), field or FakeField())


def breadth_first(rows: Iterable[str], start: tuple[int, int], goal: tuple[int, int]) -> int:
    """Presses on the shortest path, by plain breadth-first search over the room's floor."""
    grid = list(rows)
    distance = {start: 0}
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            tile = (x + dx, y + dy)
            inside = 0 <= tile[0] < len(grid[0]) and 0 <= tile[1] < len(grid)
            if inside and grid[tile[1]][tile[0]] != "#" and tile not in distance:
                distance[tile] = distance[(x, y)] + 1
                queue.append(tile)
    return distance[goal]


@pytest.mark.usefixtures("walled")
@pytest.mark.parametrize("goal", [(5, 1), (3, 1), (8, 3), (7, 4)])
def test_navigate_presses_as_many_buttons_as_the_shortest_path(goal: tuple[int, int]) -> None:
    played = game()
    result = navigate_to(played, *goal)
    state = FakeReader().read(played.backend)
    assert (result.reason, (state.x, state.y)) == ("arrived", goal)
    assert result.presses == breadth_first(WALLED, (1, 1), goal)


def test_the_path_turns_as_little_as_a_shortest_path_can() -> None:
    open_floor = np.ones((5, 5), dtype=bool)
    path = shortest_path(open_floor, (0, 0), (4, 4))
    assert path is not None
    assert len(path) == 8
    assert sum(a != b for a, b in itertools.pairwise(path)) == 1


@pytest.mark.usefixtures("walled")
@pytest.mark.parametrize("goal", [(1, 4), (4, 1), (20, 1)], ids=["walled in", "wall", "off map"])
def test_no_path_presses_nothing(goal: tuple[int, int]) -> None:
    result = navigate_to(game(), *goal)
    assert (result.reason, result.presses) == ("no_path", 0)


@pytest.mark.usefixtures("walled")
def test_walking_onto_the_door_is_a_warp() -> None:
    played = game()
    result = navigate_to(played, *DOOR)
    assert result.reason == "warp"
    assert result.presses == breadth_first(WALLED, (1, 1), DOOR)
    assert FakeReader().read(played.backend).room == 1


class TextAt(FakeBackend):
    """Opens a text box when the player steps onto `tile`, as an NPC's sight line would. Each A
    press then advances the text one page; after `pages` pages it closes."""

    def __init__(self, tile: tuple[int, int] = (0, 0), pages: int = 0) -> None:
        super().__init__()
        self.tile = tile
        self.pages = pages

    def press(self, buttons: Iterable[Button], hold: int = 8, release: int = 16) -> None:
        buttons = list(buttons)
        before = FakeReader().read(self)
        super().press(buttons, hold, release)
        ram = bytearray(self.save_state())
        here = (ram[fake.PLAYER_X], ram[fake.PLAYER_Y])
        if here == self.tile and here != (before.x, before.y):
            ram[fake.TEXT] = 1
        elif Button.A in buttons and ram[fake.TEXT]:
            ram[fake.TEXT] = (ram[fake.TEXT] + 1) % (self.pages + 1)
        self.load_state(bytes(ram))


class Grass(FakeField):
    """A wild battle starts on column 3."""

    def battle(self, state: FakeState) -> bool:
        return state.x == 3


@pytest.mark.usefixtures("walled")
def test_a_battle_stops_navigation() -> None:
    played = game(field=Grass())
    result = navigate_to(played, 5, 1)
    assert (result.reason, result.presses) == ("battle", 2)
    assert FakeReader().read(played.backend).x == 3


@pytest.mark.usefixtures("walled")
def test_a_text_box_stops_navigation() -> None:
    played = game(TextAt(tile=(3, 2)))
    result = navigate_to(played, 5, 1)
    assert (result.reason, result.presses) == ("dialogue", 3)


@pytest.mark.usefixtures("walled")
def test_navigation_stops_at_its_press_cap() -> None:
    result = navigate_to(game(), 5, 1, max_presses=3)
    assert (result.reason, result.presses) == ("press_cap", 3)


class Pages(FakeField):
    """The text is the page number; a choice shows on page `choice_on`."""

    def __init__(self, choice_on: int | None = None) -> None:
        self.choice_on = choice_on

    def text(self, backend: GameBackend, state: FakeState) -> str:
        return f"page {state.text_progress}" if state.text_progress else ""

    def choice(self, backend: GameBackend) -> bool:
        return FakeReader().read(backend).text_progress == self.choice_on


def open_box(pages: int) -> TextAt:
    backend = TextAt(tile=(2, 1), pages=pages)
    backend.press([Button.RIGHT])
    return backend


def test_advance_dialogue_presses_a_until_the_box_closes() -> None:
    result = advance_dialogue(game(open_box(pages=3), Pages()))
    assert (result.reason, result.presses, result.text) == ("closed", 3, "page 1\npage 2\npage 3")


def test_advance_dialogue_stops_before_answering_a_choice() -> None:
    backend = open_box(pages=5)
    result = advance_dialogue(game(backend, Pages(choice_on=2)))
    assert (result.reason, result.presses, result.text) == ("choice", 1, "page 1\npage 2")
    assert FakeReader().read(backend).text_progress == 2


def test_advance_dialogue_stops_at_its_press_cap() -> None:
    result = advance_dialogue(game(open_box(pages=9), Pages()), max_presses=4)
    assert (result.reason, result.presses) == ("press_cap", 4)


def test_advance_dialogue_with_no_box_presses_nothing() -> None:
    result = advance_dialogue(game(field=Pages()))
    assert (result.reason, result.presses, result.text) == ("closed", 0, "")


@pytest.mark.usefixtures("walled")
def test_every_press_is_traced_with_the_state_after_it(tmp_path: Path) -> None:
    trace = tmp_path / "presses.jsonl"
    with trace.open("w") as file:
        played = Game(FakeBackend(), FakeReader(), FakeField(), file)
        played.record(None)
        result = navigate_to(played, 3, 1)
    lines = [json.loads(line) for line in trace.read_text().splitlines()]
    assert [line["button"] for line in lines] == [None, "RIGHT", "RIGHT"]
    assert [line["state"]["x"] for line in lines] == [1, 2, 3]
    assert result.presses == len(lines) - 1
