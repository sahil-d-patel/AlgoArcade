from dataclasses import replace
from itertools import pairwise

from algoarcade.backends.fake import FakeBackend
from algoarcade.games.fake import FakeReader, FakeState
from algoarcade.rewards.penalties import LOOP_WINDOW, AntiLoop, AntiSpam

START = FakeReader().read(FakeBackend())


def at(y: int) -> FakeState:
    return replace(START, y=y)


def walk(ys: list[int]) -> list[float]:
    anti_loop = AntiLoop()
    anti_loop.reset(at(ys[0]))
    return [anti_loop.score(at(a), at(b)) for a, b in pairwise(ys)]


def test_anti_loop_fires_on_stepping_back_onto_a_recent_tile() -> None:
    assert walk([1, 2, 3, 2, 1]) == [0.0, 0.0, 1.0, 1.0]


def test_anti_loop_never_fires_on_standing_still() -> None:
    assert walk([1, 1, 1, 2, 2]) == [0.0, 0.0, 0.0, 0.0]


def test_anti_loop_forgets_a_tile_after_the_window() -> None:
    # Leave tile 1, stand on tile 2 until 1 drops out of the window, then step back.
    ys = [1, *[2] * LOOP_WINDOW, 1]
    assert walk(ys)[-1] == 0.0
    assert walk([*ys[:-2], 1])[-1] == 1.0


def text(progress: int) -> FakeState:
    return replace(START, text_box_open=progress != 0, text_progress=progress)


def test_anti_spam_fires_on_a_press_that_left_the_text_where_it_was() -> None:
    anti_spam = AntiSpam()
    anti_spam.reset(text(1))
    assert anti_spam.score(text(1), text(1)) == 1.0
    assert anti_spam.score(text(1), text(2)) == 0.0
    assert anti_spam.score(text(1), text(0)) == 0.0
    assert anti_spam.score(text(0), text(1)) == 0.0
    assert anti_spam.score(text(0), text(0)) == 0.0
