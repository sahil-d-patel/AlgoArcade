from algoarcade.backends.base import Button
from algoarcade.backends.fake import FakeBackend
from algoarcade.games.fake import FakeReader
from algoarcade.rewards.explore import Explore

read = FakeReader().read


def start() -> tuple[FakeBackend, Explore]:
    backend = FakeBackend()
    explore = Explore()
    explore.reset(read(backend))
    return backend, explore


def walk(backend: FakeBackend, explore: Explore, *buttons: Button) -> list[float]:
    rewards = []
    for button in buttons:
        prev = read(backend)
        backend.press([button])
        rewards.append(explore.score(prev, read(backend)))
    return rewards


def test_reset_step_pays_nothing() -> None:
    backend, explore = start()
    state = read(backend)
    assert explore.score(state, state) == 0.0


def test_bumping_a_wall_pays_nothing() -> None:
    assert walk(*start(), Button.UP) == [0.0]


def test_new_tile_pays_once() -> None:
    assert walk(*start(), Button.DOWN, Button.UP, Button.DOWN) == [1.0, 0.0, 0.0]


def test_second_room_is_new_ground() -> None:
    backend, explore = start()
    walk(backend, explore, Button.DOWN, Button.DOWN, *[Button.RIGHT] * 7)
    assert read(backend).location == (0, 8, 3)
    # (1, 1, 3) has the same x and y as (0, 1, 3), visited two steps in.
    assert walk(backend, explore, Button.RIGHT) == [1.0]
    assert read(backend).location == (1, 1, 3)


def test_reset_clears_visited_tiles() -> None:
    explore = Explore()
    for _ in range(2):
        backend = FakeBackend()
        explore.reset(read(backend))
        assert walk(backend, explore, Button.DOWN) == [1.0]
