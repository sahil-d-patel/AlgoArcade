import random
from dataclasses import dataclass
from itertools import pairwise

import pytest

from algoarcade.backends import fake
from algoarcade.backends.base import Button, GameBackend
from algoarcade.backends.fake import FakeBackend
from algoarcade.games.fake import FakeReader, FakeState
from algoarcade.rewards.explore import Explore
from algoarcade.rewards.progress import Progress
from algoarcade.rewards.registry import COMPONENTS, RewardRegistry

read = FakeReader().read
WEIGHTS = {"events": 1.0, "explore": 0.02, "level": 0.5, "badge": 10.0, "milestone": 2.0}
BUTTONS = [Button.UP, Button.DOWN, Button.LEFT, Button.RIGHT, Button.A, Button.B]
# The fake game has no way to earn levels, badges or milestones, so tests write the byte directly.
COUNT_BYTES = {
    "events": fake.FLAGS,
    "level": fake.LEVEL_SUM,
    "badge": fake.BADGES,
    "milestone": fake.MILESTONES,
}


def poke(backend: GameBackend, offset: int, value: int) -> None:
    ram = bytearray(backend.save_state())
    ram[offset] = value
    backend.load_state(bytes(ram))


def start(weights: dict[str, float]) -> tuple[FakeBackend, RewardRegistry[Progress]]:
    backend = FakeBackend()
    registry = RewardRegistry(COMPONENTS, weights)
    registry.reset(read(backend))
    return backend, registry


def rollout(steps: int, seed: int) -> list[tuple[float, dict[str, float]]]:
    rng = random.Random(seed)
    backend, registry = start(WEIGHTS)
    results = []
    for _ in range(steps):
        prev = read(backend)
        backend.press([rng.choice(BUTTONS)])
        if rng.random() < 0.2:
            # Mostly up, one step at a time: NewMaximum treats a larger jump as a bad read.
            offset = rng.choice(list(COUNT_BYTES.values()))
            poke(backend, offset, max(0, backend.save_state()[offset] + rng.choice((-1, 1, 1))))
        results.append(registry.step(prev, read(backend)))
    return results


def test_unknown_component_raises_and_lists_valid_names() -> None:
    with pytest.raises(ValueError, match="'speed'") as error:
        RewardRegistry(COMPONENTS, {"explore": 1.0, "speed": 1.0})
    for name in COMPONENTS:
        assert name in str(error.value)


def test_reward_is_the_weighted_sum_of_raw_values() -> None:
    results = rollout(200, seed=1)
    for reward, info in results:
        assert set(info) == {f"reward/{name}" for name in WEIGHTS}
        expected = sum(WEIGHTS[name] * info[f"reward/{name}"] for name in WEIGHTS)
        assert reward == pytest.approx(expected, abs=1e-9)
    for name in WEIGHTS:
        assert any(info[f"reward/{name}"] > 0 for _, info in results), f"{name} never paid"


@dataclass(frozen=True)
class Spot:
    location: tuple[int, int]


def test_explore_alone_needs_only_a_location() -> None:
    registry = RewardRegistry({"explore": Explore}, {"explore": 1.0})
    registry.reset(Spot((0, 0)))
    assert registry.step(Spot((0, 0)), Spot((0, 1))) == (1.0, {"reward/explore": 1.0})


def test_info_has_only_configured_components() -> None:
    backend, registry = start({"badge": 10.0})
    prev = read(backend)
    backend.press([Button.DOWN])
    assert registry.step(prev, read(backend)) == (0.0, {"reward/badge": 0.0})


@pytest.mark.parametrize("name", list(COMPONENTS))
def test_zero_on_reset_and_on_a_no_op_step(name: str) -> None:
    backend = FakeBackend()
    component = COMPONENTS[name]()
    component.reset(read(backend))
    state = read(backend)
    assert component.score(state, state) == 0.0
    backend.press([])
    assert component.score(state, read(backend)) == 0.0
    prev = read(backend)
    backend.press([Button.UP])  # into the wall above the start tile
    assert component.score(prev, read(backend)) == 0.0


def test_no_component_goes_negative() -> None:
    results = rollout(1000, seed=2)
    assert all(value >= 0.0 for _, info in results for value in info.values())


def raw_values(name: str, byte_values: list[int]) -> list[float]:
    """The first value is in the count byte at reset; each later one is one step."""
    backend = FakeBackend()
    poke(backend, COUNT_BYTES[name], byte_values[0])
    registry = RewardRegistry(COMPONENTS, {name: 1.0})
    registry.reset(read(backend))
    paid = []
    for value in byte_values[1:]:
        prev = read(backend)
        poke(backend, COUNT_BYTES[name], value)
        paid.append(registry.step(prev, read(backend))[1][f"reward/{name}"])
    return paid


def test_level_drop_and_return_to_the_old_maximum_pays_nothing() -> None:
    assert raw_values("level", [5, 6, 4, 6, 7]) == [1.0, 0.0, 0.0, 1.0]


def test_level_gain_is_capped_at_one_per_step_and_the_excess_is_lost() -> None:
    assert raw_values("level", [5, 9, 9, 10]) == [1.0, 0.0, 1.0]


def test_a_misread_badge_byte_neither_pays_nor_silences_later_badges() -> None:
    # 0xFF reads as eight badges one step after the first; a step earns at most one.
    assert raw_values("badge", [0x00, 0x01, 0xFF, 0x01, 0x03, 0x07]) == [1.0, 0.0, 0.0, 1.0, 1.0]


def test_a_count_above_its_maximum_is_ignored() -> None:
    assert raw_values("milestone", [0, 1, 200, 2]) == [1.0, 0.0, 1.0]


@pytest.mark.parametrize(
    ("name", "byte_values", "paid"),
    [
        # Above the maximum at reset.
        ("milestone", [200, 1, 1, 2], [0.0, 0.0, 1.0]),
        # 0xFF at reset reads as eight badges; the real count is one.
        ("badge", [0xFF, 0x01, 0x03, 0x07], [0.0, 1.0, 1.0]),
        # Too low at reset: the real counts are 3 badges and a level sum of 150, each more than
        # max_rise above the reset read.
        ("badge", [0x00, 0x07, 0x07, 0x0F], [0.0, 0.0, 1.0]),
        ("level", [0, 150, 150, 151], [0.0, 0.0, 1.0]),
    ],
)
def test_a_bad_reset_read_neither_becomes_the_best_nor_freezes_the_term(
    name: str, byte_values: list[int], paid: list[float]
) -> None:
    assert raw_values(name, byte_values) == paid


def test_a_misread_story_flag_count_neither_pays_nor_silences_later_flags() -> None:
    # 0xFF reads as eight flags; no script sets more than three at once.
    assert raw_values("events", [0x00, 0x01, 0xFF, 0x01, 0x03]) == [1.0, 0.0, 0.0, 1.0]


def test_chest_pays_events_once() -> None:
    backend, registry = start({"events": 1.0})
    rewards = []
    for button in [Button.RIGHT, Button.RIGHT, Button.A, Button.A]:
        prev = read(backend)
        backend.press([button])
        rewards.append(registry.step(prev, read(backend))[0])
    assert rewards == [0.0, 0.0, 1.0, 0.0]


def test_reset_clears_per_episode_state() -> None:
    registry = RewardRegistry(COMPONENTS, {"explore": 1.0, "level": 1.0})
    for _ in range(2):
        backend = FakeBackend()
        registry.reset(read(backend))
        prev = read(backend)
        backend.press([Button.DOWN])
        poke(backend, fake.LEVEL_SUM, 6)
        _, info = registry.step(prev, read(backend))
        assert info == {"reward/explore": 1.0, "reward/level": 1.0}


def total_reward(name: str, states: list[FakeState]) -> float:
    component = COMPONENTS[name]()
    component.reset(states[0])
    return sum(component.score(a, b) for a, b in pairwise(states))


def test_exploit_pacing_between_two_tiles_pays_one_tile() -> None:
    backend = FakeBackend()
    states = [read(backend)]
    for button in [Button.DOWN, Button.UP] * 50:
        backend.press([button])
        states.append(read(backend))
    assert total_reward("explore", states) <= 1.0


def test_exploit_grass_camping_pays_at_most_one_level_per_step() -> None:
    backend = FakeBackend()
    start_level = backend.save_state()[fake.LEVEL_SUM]
    states = [read(backend)]
    for value in range(start_level + 3, start_level + 121, 3):
        poke(backend, fake.LEVEL_SUM, value)
        states.append(read(backend))
    assert total_reward("level", states) <= 40.0


@pytest.mark.parametrize("name", list(COUNT_BYTES))
def test_exploit_losing_and_regaining_a_count_pays_the_gain_once(name: str) -> None:
    backend = FakeBackend()
    base = backend.save_state()[COUNT_BYTES[name]]
    states = [read(backend)]
    for value in [base + 1, base] * 50:
        poke(backend, COUNT_BYTES[name], value)
        states.append(read(backend))
    assert total_reward(name, states) <= 1.0
