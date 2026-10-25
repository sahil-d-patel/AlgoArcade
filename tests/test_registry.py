import random
from itertools import pairwise

import pytest

from algoarcade.backends import fake
from algoarcade.backends.base import Button, GameBackend
from algoarcade.backends.fake import FakeBackend
from algoarcade.games.fake import FakeReader, FakeState
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


def start(weights: dict[str, float]) -> tuple[FakeBackend, RewardRegistry]:
    backend = FakeBackend()
    registry = RewardRegistry(weights)
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
            poke(backend, rng.choice(list(COUNT_BYTES.values())), rng.randrange(0, 16))
        results.append(registry.step(prev, read(backend)))
    return results


def test_unknown_component_raises_and_lists_valid_names() -> None:
    with pytest.raises(ValueError, match="'speed'") as error:
        RewardRegistry({"explore": 1.0, "speed": 1.0})
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


def test_level_drop_and_return_to_the_old_maximum_pays_nothing() -> None:
    backend, registry = start({"level": 0.5})

    def level_to(value: int) -> float:
        prev = read(backend)
        poke(backend, fake.LEVEL_SUM, value)
        return registry.step(prev, read(backend))[1]["reward/level"]

    assert [level_to(v) for v in (6, 4, 6, 7)] == [1.0, 0.0, 0.0, 1.0]


def test_level_gain_is_capped_at_one_per_step_and_the_excess_is_lost() -> None:
    backend, registry = start({"level": 0.5})

    def level_to(value: int) -> float:
        prev = read(backend)
        poke(backend, fake.LEVEL_SUM, value)
        return registry.step(prev, read(backend))[1]["reward/level"]

    assert [level_to(v) for v in (9, 9, 10)] == [1.0, 0.0, 1.0]


def test_chest_pays_events_once() -> None:
    backend, registry = start({"events": 1.0})
    rewards = []
    for button in [Button.RIGHT, Button.RIGHT, Button.A, Button.A]:
        prev = read(backend)
        backend.press([button])
        rewards.append(registry.step(prev, read(backend))[0])
    assert rewards == [0.0, 0.0, 1.0, 0.0]


def test_reset_clears_per_episode_state() -> None:
    registry = RewardRegistry({"explore": 1.0, "level": 1.0})
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
    for value in range(start_level + 1, start_level + 41):
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
