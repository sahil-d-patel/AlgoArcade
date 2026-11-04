import random
from dataclasses import dataclass
from itertools import pairwise

import pytest

from algoarcade.backends import fake
from algoarcade.backends.base import Button, GameBackend
from algoarcade.backends.fake import FakeBackend
from algoarcade.games.fake import FakeReader, FakeState
from algoarcade.rewards.explore import Explore
from algoarcade.rewards.registry import (
    COMPONENTS,
    DEFAULT_WEIGHTS,
    LEVEL_CAP,
    WEIGHTS,
    FireRedFields,
    RewardRegistry,
)

read = FakeReader().read
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


def start(weights: dict[str, float]) -> tuple[FakeBackend, RewardRegistry[FireRedFields]]:
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
        if rng.random() < 0.1:
            poke(backend, rng.choice((fake.HP_PERCENT, fake.TEXT)), rng.choice((0, 50, 100)))
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


def test_default_config_enables_exactly_the_five_positive_terms() -> None:
    assert set(DEFAULT_WEIGHTS) == {"events", "explore", "level", "badge", "milestone"}
    assert all(DEFAULT_WEIGHTS[name] == WEIGHTS[name] for name in DEFAULT_WEIGHTS)
    assert set(WEIGHTS) == set(COMPONENTS)


# The two exceptions are the penalties time and anti_spam, which never pay and are off by
# default. anti_spam passes here because the fake's text box is closed;
# test_time_and_anti_spam_charge_the_reset_step pins both.
@pytest.mark.parametrize("name", [name for name in COMPONENTS if name != "time"])
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


def test_time_charges_every_step_even_a_no_op() -> None:
    backend = FakeBackend()
    component = COMPONENTS["time"]()
    component.reset(read(backend))
    state = read(backend)
    assert component.score(state, state) == 1.0
    backend.press([Button.UP])
    assert component.score(state, read(backend)) == 1.0


@pytest.mark.parametrize(("text", "anti_spam"), [(0, 0.0), (1, 1.0)])
def test_time_and_anti_spam_charge_the_reset_step(text: int, anti_spam: float) -> None:
    backend = FakeBackend()
    poke(backend, fake.TEXT, text)
    registry = RewardRegistry(COMPONENTS, WEIGHTS)
    state = read(backend)
    registry.reset(state)
    _, info = registry.step(state, state)
    assert info.pop("reward/time") == 1.0
    assert info.pop("reward/anti_spam") == anti_spam
    assert all(value == 0.0 for value in info.values())


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


@pytest.mark.parametrize(
    ("name", "byte_values", "paid"),
    [
        # Badge counts 0, 0, 0, 1, 0, 0, 1, 1: one frame read high while a save block moved,
        # then the real badge.
        ("badge", [0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x01, 0x01], [0.0] * 6 + [1.0]),
        # Story-flag counts 3, 3, 3, 4, 3, 3, 4, 4.
        ("events", [0x07, 0x07, 0x07, 0x0F, 0x07, 0x07, 0x0F, 0x0F], [0.0] * 6 + [1.0]),
    ],
)
def test_a_one_frame_spike_pays_nothing_and_the_real_rise_pays_once(
    name: str, byte_values: list[int], paid: list[float]
) -> None:
    assert raw_values(name, byte_values) == paid


@pytest.mark.parametrize(
    ("name", "byte_values", "paid"),
    [
        ("level", [5, 6, 7, 7], [0.0, 1.0, 1.0]),
        # Story-flag counts 0, 2, 4, 4: the last read is more than max_rise above the best.
        ("events", [0x00, 0x03, 0x0F, 0x0F], [0.0, 2.0, 2.0]),
    ],
)
def test_a_climb_on_consecutive_steps_pays_each_rise_as_it_is_confirmed(
    name: str, byte_values: list[int], paid: list[float]
) -> None:
    assert raw_values(name, byte_values) == paid


@pytest.mark.parametrize(
    ("name", "byte_values"),
    [
        ("badge", [0x01, 0x01, 0xFF, 0x01, 0x01]),
        ("level", [5, 5, 200, 5, 5]),
    ],
)
def test_a_one_frame_spike_above_max_rise_pays_nothing(name: str, byte_values: list[int]) -> None:
    assert raw_values(name, byte_values) == [0.0] * 4


def test_level_drop_and_return_to_the_old_maximum_pays_nothing() -> None:
    paid = raw_values("level", [5, 6, 6, 4, 4, 6, 6, 7, 7])
    assert paid == [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]


def test_level_gain_is_capped_at_one_per_step_and_the_excess_is_lost() -> None:
    assert raw_values("level", [5, 9, 9, 10, 10]) == [0.0, 1.0, 0.0, 1.0]


def test_a_misread_badge_byte_neither_pays_nor_silences_later_badges() -> None:
    # Badge counts 0, 1, 8, 1, 1, 2, 2: 0xFF reads as eight badges on the step after the first
    # badge, so that badge pays once two later reads agree on it.
    paid = raw_values("badge", [0x00, 0x01, 0xFF, 0x01, 0x01, 0x03, 0x03])
    assert paid == [0.0, 0.0, 0.0, 1.0, 0.0, 1.0]


def test_level_pays_nothing_past_the_level_cap() -> None:
    climb = list(range(LEVEL_CAP - 3, LEVEL_CAP + 4))
    assert raw_values("level", climb) == [0.0, 1.0, 1.0, 1.0, 0.0, 0.0]


def test_a_count_above_its_maximum_is_ignored() -> None:
    assert raw_values("milestone", [0, 1, 1, 200, 2, 2]) == [0.0, 1.0, 0.0, 0.0, 1.0]


@pytest.mark.parametrize(
    ("name", "byte_values", "paid"),
    [
        # Above the maximum at reset.
        ("milestone", [200, 1, 1, 2, 2], [0.0, 0.0, 0.0, 1.0]),
        # 0xFF at reset reads as eight badges; the real count is one.
        ("badge", [0xFF, 0x01, 0x01, 0x03, 0x03], [0.0, 0.0, 0.0, 1.0]),
        # Too low at reset: the real counts are 3 badges and 5 story flags, each more than
        # max_rise above the reset read. Level cannot show this: a sum more than max_rise above
        # any reset read is past the level cap, where nothing pays.
        ("badge", [0x00, 0x07, 0x07, 0x0F, 0x0F], [0.0, 0.0, 0.0, 1.0]),
        ("events", [0x00, 0x1F, 0x1F, 0x3F, 0x3F], [0.0, 0.0, 0.0, 1.0]),
    ],
)
def test_a_bad_reset_read_neither_becomes_the_best_nor_freezes_the_term(
    name: str, byte_values: list[int], paid: list[float]
) -> None:
    assert raw_values(name, byte_values) == paid


@pytest.mark.parametrize(
    ("name", "byte_values", "paid"),
    [
        ("badge", [0x00, 0x01, 0x01], [0.0, 1.0]),
        ("events", [0x00, 0x03, 0x03], [0.0, 2.0]),
    ],
)
def test_a_rise_on_the_first_step_pays_on_the_second(
    name: str, byte_values: list[int], paid: list[float]
) -> None:
    assert raw_values(name, byte_values) == paid


@pytest.mark.parametrize(
    ("name", "byte_values", "paid"),
    [
        # A transient low read, then the real count again.
        ("level", [30, 0, 30, 30], [0.0, 0.0, 0.0]),
        # A transient high pair sets a high best, so the real 50 and 51 pay nothing.
        ("level", [50, 200, 201, 50, 51, 51], [0.0] * 5),
        # Badge counts 3, 0, 1, 2, 2, 3, 3 and story-flag counts 5, 0, 3, 5, 5. The pair of low
        # reads becomes the best and pays nothing, but the climb back from it pays. A component
        # cannot tell this from a bad reset read followed by real gains.
        ("badge", [0x07, 0x00, 0x01, 0x03, 0x03, 0x07, 0x07], [0.0, 0.0, 0.0, 1.0, 0.0, 1.0]),
        ("events", [0x1F, 0x00, 0x07, 0x1F, 0x1F], [0.0, 0.0, 0.0, 2.0]),
    ],
)
def test_re_establishing_the_best_pays_nothing(
    name: str, byte_values: list[int], paid: list[float]
) -> None:
    assert raw_values(name, byte_values) == paid


def test_a_misread_repeated_on_two_steps_becomes_the_best() -> None:
    # Badge counts 1, 1, 8, 8, 1, 2, 3. Two agreeing reads are trusted, so the real second and
    # third badges pay nothing. Only the reader can prevent this (ENV-05, ENV-08).
    paid = raw_values("badge", [0x01, 0x01, 0xFF, 0xFF, 0x01, 0x03, 0x07])
    assert paid == [0.0] * 6


def test_a_misread_story_flag_count_neither_pays_nor_silences_later_flags() -> None:
    # Story-flag counts 0, 1, 8, 1, 2, 2. 0xFF reads as eight flags; no script sets more than
    # three at once.
    paid = raw_values("events", [0x00, 0x01, 0xFF, 0x01, 0x03, 0x03])
    assert paid == [0.0, 0.0, 0.0, 1.0, 1.0]


def test_chest_pays_events_once() -> None:
    backend, registry = start({"events": 1.0})
    rewards = []
    for button in [Button.RIGHT, Button.RIGHT, Button.A, Button.A, Button.A]:
        prev = read(backend)
        backend.press([button])
        rewards.append(registry.step(prev, read(backend))[0])
    assert rewards == [0.0, 0.0, 0.0, 1.0, 0.0]


def test_reset_clears_per_episode_state() -> None:
    registry = RewardRegistry(COMPONENTS, {"explore": 1.0, "level": 1.0})
    for _ in range(2):
        backend = FakeBackend()
        registry.reset(read(backend))
        infos = []
        for button in [Button.DOWN, Button.UP]:
            prev = read(backend)
            backend.press([button])
            poke(backend, fake.LEVEL_SUM, 6)
            infos.append(registry.step(prev, read(backend))[1])
        assert infos == [
            {"reward/explore": 1.0, "reward/level": 0.0},
            {"reward/explore": 0.0, "reward/level": 1.0},
        ]


def total_reward(name: str, states: list[FakeState]) -> float:
    component = COMPONENTS[name]()
    component.reset(states[0])
    return sum(component.score(a, b) for a, b in pairwise(states))


@pytest.mark.parametrize("name", list(COUNT_BYTES))
def test_losing_and_regaining_a_count_pays_the_gain_once(name: str) -> None:
    backend = FakeBackend()
    base = backend.save_state()[COUNT_BYTES[name]]
    states = [read(backend)]
    for value in [base + 1, base] * 50:
        poke(backend, COUNT_BYTES[name], value)
        states.append(read(backend))
    assert total_reward(name, states) <= 1.0
