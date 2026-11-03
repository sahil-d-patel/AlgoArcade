from dataclasses import replace
from itertools import pairwise

import pytest

from algoarcade.backends.fake import FakeBackend
from algoarcade.games.fake import FakeReader, FakeState
from algoarcade.rewards.party import Blackout, Heal

START = FakeReader().read(FakeBackend())


def hp(frac: float, count: int = 1) -> FakeState:
    return replace(START, party_hp_frac=frac, party_count=count)


def heal_paid(states: list[FakeState]) -> list[float]:
    heal = Heal()
    heal.reset(states[0])
    return [heal.score(a, b) for a, b in pairwise(states)]


def test_heal_pays_the_rise_once_two_reads_agree() -> None:
    assert heal_paid([hp(0.25), hp(0.25), hp(0.75), hp(0.75), hp(0.5), hp(0.5)]) == [
        0.0,
        0.0,
        0.5,
        0.0,
        0.0,
    ]


@pytest.mark.parametrize(
    "states",
    [
        # One bad read, low then high, between agreeing reads.
        [hp(1.0), hp(1.0), hp(0.2), hp(1.0), hp(1.0)],
        [hp(0.5), hp(0.5), hp(0.9), hp(0.5), hp(0.5)],
        # A whiteout: the party sits fainted, then is healed in full.
        [hp(0.5), hp(0.5), hp(0.0), hp(0.0), hp(1.0), hp(1.0)],
        # The starter joins an empty party, which reads 0.0.
        [hp(0.0, count=0), hp(0.0, count=0), hp(1.0), hp(1.0)],
        # A healthy Pokemon joins a hurt party.
        [hp(0.5), hp(0.5), hp(0.75, count=2), hp(0.75, count=2)],
        # Reads outside 0 to 1 are garbage, even when they agree.
        [hp(0.5), hp(0.5), hp(1.5), hp(1.5)],
        [hp(-1.0), hp(-1.0), hp(0.5), hp(0.5)],
    ],
)
def test_heal_pays_nothing_without_a_real_heal(states: list[FakeState]) -> None:
    assert sum(heal_paid(states)) == 0.0


def test_heal_pays_again_after_the_party_size_changes() -> None:
    states = [hp(0.5), hp(0.5), hp(0.75, count=2), hp(0.75, count=2), hp(1.0, 2), hp(1.0, 2)]
    assert heal_paid(states) == [0.0, 0.0, 0.0, 0.0, 0.25]


def test_blackout_fires_once_on_the_whiteout_heal() -> None:
    blackout = Blackout()
    states = [hp(1.0), hp(0.5), hp(0.0), hp(0.0), hp(1.0), hp(1.0)]
    blackout.reset(states[0])
    paid = [blackout.score(a, b) for a, b in pairwise(states)]
    assert paid == [0.0, 0.0, 0.0, 1.0, 0.0]


@pytest.mark.parametrize(
    ("prev", "curr"),
    [
        (hp(0.0, count=0), hp(1.0)),
        (hp(0.0), hp(1.0, count=2)),
        (hp(0.0), hp(1.5)),
        (hp(0.0), hp(0.0)),
    ],
)
def test_blackout_needs_a_fainted_party_and_a_valid_heal(prev: FakeState, curr: FakeState) -> None:
    assert Blackout().score(prev, curr) == 0.0
