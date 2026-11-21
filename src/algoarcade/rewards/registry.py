"""Builds reward components from a config of name to weight. Contract: docs/harness-spec.md."""

from collections.abc import Callable, Hashable, Iterable, Mapping
from typing import Protocol

from algoarcade.games.base import GameState
from algoarcade.games.firered.reader import STORY_FLAGS
from algoarcade.rewards.base import RewardComponent
from algoarcade.rewards.explore import Explore
from algoarcade.rewards.party import Blackout, Heal, Party
from algoarcade.rewards.penalties import AntiLoop, AntiSpam, Dialogue, Time
from algoarcade.rewards.progress import NewMaximum, Progress


class FireRedFields(Progress, Party, Dialogue, Protocol):
    """Every field FireRed's components read."""


# FireRed's components. Another game passes its own table, so its state needs only the
# fields its components read.
#
# Bounds, from pret/pokefirered at ce6406d8. Constants are in include/constants/flags.h
# (FLAGS_COUNT, NUM_BADGES), global.h (PARTY_SIZE) and pokemon.h (MAX_LEVEL); script paths are
# under data/maps/ in that repository.
# - events: no script sets more than three flags at or above 0x230 (hide and temp flags
#   excluded) without waiting for the player; the most is Route25_SeaCottage's S.S. Ticket
#   script. The whitelist is a subset, so a step adds at most three, worth 3.0 against a
#   badge's 10.0. The maximum is the whitelist's length.
# - level: at most PARTY_SIZE * MAX_LEVEL. One Pokemon joins the party per step, so the sum
#   rises by at most MAX_LEVEL. LEVEL_CAP is a guess, not a game bound: a starter near Brock's
#   Onix (level 14) plus one more Pokemon. Rises past it pay nothing, so grass camping earns at
#   most LEVEL_CAP * 0.5 = 10.0 per episode, one badge.
# - badge: at most NUM_BADGES. Each badge is set by its own gym leader's script, so a step
#   earns at most one. The rise bound is what rejects a misread 0xFF; its popcount is 8.
# - milestone: at most the nine milestones in ENV-13. Two can arrive in one step:
#   PewterCity_Gym/scripts.inc sets FLAG_DEFEATED_BROCK and FLAG_BADGE01_GET one after the
#   other. The others are a battle, a map change or a text box apart; parcel delivered and
#   Pokedex received are eight text boxes apart in PalletTown_ProfessorOaksLab/scripts.inc.
LEVEL_CAP = 20

COMPONENTS: dict[str, Callable[[], RewardComponent[FireRedFields]]] = {
    "events": lambda: NewMaximum(
        "events", lambda s: s.story_flags, maximum=len(STORY_FLAGS), max_rise=3
    ),
    "explore": Explore,
    "level": lambda: NewMaximum(
        "level",
        lambda s: s.party_level_sum,
        maximum=600,
        max_rise=100,
        cap=1.0,
        ceiling=LEVEL_CAP,
    ),
    "badge": lambda: NewMaximum("badge", lambda s: s.badge_count, maximum=8, max_rise=1),
    "milestone": lambda: NewMaximum(
        "milestone", lambda s: s.milestone_count, maximum=9, max_rise=2
    ),
    "heal": Heal,
    "blackout": Blackout,
    "anti_loop": AntiLoop,
    "anti_spam": AntiSpam,
    "time": Time,
}

# Starting weights from docs/technical-plan.md, Reward design. They are guesses; TRN-08 tunes
# them. Penalties carry their sign here, so every raw value stays at or above zero.
WEIGHTS = {
    "events": 1.0,
    "explore": 0.02,
    "level": 0.5,
    "badge": 10.0,
    "milestone": 2.0,
    "heal": 0.3,
    "blackout": -5.0,
    "anti_loop": -0.01,
    "anti_spam": -0.02,
    "time": -0.001,
}
# Heal and the penalties are off by default; turning them on is experiment E5.
DEFAULT_WEIGHTS = {
    name: WEIGHTS[name] for name in ("events", "explore", "level", "badge", "milestone")
}


class RewardRegistry[S: GameState]:
    def __init__(
        self,
        components: Mapping[str, Callable[[], RewardComponent[S]]],
        weights: Mapping[str, float],
    ) -> None:
        unknown = sorted(set(weights) - set(components))
        if unknown:
            raise ValueError(
                f"unknown reward component(s) {unknown}; valid names: {sorted(components)}"
            )
        self._weights = dict(weights)
        self._components = [components[name]() for name in weights]

    def reset(
        self,
        state: S,
        visited: Iterable[tuple[Hashable, ...]] = (),
        bests: Mapping[str, int] | None = None,
    ) -> None:
        """An archive start passes its cell's visited set and `bests`: explore does not pay the
        visited locations, and a progress term pays nothing up to its best."""
        bests = bests or {}
        for component in self._components:
            if isinstance(component, Explore):
                component.reset(state, visited)
            elif isinstance(component, NewMaximum):
                component.reset(state, bests.get(component.name))
            else:
                component.reset(state)

    def bests(self) -> dict[str, int]:
        """Each progress term's best so far this episode, by name, for an archive cell."""
        return {
            c.name: c.best
            for c in self._components
            if isinstance(c, NewMaximum) and c.best is not None
        }

    def step(self, prev: S, curr: S) -> tuple[float, dict[str, float]]:
        """The weighted reward, and each component's raw value as `reward/<name>`."""
        raw = {c.name: c.score(prev, curr) for c in self._components}
        reward = sum(self._weights[name] * value for name, value in raw.items())
        return reward, {f"reward/{name}": value for name, value in raw.items()}
