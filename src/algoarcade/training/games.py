"""The AlgoArcade games the runner trains on, by env_id: firered on mGBA, and fake for tests.

The parent process checks the ROM and the start state, then hands each SubprocVecEnv worker a
factory that builds its own emulator. A worker that raises reaches the parent only as EOFError,
so anything that can be checked before the workers start is checked here.
"""

import hashlib
from collections.abc import Callable, Mapping
from functools import partial
from pathlib import Path
from typing import Any

import gymnasium as gym

from algoarcade.backends.fake import FakeBackend
from algoarcade.backends.mgba import MgbaBackend
from algoarcade.envs.env import AlgoArcadeEnv
from algoarcade.games import fake
from algoarcade.games.base import StateReader
from algoarcade.games.firered.features import FEATURES as FIRERED_FEATURES
from algoarcade.games.firered.manifest import MANIFEST, STATES
from algoarcade.games.firered.reader import FireRedReader, FireRedState
from algoarcade.rewards.explore import Explore
from algoarcade.rewards.registry import COMPONENTS, RewardRegistry
from algoarcade.rom import identify

FIRERED_ROM = Path(__file__).resolve().parents[3] / "data" / "roms" / "firered.gba"
# The components FireRedState can feed until ENV-08 adds story_flags, badge_count and
# party_level_sum. Any other name in reward_weights is refused by the registry.
FIRERED_COMPONENTS = {"explore": Explore}
FAKE_STEP_BUDGET = 64


class Measures(gym.Wrapper[dict[str, Any], Any, dict[str, Any], Any]):
    """Adds `name: measure(state)` to every step's info, for the logs: the start state's goal
    and the episode-end counts the env does not report. It reads the state again after the
    env's own read; nothing has advanced in between, so the two agree."""

    def __init__(
        self,
        env: AlgoArcadeEnv,
        reader: StateReader,
        measures: Mapping[str, Callable[[Any], Any]],
    ) -> None:
        super().__init__(env)
        self._backend = env.backend
        self._reader = reader
        self._measures = measures

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        state = self._reader.read(self._backend)
        info |= {name: measure(state) for name, measure in self._measures.items()}
        return obs, float(reward), terminated, truncated, info


def firered_env(state: bytes, start_state: str, reward_weights: Mapping[str, float]) -> gym.Env:
    entry = next(e for e in MANIFEST if e.name == start_state)
    assert entry.step_budget is not None
    backend = MgbaBackend(FIRERED_ROM, render=False)
    reader = FireRedReader(backend.rom.sha1)
    rewards = RewardRegistry[FireRedState](FIRERED_COMPONENTS, reward_weights)
    env = AlgoArcadeEnv(backend, reader, rewards, state, entry.step_budget, FIRERED_FEATURES)
    measures: dict[str, Callable[[FireRedState], Any]] = {
        "goal": entry.goal,
        "badge_count": lambda s: s.badge_flags.bit_count(),
        "party_level_sum": lambda s: sum(p.level for p in s.party),
    }
    return Measures(env, reader, measures)


def fake_env(reward_weights: Mapping[str, float]) -> gym.Env:
    backend = FakeBackend()
    reader = fake.FakeReader()
    rewards = RewardRegistry(COMPONENTS, reward_weights)
    env = AlgoArcadeEnv(
        backend, reader, rewards, backend.save_state(), FAKE_STEP_BUDGET, fake.FEATURES
    )
    measures: dict[str, Callable[[fake.FakeState], Any]] = {
        "goal": lambda s: s.chest_opened,
        "badge_count": lambda s: s.badge_count,
        "party_level_sum": lambda s: s.party_level_sum,
        "story_flags": lambda s: s.story_flags,
    }
    return Measures(env, reader, measures)


def env_factory(
    env_id: str, start_state: str, reward_weights: Mapping[str, float]
) -> tuple[Callable[[], gym.Env], str, str]:
    """The worker factory, the backend name and the ROM SHA-1 for a run."""
    if env_id == "fake":
        return partial(fake_env, dict(reward_weights)), "fake", ""
    entry = next((e for e in MANIFEST if e.name == start_state), None)
    if entry is None or entry.step_budget is None:
        usable = [e.name for e in MANIFEST if e.step_budget is not None]
        raise ValueError(f"start_state {start_state!r}: use one of {usable}")
    path = STATES / f"{entry.name}.state"
    state = path.read_bytes()
    if hashlib.sha256(state).hexdigest() != entry.sha256:
        raise ValueError(f"{path} does not match the manifest hash")
    rom = identify(FIRERED_ROM)
    return partial(firered_env, state, entry.name, dict(reward_weights)), "mgba", rom.sha1
