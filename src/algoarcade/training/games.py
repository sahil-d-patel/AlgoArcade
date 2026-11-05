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

from algoarcade.backends.base import GameBackend
from algoarcade.backends.fake import FakeBackend
from algoarcade.backends.mgba import MgbaBackend
from algoarcade.envs.env import AlgoArcadeEnv
from algoarcade.games import fake
from algoarcade.games.base import GameState, StateReader
from algoarcade.games.firered.features import FEATURES as FIRERED_FEATURES
from algoarcade.games.firered.manifest import MANIFEST, ROM_SHA1, STATES
from algoarcade.games.firered.reader import FireRedReader, FireRedState
from algoarcade.rewards.registry import COMPONENTS, RewardRegistry
from algoarcade.rom import identify

FIRERED_ROM = Path(__file__).resolve().parents[3] / "data" / "roms" / "firered.gba"
FAKE_STEP_BUDGET = 64
# Logged at the episode's end. A save block read on the frame the block moves can dip or read
# high (docs/harness-spec.md, FireRed memory map), so each is the value of the latest two
# consecutive reads that agreed.
COUNTS = ("badge_count", "party_level_sum", "story_flags")


class LastRead:
    """A reader that keeps the last state it returned, so the logs see the state the env used,
    after its not-ready retry, without reading memory a second time."""

    def __init__(self, reader: StateReader) -> None:
        self._reader = reader
        self.state: Any = None

    def read(self, backend: GameBackend) -> GameState:
        self.state = self._reader.read(backend)
        return self.state


class Measures(gym.Wrapper[dict[str, Any], Any, dict[str, Any], Any]):
    """Adds to each step's info what the logs need and the env does not report: `goal`,
    `in_battle`, and each of COUNTS. Until two reads agree on a count, the reset read stands."""

    def __init__(
        self,
        env: AlgoArcadeEnv,
        reads: LastRead,
        goal: Callable[[Any], bool],
        in_battle: Callable[[Any], bool],
    ) -> None:
        super().__init__(env)
        self._reads = reads
        self._goal = goal
        self._in_battle = in_battle
        self._last: dict[str, int] = {}
        self._agreed: dict[str, int] = {}

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        self._last = {name: getattr(self._reads.state, name) for name in COUNTS}
        self._agreed = dict(self._last)
        return obs, info

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        state = self._reads.state
        for name in COUNTS:
            value = getattr(state, name)
            if value == self._last[name]:
                self._agreed[name] = value
            self._last[name] = value
        info |= self._agreed | {"goal": self._goal(state), "in_battle": self._in_battle(state)}
        return obs, float(reward), terminated, truncated, info


def firered_env(state: bytes, start_state: str, reward_weights: Mapping[str, float]) -> gym.Env:
    entry = next(e for e in MANIFEST if e.name == start_state)
    assert entry.step_budget is not None
    backend = MgbaBackend(FIRERED_ROM, render=False)
    reader = FireRedReader(backend.rom.sha1)
    # The start state is the same on every reset, so one check covers them all.
    backend.load_state(state)
    reader.check_options(backend)
    reads = LastRead(reader)
    rewards = RewardRegistry[FireRedState](COMPONENTS, reward_weights)
    env = AlgoArcadeEnv(backend, reads, rewards, state, entry.step_budget, FIRERED_FEATURES)
    return Measures(env, reads, entry.goal, lambda s: s.in_battle)


def fake_env(reward_weights: Mapping[str, float]) -> gym.Env:
    backend = FakeBackend()
    reads = LastRead(fake.FakeReader())
    rewards = RewardRegistry(COMPONENTS, reward_weights)
    env = AlgoArcadeEnv(
        backend, reads, rewards, backend.save_state(), FAKE_STEP_BUDGET, fake.FEATURES
    )
    # The toy game has no battles.
    return Measures(env, reads, lambda s: s.chest_opened, lambda _s: False)


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
    if rom.sha1 != ROM_SHA1:
        raise ValueError(f"{FIRERED_ROM} is {rom.name}; the start states were captured on rev 1")
    return partial(firered_env, state, entry.name, dict(reward_weights)), "mgba", rom.sha1
