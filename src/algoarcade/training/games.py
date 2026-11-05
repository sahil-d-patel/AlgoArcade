"""The AlgoArcade games the runner trains on, by env_id: firered on mGBA, super_mario_land on
PyBoy, and fake for tests.

The parent process checks the ROM and the start state, then hands each SubprocVecEnv worker a
factory that builds its own emulator. A worker that raises reaches the parent only as EOFError,
so anything that can be checked before the workers start is checked here.
"""

import hashlib
from collections.abc import Callable, Iterable, Mapping
from functools import partial
from pathlib import Path
from typing import Any

import gymnasium as gym

from algoarcade.backends.base import Button, GameBackend, read_u8
from algoarcade.backends.fake import FakeBackend
from algoarcade.backends.mgba import MgbaBackend
from algoarcade.backends.pyboy import PyBoyBackend
from algoarcade.envs.env import AlgoArcadeEnv
from algoarcade.games import fake
from algoarcade.games.base import GameState, StateReader
from algoarcade.games.firered import manifest as firered_manifest
from algoarcade.games.firered.features import FEATURES as FIRERED_FEATURES
from algoarcade.games.firered.reader import FireRedReader, FireRedState
from algoarcade.games.super_mario_land import manifest as sml_manifest
from algoarcade.games.super_mario_land import rewards as sml_rewards
from algoarcade.games.super_mario_land.features import FEATURES as SML_FEATURES
from algoarcade.games.super_mario_land.reader import (
    MENU,
    SuperMarioLandReader,
    SuperMarioLandState,
)
from algoarcade.rewards.registry import COMPONENTS, RewardRegistry
from algoarcade.rom import KNOWN_ROMS, identify

ROMS = Path(__file__).resolve().parents[3] / "data" / "roms"
FIRERED_ROM = ROMS / "firered.gba"
SML_ROM = ROMS / "super_mario_land.gb"
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
    """Adds to each step's info what the logs need and the env does not report: `name:
    measure(state)` for each measure, and each named count as the latest value two consecutive
    reads agreed on. Until two reads agree on a count, the reset read stands."""

    def __init__(
        self,
        env: AlgoArcadeEnv,
        reads: LastRead,
        measures: Mapping[str, Callable[[Any], Any]],
        counts: tuple[str, ...] = COUNTS,
    ) -> None:
        super().__init__(env)
        self._reads = reads
        self._measures = measures
        self._counts = counts
        self._last: dict[str, int] = {}
        self._agreed: dict[str, int] = {}

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        self._last = {name: getattr(self._reads.state, name) for name in self._counts}
        self._agreed = dict(self._last)
        return obs, info

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        state = self._reads.state
        for name in self._counts:
            value = getattr(state, name)
            if value == self._last[name]:
                self._agreed[name] = value
            self._last[name] = value
        info |= self._agreed | {name: measure(state) for name, measure in self._measures.items()}
        return obs, float(reward), terminated, truncated, info


class Continues:
    """Super Mario Land with endless continues. After a game over the game returns to its title
    menu, which plays a demo of a level on its own after about 27 decisions; explore would pay
    for the demo's tiles. AlgoArcadeEnv ends an episode only on its step budget, so whenever the
    menu flag is set after a press or a step, this loads the start state again before anything
    reads memory. A continue costs the time it took to lose three lives and pays nothing: the
    start tiles are already visited."""

    def __init__(self, backend: PyBoyBackend, start_state: bytes) -> None:
        self._backend = backend
        self._start_state = start_state

    def step(self, frames: int = 1) -> None:
        self._backend.step(frames)
        self._continue()

    def press(self, buttons: Iterable[Button], hold: int = 8, release: int = 16) -> None:
        self._backend.press(buttons, hold, release)
        self._continue()

    def _continue(self) -> None:
        if read_u8(self._backend, MENU):
            self._backend.load_state(self._start_state)

    def screen(self) -> Any:
        return self._backend.screen()

    def read(self, address: int, size: int) -> bytes:
        return self._backend.read(address, size)

    def save_state(self) -> bytes:
        return self._backend.save_state()

    def load_state(self, state: bytes) -> None:
        self._backend.load_state(state)

    def close(self) -> None:
        self._backend.close()


def firered_env(state: bytes, start_state: str, reward_weights: Mapping[str, float]) -> gym.Env:
    entry = next(e for e in firered_manifest.MANIFEST if e.name == start_state)
    assert entry.step_budget is not None
    backend = MgbaBackend(FIRERED_ROM, render=False)
    reader = FireRedReader(backend.rom.sha1)
    # The start state is the same on every reset, so one check covers them all.
    backend.load_state(state)
    reader.check_options(backend)
    reads = LastRead(reader)
    rewards = RewardRegistry[FireRedState](COMPONENTS, reward_weights)
    env = AlgoArcadeEnv(backend, reads, rewards, state, entry.step_budget, FIRERED_FEATURES)
    return Measures(env, reads, {"goal": entry.goal, "in_battle": lambda s: s.in_battle})


def super_mario_land_env(
    state: bytes, start_state: str, reward_weights: Mapping[str, float]
) -> gym.Env:
    entry = next(e for e in sml_manifest.MANIFEST if e.name == start_state)
    backend = Continues(PyBoyBackend(SML_ROM, render=False), state)
    reads = LastRead(SuperMarioLandReader())
    rewards = RewardRegistry[SuperMarioLandState](sml_rewards.COMPONENTS, reward_weights)
    env = AlgoArcadeEnv(backend, reads, rewards, state, entry.step_budget, SML_FEATURES)
    # The game has no battles, badges, party, story flags or milestones: each count is 0.
    nothing = {
        name: lambda _s: 0
        for name in ("badge_count", "party_level_sum", "story_flags", "milestone_count")
    }
    return Measures(
        env, reads, {"goal": entry.goal, "in_battle": lambda _s: False} | nothing, counts=()
    )


def fake_env(reward_weights: Mapping[str, float]) -> gym.Env:
    backend = FakeBackend()
    reads = LastRead(fake.FakeReader())
    rewards = RewardRegistry(COMPONENTS, reward_weights)
    env = AlgoArcadeEnv(
        backend, reads, rewards, backend.save_state(), FAKE_STEP_BUDGET, fake.FEATURES
    )
    # The toy game has no battles.
    return Measures(env, reads, {"goal": lambda s: s.chest_opened, "in_battle": lambda _s: False})


def env_factory(
    env_id: str, start_state: str, reward_weights: Mapping[str, float]
) -> tuple[Callable[[], gym.Env], str, str]:
    """The worker factory, the backend name and the ROM SHA-1 for a run."""
    if env_id == "fake":
        return partial(fake_env, dict(reward_weights)), "fake", ""
    manifest, rom_path, make, backend = {
        "firered": (firered_manifest, FIRERED_ROM, firered_env, "mgba"),
        "super_mario_land": (sml_manifest, SML_ROM, super_mario_land_env, "pyboy"),
    }[env_id]
    entry = next((e for e in manifest.MANIFEST if e.name == start_state), None)
    if entry is None or entry.step_budget is None:
        usable = [e.name for e in manifest.MANIFEST if e.step_budget is not None]
        raise ValueError(f"start_state {start_state!r}: use one of {usable}")
    path = manifest.STATES / f"{entry.name}.state"
    state = path.read_bytes()
    if hashlib.sha256(state).hexdigest() != entry.sha256:
        raise ValueError(f"{path} does not match the manifest hash")
    rom = identify(rom_path)
    if rom.sha1 != manifest.ROM_SHA1:
        expected = KNOWN_ROMS[manifest.ROM_SHA1].name
        raise ValueError(f"{rom_path} is {rom.name}; the start states were captured on {expected}")
    return partial(make, state, entry.name, dict(reward_weights)), backend, rom.sha1
