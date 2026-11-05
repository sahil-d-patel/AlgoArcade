"""Super Mario Land in the runner: continues after a game over, the start state and the env."""

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from algoarcade.backends.base import Button, read_u8
from algoarcade.envs.env import AlgoArcadeEnv
from algoarcade.games.super_mario_land import manifest
from algoarcade.games.super_mario_land.reader import MENU, SuperMarioLandReader
from algoarcade.training.config import PRESETS
from algoarcade.training.games import SML_ROM, Continues, env_factory

ROOT = Path(__file__).resolve().parents[2]
CONFIG = PRESETS["super_mario_land"][1]


class Memory:
    """Just enough of PyBoyBackend for Continues: one menu byte and a record of loads."""

    def __init__(self) -> None:
        self.menu = 0
        self.loaded: list[bytes] = []

    def step(self, frames: int = 1) -> None:
        pass

    def press(self, buttons: object, hold: int = 8, release: int = 16) -> None:
        pass

    def read(self, address: int, size: int) -> bytes:
        assert (address, size) == (MENU, 1)
        return bytes([self.menu])

    def load_state(self, state: bytes) -> None:
        self.loaded.append(state)
        self.menu = 0


def test_continues_loads_the_start_state_only_when_the_menu_is_up() -> None:
    memory = Memory()
    backend = Continues(memory, b"start")  # pyrefly: ignore[bad-argument-type]
    backend.press([Button.RIGHT])
    backend.step(1)
    assert memory.loaded == []
    memory.menu = 0x28
    backend.press([Button.RIGHT])
    assert memory.loaded == [b"start"]
    memory.menu = 0x28
    backend.step(1)
    assert memory.loaded == [b"start", b"start"]


def needs_rom() -> None:
    if not SML_ROM.is_file():
        pytest.skip(f"no ROM at {SML_ROM}")


@pytest.mark.rom
def test_capture_reproduces_the_manifest_state(tmp_path: Path) -> None:
    needs_rom()
    script = ROOT / "scripts" / "capture_sml_states.py"
    subprocess.run([sys.executable, str(script), "--out", str(tmp_path)], check=True)
    (entry,) = manifest.MANIFEST
    data = (tmp_path / f"{entry.name}.state").read_bytes()
    assert hashlib.sha256(data).hexdigest() == entry.sha256


@pytest.mark.rom
def test_the_preset_plays_through_a_game_over_without_paying_for_it() -> None:
    needs_rom()
    if not (manifest.STATES / f"{CONFIG.start_state}.state").is_file():
        pytest.skip("no level_1_1.state; run scripts/capture_sml_states.py")
    env_fn, backend, rom_sha1 = env_factory(
        CONFIG.env_id, CONFIG.start_state, CONFIG.reward_weights
    )
    assert (backend, rom_sha1) == ("pyboy", manifest.ROM_SHA1)
    env = env_fn()
    env.reset(seed=0)
    unwrapped = env.unwrapped
    assert isinstance(unwrapped, AlgoArcadeEnv)
    reader = SuperMarioLandReader()
    # Holding right walks into the first enemy: three lives end within about 80 decisions.
    lives, rewards, continues = [], [], 0
    for _ in range(150):
        _, reward, terminated, truncated, info = env.step(3)
        assert not terminated
        assert not truncated
        assert read_u8(unwrapped.backend, MENU) == 0
        assert info["badge_count"] == info["milestone_count"] == 0
        lives.append(reader.read(unwrapped.backend).lives_left)
        if lives[-2:] == [0, 2]:
            continues += 1
            assert reward == 0.0
        rewards.append(reward)
    env.close()
    assert continues >= 1
    assert sum(rewards) > 0
