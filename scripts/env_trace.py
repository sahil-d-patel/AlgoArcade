"""Run AlgoArcadeEnv from one seed and print a hash of the trace, to compare two processes.

    uv run python scripts/env_trace.py fake --out tests/fixtures/env_golden_trace.json
    uv run python scripts/env_trace.py mgba --state data/states/s1.state
    uv run python scripts/env_trace.py pyboy

--seed seeds the resets only. The actions come from random.Random(--action-seed) and are stored
in the --out file, so the trace never depends on numpy's streams, and changing --seed alone
measures what reset noise does. Episodes are 128 steps, so 500 steps cross three resets.
mgba needs --state: at power-on gSaveBlock1Ptr is 0 and the first reset raises StateNotReady.
pyboy without --state starts 120 frames after power-on, once the game's Init has run.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from gymnasium.spaces import Discrete

from algoarcade.backends.fake import FakeBackend
from algoarcade.backends.mgba import MgbaBackend
from algoarcade.backends.pyboy import PyBoyBackend
from algoarcade.envs.env import AlgoArcadeEnv
from algoarcade.games import fake
from algoarcade.games.firered.features import FEATURES as FIRERED_FEATURES
from algoarcade.games.firered.reader import FireRedReader
from algoarcade.games.super_mario_land.reader import SuperMarioLandReader
from algoarcade.rewards.explore import Explore
from algoarcade.rewards.registry import COMPONENTS, RewardRegistry

BUDGET = 128
ROMS = Path(__file__).resolve().parents[1] / "data" / "roms"


def fake_env() -> AlgoArcadeEnv:
    backend = FakeBackend()
    rewards = RewardRegistry(COMPONENTS, {"events": 1.0, "explore": 0.1})
    return AlgoArcadeEnv(
        backend, fake.FakeReader(), rewards, backend.save_state(), BUDGET, fake.FEATURES
    )


def mgba_env(state: bytes) -> AlgoArcadeEnv:
    backend = MgbaBackend(ROMS / "firered.gba", render=False)
    rewards = RewardRegistry({"explore": Explore}, {"explore": 1.0})
    reader = FireRedReader(backend.rom.sha1)
    return AlgoArcadeEnv(backend, reader, rewards, state, BUDGET, FIRERED_FEATURES)


def pyboy_env(state: bytes | None) -> AlgoArcadeEnv:
    backend = PyBoyBackend(ROMS / "super_mario_land.gb")
    reader = SuperMarioLandReader()
    rewards = RewardRegistry({"explore": Explore}, {"explore": 1.0})
    # Guessed scales: a three-digit timer, a two-digit BCD life count, a long level.
    features = {
        "time_left": lambda s: s.time_left / 999,
        "lives_left": lambda s: s.lives_left / 99,
        "level_progress": lambda s: s.level_progress / 4096,
    }
    if state is None:
        backend.step(120)  # the reader is not ready until Init sets hWorldAndLevel
        state = backend.save_state()
    return AlgoArcadeEnv(backend, reader, rewards, state, BUDGET, features)


def trace(env: AlgoArcadeEnv, seed: int, actions: list[int]) -> list[list[Any]]:
    """One row per reset and per step: action (None on reset), reward, location, truncated and
    a digest of the observation."""

    def row(action: int | None, reward: float, obs: Any, info: Any, truncated: bool) -> list[Any]:
        # An integer, not hex: the typos hook reads hex digits as misspelled words.
        digest = hashlib.sha1(obs["vector"].tobytes() + obs["visited"].tobytes()).digest()
        return [action, reward, list(info["location"]), truncated, int.from_bytes(digest[:6])]

    obs, info = env.reset(seed=seed)
    rows = [row(None, 0.0, obs, info, False)]
    for action in actions:
        obs, reward, _, truncated, info = env.step(action)
        rows.append(row(action, reward, obs, info, truncated))
        if truncated:
            obs, info = env.reset()
            rows.append(row(None, 0.0, obs, info, False))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("backend", choices=("fake", "mgba", "pyboy"))
    parser.add_argument("--state", type=Path, help="start state file")
    parser.add_argument("--seed", type=int, default=0, help="reset seed")
    parser.add_argument("--action-seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--out", type=Path, help="write the full trace here as JSON")
    args = parser.parse_args()
    state = args.state.read_bytes() if args.state else None
    match args.backend:
        case "fake":
            env = fake_env()
        case "mgba" if state:
            env = mgba_env(state)
        case "mgba":
            parser.error("mgba needs --state: the power-on state is not ready")
        case _:
            env = pyboy_env(state)
    assert isinstance(env.action_space, Discrete)
    actions = random.Random(args.action_seed).choices(range(int(env.action_space.n)), k=args.steps)
    rows = trace(env, args.seed, actions)
    end_state = hashlib.sha256(env.backend.save_state()).hexdigest()
    env.close()
    text = json.dumps(
        {"seed": args.seed, "action_seed": args.action_seed, "actions": actions, "trace": rows}
    )
    if args.out:
        args.out.write_text(text + "\n")
    steps = [r for r in rows if r[0] is not None]
    print(f"trace sha256 {hashlib.sha256(text.encode()).hexdigest()}")
    print(f"end state sha256 {end_state}")
    print(
        f"steps {len(steps)}, resets {len(rows) - len(steps)}, "
        f"return {sum(r[1] for r in steps):.4f}, "
        f"distinct locations {len({tuple(r[2]) for r in rows})}, "
        f"first {rows[0][2]}, last {rows[-1][2]}"
    )


if __name__ == "__main__":
    main()
