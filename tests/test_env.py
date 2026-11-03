import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from env_trace import fake_env, trace
from gymnasium.spaces import Discrete
from gymnasium.utils.env_checker import check_env

from algoarcade.backends import fake
from algoarcade.backends.base import Button
from algoarcade.backends.fake import FakeBackend
from algoarcade.envs.env import CROP, AlgoArcadeEnv
from algoarcade.games.fake import FEATURES, FakeReader
from algoarcade.rewards.registry import COMPONENTS, RewardRegistry

GOLDEN_TRACE = Path(__file__).parent / "fixtures" / "env_golden_trace.json"


class RecordingBackend(FakeBackend):
    """Records the buttons held on every frame."""

    def __init__(self) -> None:
        super().__init__()
        self.frames: list[tuple[Button, ...]] = []

    def step(self, frames: int = 1) -> None:
        self.frames += [()] * frames
        super().step(frames)

    def press(self, buttons: Iterable[Button], hold: int = 8, release: int = 16) -> None:
        held = tuple(buttons)
        recorded = len(self.frames)
        super().press(held, hold, release)
        # FakeBackend.press advances through step(), which logged every frame as empty.
        self.frames[recorded:] = [held] * hold + [()] * release


def make_env(backend: FakeBackend | None = None, budget: int = 50, **kwargs: bool) -> AlgoArcadeEnv:
    backend = backend or FakeBackend()
    return AlgoArcadeEnv(
        backend,
        FakeReader(),
        RewardRegistry(COMPONENTS, {"events": 1.0, "explore": 0.1}),
        backend.save_state(),
        budget,
        FEATURES,
        **kwargs,
    )


def frame(env: AlgoArcadeEnv) -> int:
    return FakeReader().read(env.backend).frame


def test_env_checker_passes_on_fake_backend() -> None:
    check_env(make_env(), skip_render_check=True)


def test_env_checker_passes_with_start_and_noop() -> None:
    env = make_env(start=True, noop=True)
    assert env.action_space == Discrete(8)
    check_env(env, skip_render_check=True)


def test_golden_trace_reproduces_seed_0() -> None:
    stored = json.loads(GOLDEN_TRACE.read_text())
    assert stored["seed"] == 0
    assert len(stored["actions"]) == 500
    assert trace(fake_env(), stored["seed"], stored["actions"]) == stored["trace"]


def test_truncated_only_on_the_budgets_last_step_and_never_terminated() -> None:
    env = make_env(budget=30)
    rng = np.random.default_rng(0)
    for episode in range(3):
        env.reset(seed=episode)
        for step in range(1, 31):
            _, _, terminated, truncated, _ = env.step(rng.integers(6))
            assert terminated is False
            assert truncated is (step == 30)


def test_one_step_is_24_frames_with_the_button_held_for_8() -> None:
    backend = RecordingBackend()
    env = make_env(backend)
    env.reset(seed=0)
    backend.frames.clear()
    before = frame(env)
    env.step(3)
    assert frame(env) - before == 24
    assert backend.frames == [(Button.RIGHT,)] * 8 + [()] * 16


def test_noop_action_holds_nothing() -> None:
    backend = RecordingBackend()
    env = make_env(backend, noop=True)
    env.reset(seed=0)
    backend.frames.clear()
    env.step(6)
    assert backend.frames == [()] * 24


def test_info_components_sum_to_the_reward() -> None:
    weights = {"events": 1.0, "explore": 1.0}
    backend = FakeBackend()
    env = AlgoArcadeEnv(
        backend,
        FakeReader(),
        RewardRegistry(COMPONENTS, weights),
        backend.save_state(),
        200,
        FEATURES,
    )
    env.reset(seed=0)
    rng = np.random.default_rng(1)
    paid = 0.0
    for _ in range(200):
        _, reward, _, _, info = env.step(rng.integers(6))
        components = {k: v for k, v in info.items() if k.startswith("reward/")}
        assert set(components) == {"reward/events", "reward/explore"}
        assert reward == sum(components.values())
        paid += reward
    assert paid > 1


def test_info_carries_location_and_milestone_count() -> None:
    env = make_env()
    _, info = env.reset(seed=0)
    assert info == {"location": (0, 1, 1), "milestone_count": 0}
    _, _, _, _, info = env.step(3)
    assert info["location"] == (0, 2, 1)
    assert info["milestone_count"] == 0


def test_reset_pays_nothing_and_clears_the_episode() -> None:
    env = make_env()
    env.reset(seed=0)
    for action in (3, 3, 1):
        env.step(action)
    obs, info = env.reset(seed=0)
    assert not any(key.startswith("reward/") for key in info)
    assert obs["visited"].sum() == 1
    _, reward, _, _, _ = env.step(2)
    assert reward == 0.0


def test_ten_reset_seeds_give_at_least_two_noop_counts() -> None:
    env = make_env()
    start = frame(env)
    counts = set()
    for seed in range(10):
        env.reset(seed=seed)
        counts.add(frame(env) - start)
    assert len(counts) >= 2
    assert counts <= set(range(8))


def test_same_seed_gives_the_same_noop_count() -> None:
    env = make_env()
    start = frame(env)
    env.reset(seed=4)
    first = frame(env) - start
    env.reset(seed=4)
    assert frame(env) - start == first


def test_visited_crop_is_centred_on_the_player() -> None:
    env = make_env()
    obs, _ = env.reset(seed=0)
    rows, cols = CROP
    centre = (0, rows // 2, cols // 2)
    assert obs["visited"].shape == (1, rows, cols)
    assert obs["visited"][centre] == 1.0
    obs, *_ = env.step(3)
    assert obs["visited"][centre] == 1.0
    assert obs["visited"][0, rows // 2, cols // 2 - 1] == 1.0
    assert obs["visited"].sum() == 2


def test_vector_is_scaled_into_the_unit_box() -> None:
    env = make_env()
    obs, _ = env.reset(seed=0)
    assert obs["vector"].shape == (len(FEATURES),)
    assert obs["vector"].min() >= 0.0
    assert obs["vector"].max() <= 1.0
    assert obs["vector"][list(FEATURES).index("x")] == np.float32(1 / (fake.WIDTH - 1))
