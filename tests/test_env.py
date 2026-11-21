import dataclasses
import json
import zlib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from env_trace import fake_env, trace
from gymnasium.spaces import Box, Dict, Discrete
from gymnasium.utils.env_checker import check_env
from numpy.typing import NDArray

from algoarcade.backends import fake
from algoarcade.backends.base import Button, GameBackend
from algoarcade.backends.fake import FakeBackend
from algoarcade.envs.env import CROP, NOT_READY_FRAMES, VISITED, WALKABLE, AlgoArcadeEnv, Cell
from algoarcade.games.base import StateNotReady
from algoarcade.games.fake import FEATURES, FakeReader, FakeState, walkable
from algoarcade.rewards.registry import COMPONENTS, WEIGHTS, RewardRegistry

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


def make_env(backend: FakeBackend | None = None, budget: int = 50, **kwargs: Any) -> AlgoArcadeEnv:
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


@pytest.mark.parametrize("obs_mode", ["symbolic", "pixels", "both"])
def test_env_checker_passes_on_fake_backend(obs_mode: str) -> None:
    check_env(make_env(obs_mode=obs_mode, walkable=walkable), skip_render_check=True)


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


def test_a_longer_hold_keeps_the_step_at_24_frames() -> None:
    backend = RecordingBackend()
    env = make_env(backend, hold=16)
    env.reset(seed=0)
    backend.frames.clear()
    env.step(4)
    assert backend.frames == [(Button.A,)] * 16 + [()] * 8


def test_noop_action_holds_nothing() -> None:
    backend = RecordingBackend()
    env = make_env(backend, noop=True)
    env.reset(seed=0)
    backend.frames.clear()
    env.step(6)
    assert backend.frames == [()] * 24


def test_info_components_sum_to_the_reward() -> None:
    weights = {"events": 2.0, "explore": 0.1}
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
        assert reward == sum(w * info[f"reward/{name}"] for name, w in weights.items())
        paid += reward
    assert paid > 1


def test_info_carries_location_and_milestone_count() -> None:
    env = make_env()
    _, info = env.reset(seed=0)
    assert info == {"location": (0, 1, 1), "milestone_count": 0, "archive_start": False}
    _, _, _, _, info = env.step(3)
    assert info["location"] == (0, 2, 1)
    assert info["milestone_count"] == 0


def test_info_keeps_the_episode_best_milestone_count() -> None:
    backend = FakeBackend()
    env = make_env(backend)
    env.reset(seed=0)
    counts = []
    for value in (3, 3, 1, 0, 4, 4):
        ram = bytearray(backend.save_state())
        ram[fake.MILESTONES] = value
        backend.load_state(bytes(ram))
        counts.append(env.step(3)[4]["milestone_count"])
    assert counts == [0, 3, 3, 3, 3, 4]
    _, info = env.reset(seed=0)
    assert info["milestone_count"] == 0


def test_reset_pays_nothing_and_clears_the_episode() -> None:
    env = make_env()
    env.reset(seed=0)
    for action in (3, 3, 1):
        env.step(action)
    obs, info = env.reset(seed=0)
    assert not any(key.startswith("reward/") for key in info)
    assert obs["grid"][VISITED].sum() == 1
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


def test_symbolic_grid_has_a_visited_and_a_walkable_channel_centred_on_the_player() -> None:
    env = make_env(walkable=walkable)
    obs, _ = env.reset(seed=0)
    rows, cols = CROP
    assert set(obs) == {"vector", "grid"}
    assert obs["grid"].shape == (2, rows, cols)
    # The player starts at (1, 1) in room 0: the wall tile above it, the floor tile right of it.
    assert obs["grid"][WALKABLE, rows // 2 - 1, cols // 2] == 0.0
    assert obs["grid"][WALKABLE, rows // 2, cols // 2 + 1] == 1.0
    assert obs["grid"][VISITED, rows // 2, cols // 2] == 1.0
    obs, *_ = env.step(3)
    assert obs["grid"][VISITED, rows // 2, cols // 2] == 1.0
    assert obs["grid"][VISITED, rows // 2, cols // 2 - 1] == 1.0
    assert obs["grid"][VISITED].sum() == 2


def test_without_map_data_every_tile_reads_walkable() -> None:
    obs, _ = make_env().reset(seed=0)
    assert obs["grid"][WALKABLE].min() == 1.0


def test_an_unknown_obs_mode_is_refused() -> None:
    with pytest.raises(ValueError, match="pixel"):
        make_env(obs_mode="pixel")


def test_pixels_stack_three_greyscale_frames_beside_the_symbolic_vector() -> None:
    symbolic, pixels = make_env(walkable=walkable), make_env(obs_mode="pixels")
    assert isinstance(pixels.observation_space, Dict)
    assert pixels.observation_space["pixels"] == Box(0, 255, (3, 72, 80), np.uint8)
    expected, _ = symbolic.reset(seed=3)
    obs, _ = pixels.reset(seed=3)
    assert set(obs) == {"vector", "pixels"}
    assert obs["pixels"].shape == (3, 72, 80)
    assert obs["pixels"].dtype == np.uint8
    assert (obs["pixels"] == obs["pixels"][2]).all()
    np.testing.assert_array_equal(obs["vector"], expected["vector"])
    for action in (3, 3, 1, 4, 2, 0, 5, 3):
        previous = obs["pixels"]
        expected, *_ = symbolic.step(action)
        obs, *_ = pixels.step(action)
        np.testing.assert_array_equal(obs["vector"], expected["vector"])
        np.testing.assert_array_equal(obs["pixels"][:2], previous[1:])
    assert not (obs["pixels"][2] == previous[2]).all()


def test_both_is_the_union_of_symbolic_and_pixels() -> None:
    env = make_env(obs_mode="both", walkable=walkable)
    obs, _ = env.reset(seed=0)
    symbolic, _ = make_env(walkable=walkable).reset(seed=0)
    pixels, _ = make_env(obs_mode="pixels").reset(seed=0)
    assert set(obs) == {"vector", "grid", "pixels"}
    for key, value in (symbolic | pixels).items():
        np.testing.assert_array_equal(obs[key], value)


class ScreenBackend(FakeBackend):
    """Shows a fixed GBA-sized frame."""

    def __init__(self, image: NDArray[np.uint8]) -> None:
        super().__init__()
        self.image = image

    def screen(self) -> NDArray[np.uint8]:
        return self.image.copy()


def test_a_240_by_160_frame_averages_3_columns_and_about_2_2_rows_into_one_pixel() -> None:
    image = np.zeros((160, 240, 3), np.uint8)
    image[:80, ::3] = 90
    obs, _ = make_env(ScreenBackend(image), obs_mode="pixels").reset(seed=0)
    # 80 of 160 rows are 72 * 80 / 160 = 36 of 72 output rows exactly.
    assert (obs["pixels"][:, :36] == 30).all()
    assert (obs["pixels"][:, 36:] == 0).all()


def test_greyscale_weights_red_green_and_blue_as_bt_601() -> None:
    image = np.zeros((160, 240, 3), np.uint8)
    image[:] = (200, 100, 50)
    obs, _ = make_env(ScreenBackend(image), obs_mode="pixels").reset(seed=0)
    assert (obs["pixels"] == round(0.299 * 200 + 0.587 * 100 + 0.114 * 50)).all()


def test_vector_is_scaled_into_the_unit_box() -> None:
    env = make_env()
    obs, _ = env.reset(seed=0)
    assert obs["vector"].shape == (len(FEATURES),)
    assert obs["vector"].min() >= 0.0
    assert obs["vector"].max() <= 1.0
    assert obs["vector"][list(FEATURES).index("x")] == np.float32(1 / (fake.WIDTH - 1))


class FlakyReader(FakeReader):
    """Raises StateNotReady on the reads listed in `fail`, counted from 0."""

    def __init__(self, fail: Iterable[int]) -> None:
        self.fail = set(fail)
        self.reads = 0

    def read(self, backend: GameBackend) -> FakeState:
        self.reads += 1
        if self.reads - 1 in self.fail:
            raise StateNotReady("stub")
        return super().read(backend)


def flaky_env(reader: FlakyReader) -> AlgoArcadeEnv:
    backend = FakeBackend()
    return AlgoArcadeEnv(
        backend,
        reader,
        RewardRegistry(COMPONENTS, {"events": 1.0, "explore": 0.1}),
        backend.save_state(),
        50,
        FEATURES,
    )


def test_a_not_ready_read_is_retried_one_frame_later() -> None:
    steady = make_env()
    steady.reset(seed=0)
    steady.step(0)
    flaky = flaky_env(FlakyReader(fail={0, 2}))
    flaky.reset(seed=0)
    flaky.step(0)
    assert frame(flaky) == frame(steady) + 2


def test_a_read_not_ready_on_every_retry_raises() -> None:
    env = flaky_env(FlakyReader(fail=range(1, 2 + NOT_READY_FRAMES)))
    env.reset(seed=0)
    with pytest.raises(StateNotReady):
        env.step(0)


def test_retries_stop_at_the_first_ready_frame() -> None:
    reader = FlakyReader(fail=range(1, 1 + NOT_READY_FRAMES))
    env = flaky_env(reader)
    env.reset(seed=0)
    env.step(0)
    assert reader.reads == 2 + NOT_READY_FRAMES


def test_a_one_read_milestone_spike_is_not_kept() -> None:
    backend = FakeBackend()
    env = make_env(backend)
    env.reset(seed=0)
    counts = []
    for value in (0, 5, 0, 0):
        ram = bytearray(backend.save_state())
        ram[fake.MILESTONES] = value
        backend.load_state(bytes(ram))
        counts.append(env.step(3)[4]["milestone_count"])
    assert counts == [0, 0, 0, 0]


def start_with_milestones(value: int) -> tuple[FakeBackend, AlgoArcadeEnv]:
    backend = FakeBackend()
    ram = bytearray(backend.save_state())
    ram[fake.MILESTONES] = value
    backend.load_state(bytes(ram))
    return backend, make_env(backend)


def test_a_reset_read_counts_once_the_first_step_agrees() -> None:
    _, env = start_with_milestones(2)
    _, info = env.reset(seed=0)
    assert info["milestone_count"] == 0
    assert env.step(3)[4]["milestone_count"] == 2


def test_a_reset_read_spike_is_not_kept() -> None:
    backend, env = start_with_milestones(5)
    _, info = env.reset(seed=0)
    ram = bytearray(backend.save_state())
    ram[fake.MILESTONES] = 0
    backend.load_state(bytes(ram))
    counts = [info["milestone_count"]] + [env.step(3)[4]["milestone_count"] for _ in range(3)]
    assert counts == [0, 0, 0, 0]


def poke(backend: FakeBackend, offset: int, value: int) -> None:
    ram = bytearray(backend.save_state())
    ram[offset] = value
    backend.load_state(bytes(ram))


def walk_right(env: AlgoArcadeEnv, steps: int) -> list[Cell | None]:
    return [env.step(3)[4].get("cell") for _ in range(steps)]


def test_a_step_reports_a_cell_the_first_time_the_episode_reaches_it() -> None:
    backend = FakeBackend()
    env = make_env(backend, archive=True)
    env.reset(seed=0)
    cells = walk_right(env, 4)
    # From (1, 1): x 2 and 3 are in cell x // 4 == 0, x 4 and 5 in cell 1.
    assert [cell and cell.key for cell in cells] == [(0, 0, 0, 0, 0), None, (0, 1, 0, 0, 0), None]
    cell = cells[2]
    assert cell is not None
    assert cell.visited == {(0, x, 1) for x in range(1, 5)}
    backend.load_state(zlib.decompress(cell.state))
    assert FakeReader().read(backend).location == (0, 4, 1)
    env.reset(seed=0)
    assert [cell and cell.key for cell in walk_right(env, 1)] == [(0, 0, 0, 0, 0)]


def test_without_archive_no_cell_is_reported_and_a_next_start_is_ignored() -> None:
    archived = make_env(archive=True)
    archived.reset(seed=0)
    cell = walk_right(archived, 3)[-1]
    assert cell is not None
    env = make_env()
    env.reset(seed=0)
    assert walk_right(env, 4) == [None] * 4
    env.set_next_start(cell)
    assert env.reset(seed=0)[1] == {
        "location": (0, 1, 1),
        "milestone_count": 0,
        "archive_start": False,
    }


def test_an_archive_reset_and_a_step_onto_a_visited_tile_pay_nothing() -> None:
    backend = FakeBackend()
    env = AlgoArcadeEnv(
        backend,
        FakeReader(),
        RewardRegistry(COMPONENTS, WEIGHTS),
        backend.save_state(),
        50,
        FEATURES,
        archive=True,
    )
    env.reset(seed=0)
    # Progress the archived state holds, so a progress term that paid it again would show.
    for offset, value in (
        (fake.FLAGS, fake.FLAG_CHEST_OPENED),
        (fake.LEVEL_SUM, 9),
        (fake.BADGES, 1),
        (fake.MILESTONES, 3),
        (fake.HP_PERCENT, 50),
    ):
        poke(backend, offset, value)
    cell = walk_right(env, 3)[-1]
    assert cell is not None
    assert cell.key == (0, 1, 0, 1, 3)

    env.set_next_start(cell)
    _, info = env.reset(seed=1)
    assert info["location"] == (0, 4, 1)
    assert info["archive_start"] is True
    # B does nothing in the fake; left goes to (3, 1), visited on the way to the cell.
    for action in (5, 2):
        _, _, _, _, info = env.step(action)
        raw = {name: info[f"reward/{name}"] for name in WEIGHTS}
        # time charges every step by design (docs/harness-spec.md, Reward registry).
        assert raw == dict.fromkeys(WEIGHTS, 0.0) | {"time": 1.0}
    assert info["location"] == (0, 3, 1)


def test_a_reset_into_a_cell_reached_on_a_rising_step_pays_nothing() -> None:
    # The level sum rises on the step that reaches the cell. Its term confirms a rise one step
    # late, so the cell's bests must already include it or every reset into the cell repays it.
    backend = FakeBackend()
    env = AlgoArcadeEnv(
        backend,
        FakeReader(),
        RewardRegistry(COMPONENTS, WEIGHTS),
        backend.save_state(),
        50,
        FEATURES,
        archive=True,
    )
    env.reset(seed=0)
    walk_right(env, 2)
    poke(backend, fake.LEVEL_SUM, 9)
    cell = walk_right(env, 1)[-1]
    assert cell is not None
    assert cell.key == (0, 1, 0, 0, 0)

    env.set_next_start(cell)
    env.reset(seed=1)
    _, _, _, _, info = env.step(5)
    raw = {name: info[f"reward/{name}"] for name in WEIGHTS}
    assert raw == dict.fromkeys(WEIGHTS, 0.0) | {"time": 1.0}


def test_a_cell_reached_on_a_one_frame_high_read_keeps_the_real_best() -> None:
    # A count that reads far too high for one frame is refused by its term's max_rise guard. The
    # cell reached on that step must not store the bad read as its best, or a reset into the
    # cell would silence the term until the count passed it.
    backend = FakeBackend()
    env = AlgoArcadeEnv(
        backend,
        FakeReader(),
        RewardRegistry(COMPONENTS, WEIGHTS),
        backend.save_state(),
        50,
        FEATURES,
        archive=True,
    )
    env.reset(seed=0)
    first = walk_right(env, 2)[0]
    assert first is not None
    # Five badges at once: the badge term refuses a rise of more than one between reads.
    poke(backend, fake.BADGES, 0b11111)
    cell = walk_right(env, 1)[-1]
    assert cell is not None
    assert cell.bests["badge"] == first.bests["badge"]


COUNTS = (fake.FLAGS, fake.LEVEL_SUM, fake.BADGES, fake.MILESTONES)


def test_a_reload_does_not_repay_counts_that_dropped_before_the_cell() -> None:
    # A FireRed count can fall below the episode's best: forest_exited and gym_entered stop
    # holding off their maps, and depositing a Pokemon lowers the level sum. A cell reached after
    # the drop must not let a reload pay the climb back to the best.
    high = (0b11, 9, 0b11, 4)
    low = (0b01, 7, 0b01, 2)
    backend = FakeBackend()
    for offset, value in zip(COUNTS, high, strict=True):
        poke(backend, offset, value)
    env = AlgoArcadeEnv(
        backend,
        FakeReader(),
        RewardRegistry(COMPONENTS, WEIGHTS),
        backend.save_state(),
        50,
        FEATURES,
        archive=True,
    )
    env.reset(seed=0)
    walk_right(env, 1)
    for offset, value in zip(COUNTS, low, strict=True):
        poke(backend, offset, value)
    cell = walk_right(env, 2)[-1]
    assert cell is not None
    assert cell.key == (0, 1, 0, 1, 4)

    for episode in range(3):
        env.set_next_start(cell)
        _, info = env.reset(seed=episode)
        assert info["milestone_count"] == 4
        for offset, value in zip(COUNTS, high, strict=True):
            poke(backend, offset, value)
        for action in (5, 5, 5):
            _, reward, _, _, info = env.step(action)
            assert reward == WEIGHTS["time"]
        assert info["milestone_count"] == 4


class TornReader(FakeReader):
    """Read number `torn_at` reports `badge_count` badges; the RAM, and so the save state, stay real."""

    def __init__(self, torn_at: int, badge_count: int) -> None:
        self.reads, self.torn_at, self.badge_count = 0, torn_at, badge_count

    def read(self, backend: GameBackend) -> FakeState:
        self.reads += 1
        state = super().read(backend)
        if self.reads != self.torn_at:
            return state
        return dataclasses.replace(state, badge_count=self.badge_count)


def test_one_torn_read_on_the_step_that_reaches_a_cell_is_not_its_best() -> None:
    backend = FakeBackend()
    # Read 1 is the reset, reads 2 to 4 are three steps right; step 3 reaches the cell at x 4.
    env = AlgoArcadeEnv(
        backend,
        TornReader(torn_at=4, badge_count=1),
        RewardRegistry(COMPONENTS, WEIGHTS),
        backend.save_state(),
        50,
        FEATURES,
        archive=True,
    )
    env.reset(seed=0)
    cell = walk_right(env, 3)[-1]
    assert cell is not None
    assert cell.key == (0, 1, 0, 0, 0)
    assert cell.bests["badge"] == 0

    env.set_next_start(cell)
    env.reset(seed=1)
    env.step(5)
    poke(backend, fake.BADGES, 0b1)
    assert sum(env.step(5)[4]["reward/badge"] for _ in range(3)) == 1.0
