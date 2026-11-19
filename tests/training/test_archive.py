"""Archive resets on the fake game (ENV-09)."""

from collections import Counter
from functools import partial

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")

from stable_baselines3.common.vec_env import DummyVecEnv

from algoarcade.backends.fake import FakeBackend
from algoarcade.envs.env import CROP, AlgoArcadeEnv, Cell
from algoarcade.games.fake import FEATURES, FakeReader
from algoarcade.rewards.registry import COMPONENTS, DEFAULT_WEIGHTS, RewardRegistry
from algoarcade.training.archive import Archive, ArchiveResets

START = (0, 1, 1)


def fake_env(budget: int, archive: bool = True) -> AlgoArcadeEnv:
    backend = FakeBackend()
    return AlgoArcadeEnv(
        backend,
        FakeReader(),
        RewardRegistry(COMPONENTS, DEFAULT_WEIGHTS),
        backend.save_state(),
        budget,
        FEATURES,
        archive=archive,
    )


def reach_cell() -> Cell:
    """The cell at (4, 1), reached by walking right from the start."""
    env = fake_env(50)
    env.reset(seed=0)
    cell = [env.step(3)[4].get("cell") for _ in range(3)][-1]
    assert cell is not None
    return cell


def resets(p: float, archive: bool, n: int) -> tuple[ArchiveResets, list[dict]]:
    """n resets of a one-step episode, with one cell in the archive from the start."""
    venv = ArchiveResets(DummyVecEnv([partial(fake_env, 1, archive)]), p, seed=0)
    venv.archive.add(reach_cell())
    venv.seed(0)
    venv.reset()
    infos = [venv.reset_infos[0]]
    for _ in range(n - 1):
        venv.step(np.array([5]))
        infos.append(venv.reset_infos[0])
    return venv, infos


def test_with_p_1_and_one_cell_a_reset_lands_on_the_cell_with_its_visited_set() -> None:
    cell = reach_cell()
    venv = ArchiveResets(DummyVecEnv([partial(fake_env, 50)]), 1.0, seed=0)
    venv.archive.add(cell)
    obs = venv.reset()
    assert venv.reset_infos[0]["location"] == (0, 4, 1)
    assert venv.reset_infos[0]["archive_start"] is True
    rows, cols = CROP
    seen = np.nonzero(obs["visited"][0, 0])
    visited = {
        (0, 4 + col - cols // 2, 1 + row - rows // 2) for row, col in zip(*seen, strict=True)
    }
    assert visited == cell.visited


@pytest.mark.parametrize(("p", "archive"), [(0.0, True), (1.0, False)], ids=["p0", "eval"])
def test_200_resets_load_the_fixed_start_at_p_0_and_in_evaluation(p: float, archive: bool) -> None:
    venv, infos = resets(p, archive, 200)
    assert len(infos) == 200
    assert all(info["location"] == START and not info["archive_start"] for info in infos)
    assert venv.share == 0.0


def test_draws_are_weighted_one_over_root_one_plus_selections() -> None:
    archive = Archive()
    for x, selections in enumerate((0, 3, 8)):
        cell = Cell((x,), b"", frozenset(), {}, 0)
        archive.add(cell)
        archive.selections[cell.key] = selections
    rng = np.random.default_rng(0)
    draws = Counter(archive.sample(rng).key for _ in range(20_000))
    for key, expected in (((0,), 0.545), ((1,), 0.273), ((2,), 0.182)):
        assert abs(draws[key] / 20_000 - expected) < 0.02


def test_at_p_half_the_share_over_1000_resets_is_near_half() -> None:
    venv, infos = resets(0.5, True, 1000)
    assert venv.starts == 1000
    assert 0.45 <= venv.share <= 0.55
    assert venv.share == sum(info["archive_start"] for info in infos) / 1000


def test_a_cell_keeps_the_first_state_that_reached_it() -> None:
    archive = Archive()
    archive.add(Cell((0,), b"first", frozenset(), {}, 0))
    archive.add(Cell((0,), b"second", frozenset(), {}, 0))
    assert archive.cells[(0,)].state == b"first"


def test_a_key_the_archive_holds_is_not_sent_again() -> None:
    # Two envs walk the same row in lockstep at p = 0, three 4-step episodes each. Both find each
    # cell on the same step, so both send it then; no later episode sends it again.
    venv = ArchiveResets(DummyVecEnv([partial(fake_env, 4), partial(fake_env, 4)]), 0.0, seed=0)
    venv.seed(0)
    venv.reset()
    sent = []
    for _ in range(12):
        _, _, _, infos = venv.step(np.array([3, 3]))
        sent.append([info["cell"].key for info in infos if "cell" in info])
    assert sent[:4] == [[(0, 0, 0, 0, 0)] * 2, [], [(0, 1, 0, 0, 0)] * 2, []]
    assert sent[4:] == [[]] * 8


def test_p_outside_0_to_1_raises() -> None:
    with pytest.raises(ValueError, match="archive_p"):
        ArchiveResets(DummyVecEnv([partial(fake_env, 1)]), 1.5, seed=0)
