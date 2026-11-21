import json
from pathlib import Path

import pytest

from algoarcade.logs import fake
from algoarcade.rewards.registry import WEIGHTS
from algoarcade.viz import decomposition, site

# Two checkpoint windows of training episodes.
ENV_STEPS = decomposition.WINDOW_STEPS + 20_000
ARMS = {"positive-only": fake.WEIGHTS, "plus-penalties": WEIGHTS}


@pytest.fixture(scope="module")
def root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    runs = tmp_path_factory.mktemp("runs")
    for arm, weights in ARMS.items():
        fake.main(
            [
                *("--run-id", arm, "--seed", "0", "--env-steps", str(ENV_STEPS)),
                *("--experiment-id", "E5", "--arm", arm, "--weights", json.dumps(weights)),
                *("--root", str(runs)),
            ]
        )
    return runs


@pytest.mark.parametrize("arm", ARMS)
def test_each_windows_components_sum_to_the_mean_episode_return(root: Path, arm: str) -> None:
    starts, returns, parts = decomposition.decompose(site.connect(root), [arm])
    assert starts == [0, decomposition.WINDOW_STEPS]
    assert list(parts) == list(ARMS[arm])
    for i, mean_return in enumerate(returns):
        assert sum(values[i] for values in parts.values()) == pytest.approx(mean_return, abs=1e-6)
    assert decomposition.largest_gap(site.connect(root), arm) <= 1e-6


def test_penalties_stack_below_zero(root: Path) -> None:
    _, _, parts = decomposition.decompose(site.connect(root), ["plus-penalties"])
    assert all(value < 0 for value in parts["time"])
    assert all(value > 0 for value in parts["explore"])


def test_figure_has_a_panel_per_arm_and_a_bar_per_component(root: Path) -> None:
    arms = {arm: ("E5", arm) for arm in ARMS}
    fig = decomposition.figure(site.connect(root), arms).to_dict()
    assert fig["layout"]["barmode"] == "relative"
    bars = [trace for trace in fig["data"] if trace["type"] == "bar"]
    assert len(bars) == len(fake.WEIGHTS) + len(WEIGHTS)
    titles = [note["text"] for note in fig["layout"]["annotations"]]
    assert titles == ["positive-only (n=1)", "plus-penalties (n=1)"]


def test_an_arm_with_no_runs_gets_an_empty_panel(root: Path) -> None:
    fig = decomposition.figure(site.connect(root), {"baseline": ("E3", "count")}).to_dict()
    assert fig["data"] == []
    assert fig["layout"]["annotations"][0]["text"] == "baseline (n=0)"
