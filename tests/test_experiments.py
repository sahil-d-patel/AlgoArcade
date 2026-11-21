"""The experiment files: each arm differs from the shared baseline arm only in its factor."""

import dataclasses
import runpy
from pathlib import Path
from typing import Any

import pytest

from algoarcade.rewards.registry import DEFAULT_WEIGHTS, WEIGHTS
from algoarcade.runqueue.queue import load_experiment, run_name
from algoarcade.training.config import BASELINE_V1, TrainConfig

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
FROM_S1 = dataclasses.replace(BASELINE_V1, start_state="s1", track=True)
# The shared baseline arm's experiment id is "baseline"; algoarcade.viz.comparison reads it.
E1_TO_E5 = ("baseline", "E1", "E2", "E3", "E4", "E5")
LABELS = ("experiment_id", "arm", "seed")
E6 = {"archive": True, "total_timesteps": 100_000_000, "eval_every": 10_000_000}


def test_e3_queues_none_and_rnd_on_the_baseline_seeds() -> None:
    configs = load_experiment(EXPERIMENTS / "e3_rnd.py")
    assert sorted((c.arm, c.seed) for c in configs) == [
        (arm, seed) for arm in ("none", "rnd") for seed in range(5)
    ]
    no_count = {name: w for name, w in DEFAULT_WEIGHTS.items() if name != "explore"}
    for c in configs:
        rnd = c.arm == "rnd"
        assert c == dataclasses.replace(
            FROM_S1,
            reward_weights=no_count,
            second_value_head=rnd,
            int_coef=0.5 if rnd else 0.0,
            int_gamma=0.99,
            experiment_id="E3",
            arm=c.arm,
            seed=c.seed,
        )


def test_e5_lists_the_components_on_in_each_arm_with_three_seeds() -> None:
    configs = load_experiment(EXPERIMENTS / "e5_penalties.py")
    assert sorted((c.arm, c.seed) for c in configs) == [
        (arm, seed) for arm in ("plus-penalties", "positive-only") for seed in range(3)
    ]
    for c in configs:
        weights = DEFAULT_WEIGHTS if c.arm == "positive-only" else WEIGHTS
        assert c == dataclasses.replace(
            FROM_S1, reward_weights=weights, experiment_id="E5", arm=c.arm, seed=c.seed
        )


def changed(config: TrainConfig) -> dict[str, Any]:
    """The fields that differ from FROM_S1, leaving out the queue's labels."""
    return {
        field.name: getattr(config, field.name)
        for field in dataclasses.fields(config)
        if field.name not in LABELS and getattr(config, field.name) != getattr(FROM_S1, field.name)
    }


@pytest.mark.parametrize(
    ("file", "names", "fields"),
    [
        ("baseline.py", [f"baseline-v1-seed{s}" for s in range(5)], {}),
        ("e4_archive.py", [f"E4-on-seed{s}" for s in range(5)], {"archive": True}),
        ("e6_brock_s1.py", [f"E6-s1-seed{s}" for s in range(3)], E6),
        ("e6_brock_s0.py", [f"E6-s0-seed{s}" for s in range(2)], E6 | {"start_state": "s0"}),
    ],
)
def test_an_ops06_experiment_changes_only_its_factor_from_s1(
    file: str, names: list[str], fields: dict[str, Any]
) -> None:
    configs = load_experiment(EXPERIMENTS / file)
    assert [run_name(config) for config in configs] == names
    assert all(changed(config) == fields for config in configs)


def test_every_e1_to_e5_file_runs_on_s1_from_the_same_base() -> None:
    bases = {path.name: runpy.run_path(str(path)) for path in EXPERIMENTS.glob("*.py")}
    s1 = {name for name, module in bases.items() if module["EXPERIMENT"] in E1_TO_E5}
    assert {"baseline.py", "e3_rnd.py", "e4_archive.py", "e5_penalties.py"} <= s1
    assert all(bases[name]["BASE"] == FROM_S1 for name in s1)
    assert {
        config.start_state for name in s1 for config in load_experiment(EXPERIMENTS / name)
    } == {"s1"}


def test_trn08_keeps_the_evaluation_its_runs_were_queued_with() -> None:
    configs = load_experiment(EXPERIMENTS / "trn08_baseline.py")
    assert len(configs) == 15
    for c in configs:
        assert c == dataclasses.replace(
            BASELINE_V1,
            track=True,
            eval_every=1_500_000,
            eval_episodes=20,
            start_state=c.start_state,
            experiment_id="TRN-08",
            arm=c.arm,
            seed=c.seed,
        )
