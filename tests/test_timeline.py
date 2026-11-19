from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from algoarcade.logs.fake import fake_run
from algoarcade.logs.schema import STEPS
from algoarcade.viz.timeline import COLUMNS, first_observed, main, timeline


@pytest.fixture
def run() -> dict:
    return fake_run("test", seed=0, env_steps=40)["runs"].to_pylist()[-1]


def samples(rows: list[tuple[int, int, int, int, int]]) -> pa.Table:
    return pa.Table.from_pylist(
        [
            {
                "run_id": "test",
                "seed": 0,
                "env_idx": env,
                "episode_id": episode,
                "episode_step": local,
                "env_steps": step,
                "milestone_count": stage,
            }
            for step, env, episode, local, stage in rows
        ],
        schema=pa.schema([STEPS.field(name) for name in COLUMNS]),
    )


def test_first_recorded_thresholds_do_not_invent_exact_events_or_reset_progress(run: dict) -> None:
    steps = samples([(40, 0, 1, 4, 8), (32, 1, 0, 8, 6), (16, 0, 0, 4, 4), (16, 1, 0, 4, 0)])
    observed = first_observed(steps, run)
    assert observed == {
        1: (16, True),
        2: (16, True),
        3: (16, True),
        4: (16, True),
        5: (32, False),
        6: (32, False),
        7: (40, True),
        8: (40, True),
    }
    assert 9 not in observed


def test_parallel_samples_and_file_order_do_not_change_first_observation(run: dict) -> None:
    rows = [(16, 0, 0, 4, 0), (32, 0, 0, 8, 2), (32, 1, 0, 4, 4), (40, 0, 0, 10, 1)]
    forward = first_observed(samples(rows), run)
    assert forward == first_observed(samples(list(reversed(rows))), run)
    assert forward[1] == (32, False)
    assert forward[3] == (32, True)


def test_no_sampled_progress_stays_missing_instead_of_becoming_step_zero(run: dict) -> None:
    assert first_observed(samples([]), run) == {}
    assert first_observed(samples([(16, 0, 0, 4, 0)]), run) == {}


@pytest.mark.parametrize(("field", "value"), [("seed", 1), ("run_id", "other")])
def test_mixed_run_or_seed_is_rejected(run: dict, field: str, value: object) -> None:
    row = samples([(16, 0, 0, 4, 1)]).to_pylist()[0]
    row[field] = value
    with pytest.raises(ValueError, match="another run or seed"):
        first_observed(pa.Table.from_pylist([row], schema=samples([]).schema), run)


@pytest.mark.parametrize("row", [(41, 0, 0, 4, 1), (16, 0, 0, 4, 10), (16, 4, 0, 4, 1)])
def test_invalid_steps_and_stage_values_are_rejected(run: dict, row: tuple) -> None:
    with pytest.raises(ValueError, match=r"outside the completed run|Invalid environment"):
        first_observed(samples([row]), run)


def test_chart_groups_tied_stages_and_retains_a_row_with_no_observations(run: dict) -> None:
    other = run | {"seed": 1, "run_id": "other"}
    figure = timeline(
        [(run, {1: (16, True), 2: (16, True), 3: (32, False)}), (other, {})]
    ).to_dict()
    assert figure["layout"]["yaxis"]["categoryarray"] == ["Seed 0", "Seed 1"]
    initial = next(
        trace for trace in figure["data"] if trace.get("name") == "Present at first episode sample"
    )
    assert initial["x"] == [16]
    assert "Starter chosen" in initial["text"][0]
    assert "Rival beaten" in initial["text"][0]
    assert "not exact event-completion times" in figure["layout"]["title"]["text"]


def test_chart_refuses_duplicate_seeds_and_different_start_states(run: dict) -> None:
    with pytest.raises(ValueError, match="one run per seed"):
        timeline([(run, {}), (run, {})])
    with pytest.raises(ValueError, match="start state"):
        timeline([(run, {}), (run | {"seed": 1, "start_state": "s1"}, {})])


def test_cli_exports_an_empty_sampled_run_without_claiming_progress(tmp_path: Path) -> None:
    run = fake_run("empty", seed=0, env_steps=0)
    (tmp_path / "runs").mkdir()
    pq.write_table(run["runs"], tmp_path / "runs" / "part.parquet")
    out = tmp_path / "timeline.html"
    main([tmp_path], out)
    content = out.read_text(encoding="utf-8")
    assert "Synthetic progress" in content
    assert "Seed 0" in content
    assert '<script src="https://' not in content
