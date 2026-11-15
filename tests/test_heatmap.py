from collections import Counter

import pyarrow as pa
import pytest

from algoarcade.logs.fake import fake_run
from algoarcade.logs.schema import VISITS
from algoarcade.viz.heatmap import counts


def test_known_walk_sums_parallel_rows_and_checkpoint_windows() -> None:
    run = fake_run("walk", seed=0, env_steps=12)
    locations = [(3, 0, 2, 3), (3, 0, 4, 5), (1, 0, 2, 3)]
    rows = []
    for window, tile, visits in [(0, 0, 2), (0, 0, 3), (500_000, 0, 1), (0, 1, 2), (0, 2, 4)]:
        group, num, x, y = locations[tile]
        rows.append(
            {
                "run_id": "walk",
                "window_start_step": window,
                "map_group": group,
                "map_num": num,
                "x": x,
                "y": y,
                "visits": visits,
            }
        )
    assert counts(pa.Table.from_pylist(rows, schema=VISITS), run["runs"]) == Counter(
        dict(zip(locations, [6, 2, 4], strict=True))
    )


def test_fake_log_exact_counts_agree_with_every_step_when_unsampled() -> None:
    run = fake_run("walk", seed=7, env_steps=400, step_log_every=1)
    expected = Counter(
        (r["map_group"], r["map_num"], r["x"], r["y"]) for r in run["steps"].to_pylist()
    )
    assert counts(run["visits"], run["runs"]) == expected
    assert expected.total() == 400


def test_missing_visits_fail_instead_of_drawing_incomplete_counts() -> None:
    run = fake_run("walk", seed=0, env_steps=40)
    with pytest.raises(ValueError, match="run steps"):
        counts(run["visits"].slice(1), run["runs"])


def test_mixed_run_rows_are_rejected() -> None:
    run = fake_run("walk", seed=0, env_steps=40)
    other = fake_run("other", seed=0, env_steps=40)
    with pytest.raises(ValueError, match="another run"):
        counts(other["visits"], run["runs"])
