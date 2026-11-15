from base64 import b64decode
from collections import Counter

import numpy as np
import pyarrow as pa
import pytest

from algoarcade.games.firered.maps import MapGrid
from algoarcade.logs.fake import fake_run
from algoarcade.logs.schema import VISITS
from algoarcade.viz.heatmap import counts, interactive, windows


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


@pytest.fixture
def checkpoint_run() -> dict[str, pa.Table]:
    run = fake_run("windows", seed=0, env_steps=4)
    metadata = run["runs"].to_pylist()
    metadata[-1]["env_steps"] = 1_000_040
    run["runs"] = pa.Table.from_pylist(metadata, schema=run["runs"].schema)
    run["visits"] = pa.Table.from_pylist(
        [
            {
                "run_id": "windows",
                "window_start_step": start,
                "map_group": 3,
                "map_num": 0,
                "x": x,
                "y": 0,
                "visits": visits,
            }
            for start, x, visits in [
                (0, 0, 200_000),
                (0, 0, 300_000),
                (500_000, 1, 500_000),
                (1_000_000, 0, 40),
            ]
        ],
        schema=VISITS,
    )
    return run


def test_windows_keep_boundaries_duplicates_and_partial_final_window(checkpoint_run: dict) -> None:
    frames = windows(checkpoint_run["visits"], checkpoint_run["runs"])
    assert [(start, end, batch.total()) for start, end, batch in frames] == [
        (0, 500_000, 500_000),
        (500_000, 1_000_000, 500_000),
        (1_000_000, 1_000_040, 40),
    ]
    assert frames[0][2] == {(3, 0, 0, 0): 500_000}
    assert frames[1][2] == {(3, 0, 1, 0): 500_000}
    assert frames[2][2] == {(3, 0, 0, 0): 40}


def test_invalid_window_totals_fail_even_if_the_run_total_matches(checkpoint_run: dict) -> None:
    rows = checkpoint_run["visits"].to_pylist()
    rows[0]["visits"] -= 1
    rows[2]["visits"] += 1
    with pytest.raises(ValueError, match="checkpoint window 0"):
        windows(pa.Table.from_pylist(rows, schema=VISITS), checkpoint_run["runs"])


def test_long_run_groups_adjacent_windows_without_losing_visits(checkpoint_run: dict) -> None:
    rows = [
        dict(checkpoint_run["visits"].to_pylist()[0], window_start_step=i * 500_000, visits=500_000)
        for i in range(51)
    ]
    metadata = checkpoint_run["runs"].to_pylist()
    metadata[-1]["env_steps"] = 51 * 500_000
    frames = windows(
        pa.Table.from_pylist(rows, schema=VISITS),
        pa.Table.from_pylist(metadata, schema=checkpoint_run["runs"].schema),
    )
    assert len(frames) == 26
    assert frames[0][:2] == (0, 1_000_000)
    assert frames[-1][:2] == (25_000_000, 25_500_000)
    assert sum(batch.total() for _, _, batch in frames) == 25_500_000


def test_slider_clears_previous_tiles_and_keeps_one_color_scale(checkpoint_run: dict) -> None:
    maps = {
        key: MapGrid(
            "MAP_TEST", np.zeros((2, 2), dtype=np.uint16), np.zeros((2, 2), dtype=np.uint16), ()
        )
        for key in [(3, 0), (1, 0), (6, 2)]
    }
    figure = interactive(checkpoint_run["visits"], checkpoint_run["runs"], maps).to_dict()
    assert len(figure["frames"]) == 3
    second = figure["frames"][1]["data"][0]
    z = np.frombuffer(b64decode(second["z"]["bdata"]), dtype=second["z"]["dtype"]).reshape(2, 2)
    raw = second["customdata"]
    values = np.frombuffer(b64decode(raw["bdata"]), dtype=raw["dtype"]).reshape(2, 2)
    assert np.isnan(z[0, 0])
    assert values[0, 1] == 500_000
    assert figure["layout"]["coloraxis"]["cmax"] == pytest.approx(np.log10(500_000))
    assert all("coloraxis" not in frame["layout"] for frame in figure["frames"])
    assert "1,000,001-1,000,040" in figure["frames"][-1]["layout"]["title"]["text"]
