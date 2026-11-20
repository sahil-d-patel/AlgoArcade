from pathlib import Path

import imageio_ffmpeg
import pytest

from algoarcade.logs import fake
from algoarcade.viz import site

# Enough for each of the fake logger's walkers to finish one episode.
ENV_STEPS = fake.NUM_ENVS * fake.EPISODE_STEPS + 1_000


@pytest.fixture
def root(tmp_path: Path) -> Path:
    runs = tmp_path / "runs"
    for seed in range(2):
        args = ["--run-id", f"fake{seed}", "--seed", str(seed), "--env-steps", str(ENV_STEPS)]
        fake.main([*args, "--root", str(runs)])
    return runs


def test_connect_reads_every_table(root: Path) -> None:
    con = site.connect(root)
    assert con.sql("SELECT count(DISTINCT run_id) FROM runs").fetchall() == [(2,)]
    for table in ("steps", "visits", "episodes"):
        assert con.sql(f"SELECT DISTINCT run_id FROM {table} ORDER BY 1").fetchall() == [
            ("fake0",),
            ("fake1",),
        ]


def test_returns_figure_has_one_trace_per_run(root: Path) -> None:
    con = site.connect(root)
    traces = site.returns_figure(con).to_dict()["data"]
    assert [trace["name"] for trace in traces] == ["fake0", "fake1"]
    [(episodes,)] = con.sql("SELECT count(*) FROM episodes WHERE NOT is_eval").fetchall()
    assert sum(len(trace["x"]) for trace in traces) == episodes


def test_show_leaves_plotly_to_the_page_header(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    site.show(site.returns_figure(site.connect(root)))
    out = capsys.readouterr().out
    assert out.startswith("```{=html}\n")
    assert out.endswith("\n```\n")
    assert "Plotly.newPlot" in out
    assert "plotly.min.js" not in out
    assert "cdn.plot.ly" not in out


def test_walk_clip_has_a_frame_per_logged_step(root: Path, tmp_path: Path) -> None:
    con = site.connect(root)
    out = tmp_path / "walk.mp4"
    site.walk_clip(con, out)
    frames, _ = imageio_ffmpeg.count_frames_and_secs(str(out))
    assert frames == fake.EPISODE_STEPS // fake.STEP_LOG_EVERY


def test_walk_clip_skips_a_run_whose_steps_are_still_open(root: Path, tmp_path: Path) -> None:
    # A run copied mid-training has its runs row but no closed steps file.
    fake.main(
        ["--run-id", "a-live", "--seed", "2", "--env-steps", str(ENV_STEPS), "--root", str(root)]
    )
    for path in (root / "run_id=a-live" / "steps").glob("*.parquet"):
        path.rename(path.with_name(path.name + ".inprogress"))
    out = tmp_path / "walk.mp4"
    site.walk_clip(site.connect(root), out)
    frames, _ = imageio_ffmpeg.count_frames_and_secs(str(out))
    assert frames == fake.EPISODE_STEPS // fake.STEP_LOG_EVERY


def test_main_writes_the_footer_and_the_assets(root: Path, tmp_path: Path) -> None:
    out = tmp_path / "dashboard"
    out.mkdir()
    site.main(root, out)
    variables = (out / "_variables.yml").read_text().splitlines()
    assert variables[2] == "runs: 2"
    assert (out / "plotly.min.js").stat().st_size > 1_000_000
    assert (out / "videos" / "walk.mp4").exists()


def test_main_refuses_an_empty_runs_folder(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="No runs under"):
        site.main(tmp_path, tmp_path)
