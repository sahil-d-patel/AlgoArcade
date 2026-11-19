from pathlib import Path

import pytest

from algoarcade.logs import fake
from algoarcade.viz import render_all
from algoarcade.viz.render_all import final_checkpoint

ENV_STEPS = 1_000


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """Three fake runs: two with checkpoints, one without."""
    runs = tmp_path / "runs"
    for seed, steps in enumerate(([500_000, 1_000_000], [500_000], [])):
        args = ["--run-id", f"fake{seed}", "--seed", str(seed), "--env-steps", str(ENV_STEPS)]
        fake.main([*args, "--root", str(runs)])
        checkpoints = runs / f"run_id=fake{seed}" / "checkpoints"
        checkpoints.mkdir()
        for step in steps:
            (checkpoints / f"step_{step}.zip").touch()
    return runs


@pytest.fixture
def renders(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Swaps the real renderer, which needs a trained policy and the game, for one that writes an
    empty clip and records the checkpoint it was given."""
    calls: list[Path] = []

    def fake_render(checkpoint: Path, out: Path) -> None:
        calls.append(checkpoint)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.touch()

    monkeypatch.setattr(render_all, "render", fake_render)
    return calls


def test_final_checkpoint_compares_steps_as_numbers(root: Path) -> None:
    assert final_checkpoint(root / "run_id=fake0") == (
        root / "run_id=fake0" / "checkpoints" / "step_1000000.zip"
    )
    assert final_checkpoint(root / "run_id=fake2") is None


def test_renders_each_final_checkpoint_once(
    root: Path, tmp_path: Path, renders: list[Path], capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "videos"
    render_all.main(root, out)
    assert capsys.readouterr().out.splitlines()[-1] == "rendered 2, skipped 0, failed 0"
    assert sorted(path.name for path in renders) == ["step_1000000.zip", "step_500000.zip"]
    render_all.main(root, out)
    assert capsys.readouterr().out.splitlines()[-1] == "rendered 0, skipped 2, failed 0"
    assert len(renders) == 2


def test_a_failed_render_is_named_and_leaves_no_clip(
    root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def broken_render(checkpoint: Path, out: Path) -> None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.touch()
        raise RuntimeError("no ROM")

    monkeypatch.setattr(render_all, "render", broken_render)
    out = tmp_path / "videos"
    render_all.main(root, out)
    lines = capsys.readouterr().out.splitlines()
    assert "failed fake0: no ROM" in lines
    assert lines[-1] == "rendered 0, skipped 0, failed 2"
    assert not list(out.glob("*.mp4"))


def test_an_interrupted_render_is_rendered_again(
    root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    renders: list[Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    working = render_all.render

    def interrupted(checkpoint: Path, out: Path) -> None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.touch()
        raise KeyboardInterrupt

    monkeypatch.setattr(render_all, "render", interrupted)
    out = tmp_path / "videos"
    with pytest.raises(KeyboardInterrupt):
        render_all.main(root, out)
    monkeypatch.setattr(render_all, "render", working)
    render_all.main(root, out)
    assert capsys.readouterr().out.splitlines()[-1] == "rendered 2, skipped 0, failed 0"
