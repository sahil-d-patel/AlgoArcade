"""A scripted seg_05 attempt through the LLM loop on the real ROM keeps its curriculum states."""

import json
from pathlib import Path

import pytest

from algoarcade.agent.loop import Run
from algoarcade.games.firered.manifest import STATES
from algoarcade.training.games import FIRERED_ROM

STATE = STATES / "seg_05_forest.state"


@pytest.mark.rom
def test_a_successful_seg_05_attempt_writes_ordered_states_and_a_manifest(tmp_path: Path) -> None:
    if not FIRERED_ROM.is_file() or not STATE.is_file():
        pytest.skip(f"needs {FIRERED_ROM} and {STATE}")
    pytest.importorskip(
        "mgba.core", reason="mGBA binding did not load; see docs/setup.md", exc_type=ImportError
    )
    from scripted_seg05 import FREE, play

    from algoarcade.backends.mgba import MgbaBackend
    from algoarcade.games.firered.reader import FireRedReader

    settings = Run(
        name="scripted",
        segment="seg_05_forest",
        level=2,
        model="scripted",
        prices=FREE,
        max_actions=2_000,
        max_dollars=0.0,
        capture_every=10,
        out=tmp_path / "llm",
        runs=tmp_path / "runs",
        curriculum=tmp_path / "curriculum",
    )
    attempt = play(settings)
    assert attempt.outcome == "goal"
    assert attempt.goal_step is not None
    folder = tmp_path / "curriculum" / "scripted"
    manifest = json.loads((folder / "manifest.json").read_text())
    actions = [entry["action"] for entry in manifest["states"]]
    assert actions == sorted(actions)
    assert len(actions) >= 10
    assert actions[-1] < attempt.goal_step
    # Every kept state is in Viridian Forest (MAP_VIRIDIAN_FOREST), before the north gate.
    assert all(entry["location"][:2] == [1, 0] for entry in manifest["states"])
    backend = MgbaBackend(FIRERED_ROM)
    try:
        reader = FireRedReader(backend.rom.sha1)
        for entry in (manifest["states"][0], manifest["states"][-1]):
            backend.load_state((folder / entry["file"]).read_bytes())
            assert list(reader.read(backend).location) == entry["location"]
    finally:
        backend.close()
