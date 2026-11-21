"""The agent video grid: RL checkpoints and an LLM attempt playing the same start state side by
side, one frame per button press. Contract: docs/technical-plan.md, Visualization and LLM arm.

    uv run python -m algoarcade.viz.video_grid \\
        --checkpoints <run dir>/checkpoints/step_<n>.zip ... --attempt data/llm/E7/<run id> \\
        --out data/videos/grid/<segment>.mp4

Each checkpoint plays seed 0 until its goal or its step budget. The attempt is replayed from its
logged presses. The composer refuses cells whose frame 0 differs, so every cell must start from
the same state.
"""

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import tyro
from pydantic_core import to_jsonable_python

from algoarcade.agent.loop import SEGMENTS
from algoarcade.agent.server import GAMES, start_state
from algoarcade.backends.base import Button
from algoarcade.games.base import StateNotReady
from algoarcade.viz.render import Policy, Shot, compose


class Replay:
    """An LLM attempt played again from data/llm/<experiment>/<run id>/: its attempt.json and
    the server's trace, presses.jsonl, on a fresh emulator from the attempt's start state. Each
    press steps the idle frames the trace logged before it, then presses as the server did, so
    the replay ends where the attempt ended. Raises if the start state's hash or any state after
    a press differs from the trace. After it is played, `goal_step` is the first press after which
    the segment's goal held, graded as the loop grades it."""

    def __init__(self, folder: Path) -> None:
        self.attempt = json.loads((folder / "attempt.json").read_text())
        self.presses = [
            json.loads(line) for line in (folder / "presses.jsonl").read_text().splitlines()
        ]
        self.label = f"{self.attempt['model']} L{self.attempt['level']}"
        self.goal_step: int | None = None

    def __iter__(self) -> Iterator[Shot]:
        segment = SEGMENTS[self.attempt["segment"]]
        game = GAMES[segment.game]()
        backend = game.backend
        try:
            if segment.start_state:
                backend.load_state(start_state(segment.start_state))
            start = hashlib.sha256(backend.save_state()).hexdigest()
            if start != self.presses[0]["start_sha256"]:
                raise ValueError(f"the replay starts from {start}, the attempt did not")
            yield Shot(backend.screen(), (self.label, "press 0", self.attempt["segment"]), {})
            place = ""
            for step, press in enumerate(self.presses[1:], start=1):
                backend.step(press["idle_frames"])
                backend.press([Button[press["button"]]])
                try:
                    state: Any = game.reader.read(backend)
                except StateNotReady:
                    state = None
                else:
                    place = f"{state.location}  badges {state.badge_count}"
                read = to_jsonable_python(state)
                if read != press["state"]:
                    raise ValueError(f"press {step}: the replay read {read}, the trace {press}")
                if self.goal_step is None and read and segment.goal(read):
                    self.goal_step = step
                yield Shot(
                    backend.screen(), (self.label, f"press {step}  {press['button']}", place), {}
                )
        finally:
            backend.close()


def main(checkpoints: list[Path], attempt: Path, out: Path) -> None:
    """Render each checkpoint and the LLM attempt side by side, and print where each reached
    its goal."""
    policies = [Policy(checkpoint, until_goal=True) for checkpoint in checkpoints]
    replay = Replay(attempt)
    compose([*policies, replay], out)
    for cell in [*policies, replay]:
        goal = "no goal" if cell.goal_step is None else f"goal at press {cell.goal_step}"
        print(f"{cell.label}: {goal}")  # noqa: T201
    print(out)  # noqa: T201


if __name__ == "__main__":
    tyro.cli(main)
