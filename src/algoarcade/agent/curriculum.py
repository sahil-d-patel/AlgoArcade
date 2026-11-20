"""The states a successful LLM attempt passed through, kept as E8's curriculum.

Contract: docs/harness-spec.md, Curriculum resets. The server saves a state every few presses
(Game.capture); this keeps the ones saved before the goal, with manifest.json:

    {"start_state", "start_sha256", "goal_step", "capture_every",
     "states": [{"file", "action", "location", "sha256", "visited", "bests", "milestone_count"}]}

`action` is the press the state was saved after, counted as the loop counts `goal_step`.
`visited`, `bests` and `milestone_count` are what an AlgoArcadeEnv episode from the start state
would hold after the same reads, so a reset into the state follows the archive-reset rules: the
path to it and progress up to its bests pay nothing again. They come from the trace's ready
reads; a press whose read was not ready is skipped. `bests` also holds a rise the saved read
shows that the next read would have confirmed.
"""

import copy
import hashlib
import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from algoarcade.rewards.registry import COMPONENTS, WEIGHTS, RewardRegistry


def keep(  # noqa: PLR0913 the trace, both folders and the attempt's facts
    presses: list[dict[str, Any]],
    captured: Path,
    out: Path,
    goal_step: int,
    parse: Callable[[dict[str, Any]], Any],
    start_state: str,
    capture_every: int,
) -> None:
    """Move the states in `captured` saved before `goal_step` to `out`, with the manifest."""
    first = parse(presses[0]["state"])
    rewards = RewardRegistry(COMPONENTS, WEIGHTS)
    rewards.reset(first)
    visited = {first.location}
    previous = first
    milestone_count = 0
    states: list[dict[str, Any]] = []
    for action, press in enumerate(presses[1:goal_step], start=1):
        if press["state"] is None:
            continue
        state = parse(press["state"])
        rewards.step(previous, state)
        visited.add(state.location)
        # AlgoArcadeEnv's latch: the best of the lower of two consecutive reads.
        milestone_count = max(milestone_count, min(previous.milestone_count, state.milestone_count))
        previous = state
        if "saved" in press:
            data = (captured / press["saved"]).read_bytes()
            # A rise this read shows is confirmed one step late, so a reset into the state
            # would pay it on its first step, every time. The best includes it.
            confirmed = copy.deepcopy(rewards)
            confirmed.step(state, state)
            states.append(
                {
                    "file": press["saved"],
                    "action": action,
                    "location": state.location,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "visited": sorted(visited),
                    "bests": confirmed.bests(),
                    "milestone_count": milestone_count,
                }
            )
    out.mkdir(parents=True)
    for entry in states:
        shutil.move(captured / entry["file"], out / entry["file"])
    manifest = {
        "start_state": start_state,
        "start_sha256": presses[0]["start_sha256"],
        "goal_step": goal_step,
        "capture_every": capture_every,
        "states": states,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
