"""Writes an LLM attempt in the frozen logging schema, so it joins the RL runs on start_state.

Contract: docs/harness-spec.md, Logging schema. An attempt is a run of one episode. Each press
the server traced is a decision: one per-step row (step_log_every is 1) and one visit. The
model earns no reward, so reward_total and episode_return are 0 and every reward column is null.
"""

from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pyarrow as pa

from algoarcade.backends.base import Button
from algoarcade.logs.schema import EPISODES, REWARD_COMPONENTS, RUNS, STEPS, VALUES, VISITS
from algoarcade.logs.writer import RollingWriter

WRITER = "llm"
NO_REWARDS = {f"reward_{name}": None for name in REWARD_COMPONENTS}
# Counts logged at the episode's end, each the latest value two consecutive reads agreed on, as
# training/games.py does: a read on the frame a save block moves can dip or read high.
COUNTS = ("badge_count", "party_level_sum", "story_flags")


def write_run_row(run_dir: Path, row: dict[str, Any]) -> None:
    """Each row in its own closed file, so an attempt killed before its end row reads running."""
    runs = RollingWriter(run_dir, "runs", WRITER, resume=True)
    runs.write(pa.Table.from_pylist([row], schema=RUNS))
    runs.close()


def place(location: tuple[int, ...]) -> tuple[int, ...]:
    """(map_group, map_num, x, y). The fake game's (room, x, y) logs its room as map_num."""
    return (0,) * (4 - len(location)) + tuple(location)


def write_attempt(  # noqa: PLR0913 the run, its trace and the attempt's totals
    run_dir: Path,
    ids: dict[str, Any],
    presses: list[dict[str, Any]],
    parse: Callable[[dict[str, Any]], Any],
    started: float,
    episode: dict[str, Any],
) -> None:
    """The steps, visits and episodes tables for one attempt. `presses` is the server's trace,
    the start state first; a press whose state was not ready logs the last ready one. `episode`
    holds the episode columns the trace cannot give: the end reason, goal and the model's
    usage."""
    state = parse(presses[0]["state"])
    last = {name: getattr(state, name) for name in COUNTS}
    agreed = dict(last)
    milestones = previous_milestones = state.milestone_count
    tiles = {place(state.location)}
    visits: Counter[tuple[int, ...]] = Counter()
    rows = []
    for step, press in enumerate(presses[1:], start=1):
        if press["state"] is not None:
            state = parse(press["state"])
            for name in COUNTS:
                value = getattr(state, name)
                if value == last[name]:
                    agreed[name] = value
                last[name] = value
            # The best rises only to the lower of two consecutive reads, as in AlgoArcadeEnv.
            milestones = max(milestones, min(previous_milestones, state.milestone_count))
            previous_milestones = state.milestone_count
        location = place(state.location)
        tiles.add(location)
        visits[location] += 1
        map_group, map_num, x, y = location
        rows.append(
            ids
            | {
                "env_idx": 0,
                "env_steps": step,
                "wall_clock_s": press["time"] - started,
                "episode_id": 0,
                "episode_step": step,
                # The Button value, the index into the ten buttons the server takes.
                "action": Button[press["button"]].value,
                "map_group": map_group,
                "map_num": map_num,
                "x": x,
                "y": y,
                "in_battle": getattr(state, "in_battle", False),
                "milestone_count": milestones,
                "reward_total": 0.0,
            }
            | NO_REWARDS
        )
    steps = RollingWriter(run_dir, "steps", WRITER)
    if rows:
        steps.write(pa.Table.from_pylist(rows, schema=STEPS))
    steps.close()
    visit_rows = [
        {"run_id": ids["run_id"], "window_start_step": 0}
        | dict(zip(("map_group", "map_num", "x", "y"), location, strict=True))
        | {"visits": count}
        for location, count in visits.items()
    ]
    if visit_rows:
        table = RollingWriter(run_dir, "visits", WRITER)
        table.write(pa.Table.from_pylist(visit_rows, schema=VISITS))
        table.close()
    end_reason = episode["end_reason"]
    # The frozen end_reason set has no value for a model that stopped calling tools; such an
    # attempt was cut off short of its goal, which is what truncated means for an RL episode.
    if end_reason not in VALUES[("episodes", "end_reason")]:
        end_reason = "truncated"
    row = (
        ids
        | {
            "env_idx": 0,
            "episode_id": 0,
            "env_steps": len(rows),
            "wall_clock_s": presses[-1]["time"] - started,
            "episode_return": 0.0,
            "episode_length": len(rows),
            "unique_tiles": len(tiles),
            "milestone_count": milestones,
            "archive_start": False,
            "is_eval": False,
            "checkpoint_step": None,
        }
        | agreed
        | NO_REWARDS
        | episode
        | {"end_reason": end_reason}
    )
    episodes = RollingWriter(run_dir, "episodes", WRITER)
    episodes.write(pa.Table.from_pylist([row], schema=EPISODES))
    episodes.close()
