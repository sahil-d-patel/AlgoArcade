"""Synthetic runs in the logged schema, for building panels before real runs exist.

A seeded random walker drifts from the player's bedroom to Pewter Gym over real FireRed maps,
fights random battles in the grass and sometimes earns the Boulder Badge, the run's goal. The same seed gives
identical tables. Contract: docs/harness-spec.md, Fake backend and fake logger.

Write one run with: uv run python -m algoarcade.logs.fake --run-id fake0 --seed 0
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa

from algoarcade.logs.schema import EPISODES, REWARD_COMPONENTS, RUNS, STEPS, VISITS
from algoarcade.logs.writer import ROWS_PER_FILE, RollingWriter, run_dir

# (name, map_group, map_num, width, height) from pret/pokefirered at commit
# ce6406d8c2eaf46d543c8a2606de0c3cb1951982: data/maps/map_groups.json gives the ids,
# data/layouts/layouts.json the size. Walked in this order.
MAPS = (
    ("PalletTown_PlayersHouse_2F", 4, 1, 12, 9),
    ("PalletTown_PlayersHouse_1F", 4, 0, 13, 10),
    ("PalletTown", 3, 0, 24, 20),
    ("Route1", 3, 19, 24, 40),
    ("ViridianCity", 3, 1, 48, 40),
    ("Route2", 3, 20, 24, 80),
    ("ViridianForest", 1, 0, 54, 69),
    ("PewterCity", 3, 2, 48, 40),
    ("PewterCity_Gym", 6, 2, 13, 16),
)
GRASS = {"Route1", "Route2", "ViridianForest"}
GYM = len(MAPS) - 1
# Up, down, left, right, A, B.
MOVES = ((0, -1), (0, 1), (-1, 0), (1, 0), (0, 0), (0, 0))

# The default components and starting weights in docs/technical-plan.md, Reward design.
WEIGHTS = {"events": 1.0, "explore": 0.02, "level": 0.5, "badge": 10.0, "milestone": 2.0}

NUM_ENVS = 4
STEP_LOG_EVERY = 4
EVAL_EPISODES = 2
EPISODE_STEPS = 2_048
WINDOW_STEPS = 500_000
STEPS_PER_SECOND = 500.0
WARP_CHANCE = 0.05
FORWARD_CHANCE = 0.6
BATTLE_CHANCE = 0.02
BADGE_CHANCE = 0.0005
START_LEVEL_SUM = 5


class _Walker:
    def __init__(self, rng: np.random.Generator) -> None:
        self.episode_id = -1
        self.reset(rng)

    @property
    def location(self) -> tuple[int, int, int, int]:
        _, group, num, _, _ = MAPS[self.map]
        return group, num, self.x, self.y

    @property
    def milestone_count(self) -> int:
        return self.furthest + self.badges

    def reset(self, rng: np.random.Generator) -> None:
        self.episode_id += 1
        self.episode_step = 0
        self.map = 0
        self._place(rng)
        self.battle = 0
        self.level_sum = START_LEVEL_SUM
        self.badges = 0
        self.story_flags = 0
        self.furthest = 0
        self.goal_step: int | None = None
        self.visited = {self.location}
        self.totals = dict.fromkeys(WEIGHTS, 0.0)
        self.episode_return = 0.0

    def _place(self, rng: np.random.Generator) -> None:
        _, _, _, width, height = MAPS[self.map]
        self.x, self.y = int(rng.integers(width)), int(rng.integers(height))

    def step(self, rng: np.random.Generator) -> tuple[int, dict[str, float], float]:
        """Moves one decision. Returns the action, each component's raw value and the reward."""
        self.episode_step += 1
        action = int(rng.integers(len(MOVES)))
        level_sum, badges, milestones = self.level_sum, self.badges, self.milestone_count
        name, _, _, width, height = MAPS[self.map]
        if self.battle:
            self.battle -= 1
            if not self.battle and rng.random() < 0.5:
                self.level_sum += 1
        elif rng.random() < WARP_CHANCE:
            forward = rng.random() < FORWARD_CHANCE
            self.map = min(max(self.map + (1 if forward else -1), 0), GYM)
            self._place(rng)
        else:
            dx, dy = MOVES[action]
            self.x = min(max(self.x + dx, 0), width - 1)
            self.y = min(max(self.y + dy, 0), height - 1)
            if name in GRASS and rng.random() < BATTLE_CHANCE:
                self.battle = int(rng.integers(5, 30))
        if self.map == GYM and not self.badges and rng.random() < BADGE_CHANCE:
            self.badges = 1
            self.goal_step = self.episode_step
        self.furthest = max(self.furthest, self.map)
        new_milestones = self.milestone_count - milestones
        self.story_flags += new_milestones
        new_tile = self.location not in self.visited
        self.visited.add(self.location)
        raw = {
            "events": float(new_milestones),
            "explore": float(new_tile),
            "level": float(min(self.level_sum - level_sum, 1)),
            "badge": float(self.badges - badges),
            "milestone": float(new_milestones),
        }
        for c, value in raw.items():
            self.totals[c] += value
        reward = sum(WEIGHTS[c] * value for c, value in raw.items())
        self.episode_return += reward
        return action, raw, reward


def _rewards(columns: dict[str, list], raw: dict[str, float]) -> None:
    for c in REWARD_COMPONENTS:
        columns[f"reward_{c}"].append(raw.get(c))


def _end_episode(episodes: dict[str, list], w: _Walker, rng: np.random.Generator, **keys) -> None:
    row = {
        **keys,
        "episode_id": w.episode_id,
        "episode_return": w.episode_return,
        "episode_length": w.episode_step,
        "badge_count": w.badges,
        "party_level_sum": w.level_sum,
        "story_flags": w.story_flags,
        "unique_tiles": len(w.visited),
        "milestone_count": w.milestone_count,
        "start_state": "S0",
        "archive_start": False,
        "end_reason": "truncated",
        "goal_reached": w.goal_step is not None,
        "goal_step": w.goal_step,
        **dict.fromkeys(n for n in EPISODES.names if n.startswith("llm_")),
    }
    for name, value in row.items():
        episodes[name].append(value)
    _rewards(episodes, w.totals)
    w.reset(rng)


def fake_run(
    run_id: str, seed: int, env_steps: int, step_log_every: int = STEP_LOG_EVERY
) -> dict[str, pa.Table]:
    """One synthetic run of `env_steps` training decisions across NUM_ENVS walkers, as the four
    tables. At every checkpoint, EVAL_EPISODES evaluation episodes run on a separate walker; they
    write per-episode rows only."""
    rng = np.random.default_rng(seed)
    walkers = [_Walker(rng) for _ in range(NUM_ENVS)]
    evaluator = _Walker(rng)
    steps: dict[str, list] = {name: [] for name in STEPS.names}
    episodes: dict[str, list] = {name: [] for name in EPISODES.names}
    visits: Counter[tuple[int, int, int, int, int]] = Counter()
    total = 0
    for _ in range(env_steps // NUM_ENVS):
        total += NUM_ENVS
        clock = total / STEPS_PER_SECOND
        window = (total - 1) // WINDOW_STEPS * WINDOW_STEPS
        keys = {"run_id": run_id, "seed": seed, "env_steps": total, "wall_clock_s": clock}
        for env_idx, w in enumerate(walkers):
            action, raw, reward = w.step(rng)
            visits[(window, *w.location)] += 1
            if w.episode_step % step_log_every == 0:
                group, num, x, y = w.location
                row = {
                    **keys,
                    "env_idx": env_idx,
                    "episode_id": w.episode_id,
                    "episode_step": w.episode_step,
                    "action": action,
                    "map_group": group,
                    "map_num": num,
                    "x": x,
                    "y": y,
                    "in_battle": w.battle > 0,
                    "milestone_count": w.milestone_count,
                    "reward_total": reward,
                }
                for name, value in row.items():
                    steps[name].append(value)
                _rewards(steps, raw)
            if w.episode_step == EPISODE_STEPS:
                train = {"env_idx": env_idx, "is_eval": False, "checkpoint_step": None}
                _end_episode(episodes, w, rng, **keys, **train)
        if total % WINDOW_STEPS == 0:
            for _ in range(EVAL_EPISODES * EPISODE_STEPS):
                evaluator.step(rng)
                if evaluator.episode_step == EPISODE_STEPS:
                    evaluation = {"env_idx": 0, "is_eval": True, "checkpoint_step": total}
                    _end_episode(episodes, evaluator, rng, **keys, **evaluation)
    weights = json.dumps(WEIGHTS, sort_keys=True)
    start = {
        "run_id": run_id,
        "status": "running",
        "experiment_id": "fake",
        "arm": "fake",
        "algorithm": "random",
        "trainer": "fake",
        "seed": seed,
        "env_id": "fake",
        "backend": "fake",
        "rom_sha1": "none",
        "start_state": "S0",
        "obs_mode": "symbolic",
        "frames_per_decision": 24,
        "num_envs": NUM_ENVS,
        "reward_config_hash": hashlib.sha1(weights.encode()).hexdigest(),
        "git_commit": "none",
        "hyperparameters": json.dumps({"gamma": 0.997, "reward_weights": WEIGHTS}),
        "step_log_every": step_log_every,
        "machine": "fake",
        "env_steps": 0,
        "wall_clock_s": 0.0,
    }
    end = {
        **start,
        "status": "finished",
        "env_steps": total,
        "wall_clock_s": total / STEPS_PER_SECOND,
    }
    tiles = sorted(visits)
    visit_columns = {
        "run_id": [run_id] * len(tiles),
        **{name: [k[i] for k in tiles] for i, name in enumerate(VISITS.names[1:-1])},
        "visits": [visits[k] for k in tiles],
    }
    return {
        "steps": pa.table(steps, schema=STEPS),
        "visits": pa.table(visit_columns, schema=VISITS),
        "episodes": pa.table(episodes, schema=EPISODES),
        "runs": pa.Table.from_pylist([start, end], schema=RUNS),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m algoarcade.logs.fake", description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--env-steps", type=int, default=2_000_000)
    parser.add_argument("--step-log-every", type=int, default=STEP_LOG_EVERY)
    parser.add_argument("--root", type=Path, default=Path("data/runs"))
    args = parser.parse_args(argv)
    run = fake_run(args.run_id, args.seed, args.env_steps, args.step_log_every)
    for table, batch in run.items():
        # One file per runs row: a real run writes its start row at launch, its end row at exit.
        rows_per_file = 1 if table == "runs" else ROWS_PER_FILE
        writer = RollingWriter(run_dir(args.run_id, args.root), table, "fake", rows_per_file)
        for chunk in batch.to_batches(max_chunksize=rows_per_file):
            writer.write(pa.Table.from_batches([chunk]))
        writer.close()


if __name__ == "__main__":
    main()
