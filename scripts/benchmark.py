"""Throughput benchmark (OPS-02). Prints the docs/benchmarks.md tables as Markdown on stdout.

    uv run python scripts/benchmark.py --backend mgba --rom data/roms/firered.gba \
        --state data/states/s1.state --menu-state data/states/menu_open.state

Every cell runs one emulator per spawned process, loaded from --state, under a uniform random
policy for --seconds, and reports the median of --runs runs. Progress goes to stderr.
"""

import argparse
import datetime
import multiprocessing as mp
import platform
import queue
import random
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from multiprocessing.synchronize import Barrier
from pathlib import Path

import numpy as np

from algoarcade.backends import fake
from algoarcade.backends.base import Button, GameBackend

# Frames per decision -> (hold, release). 24 is the project default; the others are OPS-02's.
SPLITS = {8: (4, 4), 16: (8, 8), 24: (8, 16), 48: (8, 40)}
DEFAULT_FRAMES = 24
ENV_COUNTS = (4, 8, 12, 16)
ACTIONS = (Button.UP, Button.DOWN, Button.LEFT, Button.RIGHT, Button.A, Button.B)
MENU_PRESSES = 20
GBA_RAM = ((0x0200_0000, 0x4_0000), (0x0300_0000, 0x8000))  # EWRAM, IWRAM


@dataclass(frozen=True)
class Cell:
    backend: str
    rom: Path | None
    frames: int
    render: bool
    read_state: bool
    seconds: float
    state: bytes | None = field(repr=False)


def make_backend(name: str, rom: Path | None, render: bool) -> tuple[GameBackend, tuple]:
    if name == "fake":
        return fake.FakeBackend(), ((fake.RAM_BASE, fake.RAM_SIZE),)
    from algoarcade.backends.mgba import MgbaBackend  # arrives with ENV-04

    return MgbaBackend(rom, render=render), GBA_RAM


def decide(backend: GameBackend, regions: tuple, cell: Cell, rng: random.Random) -> None:
    hold, release = SPLITS[cell.frames]
    backend.press([rng.choice(ACTIONS)], hold, release)
    if cell.render:
        backend.screen()
    if cell.read_state:
        for address, size in regions:
            backend.read(address, size)


def worker(cell: Cell, seed: int, barrier: Barrier, results: mp.Queue) -> None:
    backend, regions = make_backend(cell.backend, cell.rom, cell.render)
    if cell.state is not None:
        backend.load_state(cell.state)
    rng = random.Random(seed)
    decide(backend, regions, cell, rng)
    barrier.wait()
    steps = 0
    start = now = time.perf_counter()
    while now - start < cell.seconds:
        decide(backend, regions, cell, rng)
        steps += 1
        now = time.perf_counter()
    results.put(steps / (now - start))
    backend.close()


def steps_per_second(cell: Cell, envs: int) -> float:
    ctx = mp.get_context("spawn")
    barrier, results = ctx.Barrier(envs), ctx.Queue()
    procs = [ctx.Process(target=worker, args=(cell, i, barrier, results)) for i in range(envs)]
    for p in procs:
        p.start()
    # Poll so a worker that dies before the barrier does not leave the others waiting on it.
    deadline = time.monotonic() + cell.seconds + 120
    rates = []
    while len(rates) < envs:
        try:
            rates.append(results.get(timeout=1))
        except queue.Empty:
            if any(p.exitcode for p in procs) or time.monotonic() > deadline:
                for p in procs:
                    p.terminate()
                sys.exit(f"a worker failed or stalled during {cell}, envs={envs}; see stderr above")
    for p in procs:
        p.join()
    return sum(rates)


def median_rate(cell: Cell, envs: int, runs: int) -> float:
    rates = []
    for run in range(runs):
        rates.append(steps_per_second(cell, envs))
        print(f"  {cell} envs={envs} run {run + 1}/{runs}: {rates[-1]:,.0f}", file=sys.stderr)
    return statistics.median(rates)


def menu_check(args: argparse.Namespace, state: bytes, frames: int) -> int:
    """Presses DOWN 20 times from the menu state and counts presses that changed the screen.

    Each press is compared with a no-input step from the same saved state, so animation does not
    count as a move.
    """
    hold, release = SPLITS[frames]
    backend, _ = make_backend(args.backend, args.rom, render=True)
    backend.load_state(state)
    moved = 0
    for _ in range(MENU_PRESSES):
        before = backend.save_state()
        backend.step(hold + release)
        idle = backend.screen()
        backend.load_state(before)
        backend.press([Button.DOWN], hold, release)
        moved += not np.array_equal(idle, backend.screen())
    backend.close()
    return moved


def header(state: Path | None) -> list[str]:
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout
    return [
        f"Command: `uv run python {' '.join(sys.argv)}`",
        f"Commit: `{commit}`{' (uncommitted changes)' if dirty.strip() else ''}",
        f"OS: {platform.platform()}, {platform.machine()}, {mp.cpu_count()} logical CPUs",
        f"Python: {platform.python_version()}",
        f"Date: {datetime.datetime.now(datetime.UTC):%Y-%m-%d %H:%M} UTC",
        f"Start state: `{state}`" if state else "Start state: ROM boot",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", choices=["fake", "mgba"], default="mgba")
    parser.add_argument("--rom", type=Path)
    parser.add_argument(
        "--state", type=Path, help="save state every worker loads first (default: ROM boot)"
    )
    parser.add_argument(
        "--menu-state",
        type=Path,
        help="save state with the start menu open; a menu that does not wrap caps the count",
    )
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    if args.backend != "fake" and args.rom is None:
        parser.error("--rom is required for a real backend")
    start = args.state.read_bytes() if args.state else None
    menu = args.menu_state.read_bytes() if args.menu_state else None

    cells = (4 + len(SPLITS) * len(ENV_COUNTS)) * args.runs
    print(f"about {cells * args.seconds / 60:.0f} minutes of measurement", file=sys.stderr)
    lines = [*header(args.state), ""]

    lines += [
        f"### Emulator frames per second, one process, {DEFAULT_FRAMES} frames per decision",
        "",
        "| Render | No state read | State read |",
        "|---|---|---|",
    ]
    for render in (False, True):
        fps = [
            median_rate(
                Cell(args.backend, args.rom, DEFAULT_FRAMES, render, read, args.seconds, start),
                1,
                args.runs,
            )
            * DEFAULT_FRAMES
            for read in (False, True)
        ]
        lines.append(f"| {'on' if render else 'off'} | {fps[0]:,.0f} | {fps[1]:,.0f} |")

    lines += [
        "",
        "### Agent steps per second, random policy, render off, state read on",
        "",
        "| Frames per decision | Hold / release | "
        + " | ".join(f"{n} envs" for n in ENV_COUNTS)
        + " |",
        "|---|---|" + "---|" * len(ENV_COUNTS),
    ]
    for frames, (hold, release) in SPLITS.items():
        cell = Cell(args.backend, args.rom, frames, False, True, args.seconds, start)
        rates = [median_rate(cell, n, args.runs) for n in ENV_COUNTS]
        lines.append(
            f"| {frames} | {hold} / {release} | " + " | ".join(f"{r:,.0f}" for r in rates) + " |"
        )

    lines += [
        "",
        f"### Menu presses that moved the cursor, out of {MENU_PRESSES}",
        "",
    ]
    if menu is None:
        lines.append("Not run: no --menu-state given.")
    else:
        lines += [
            "| Frames per decision | Hold / release | Moved |",
            "|---|---|---|",
        ]
        for frames, (hold, release) in SPLITS.items():
            moved = menu_check(args, menu, frames)
            lines.append(f"| {frames} | {hold} / {release} | {moved} |")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
