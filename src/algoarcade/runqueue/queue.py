"""The experiment queue for the training box. Contract: docs/technical-plan.md, Run queue.

    uv run python -m algoarcade.runqueue submit <experiment.py> [--dry-run]
    uv run python -m algoarcade.runqueue run
    uv run python -m algoarcade.runqueue status

An experiment file sets EXPERIMENT (its id), BASE (a TrainConfig), ARMS (arm name to a dict of
TrainConfig fields to replace) and SEEDS. It queues one run per arm and seed, seed by seed, so a
queue cut short leaves every arm with the same seeds.

The state lives in data/runs/queue/: queue.json (every run, in submit order), configs/<run>.json
(the resolved config, fixed at submit) and logs/<run>-<attempt>.log (the subprocess's output).
"""

import argparse
import dataclasses
import importlib.util
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from algoarcade.training.config import TrainConfig, save_config

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

QUEUE_DIR = Path("data/runs/queue")
MAX_ATTEMPTS = 2
WEBHOOK_ENV = "ALGOARCADE_WEBHOOK_URL"
# Discord rejects a message over 2,000 characters.
MESSAGE_CHARS = 1_900
TAIL_LINES = 30
# Run names become file names, on Windows too.
NAME = re.compile(r"[A-Za-z0-9_.-]+")
STATUSES = ("queued", "running", "done", "failed")


def load_experiment(path: Path) -> list[TrainConfig]:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    experiment, base, arms, seeds = module.EXPERIMENT, module.BASE, module.ARMS, module.SEEDS
    if not isinstance(base, TrainConfig):
        raise ValueError(f"{path}: BASE is a {type(base).__name__}, not a TrainConfig")
    for name in (experiment, *arms):
        if not NAME.fullmatch(name):
            raise ValueError(f"{path}: {name!r} may use only letters, digits, '_', '.' and '-'")
    return [
        dataclasses.replace(base, **overrides, experiment_id=experiment, arm=arm, seed=seed)
        for seed in seeds
        for arm, overrides in arms.items()
    ]


def run_name(config: TrainConfig) -> str:
    return f"{config.experiment_id}-{config.arm}-seed{config.seed}"


@contextmanager
def locked(path: Path, *, wait: bool) -> Iterator[bool]:
    """Yield whether this process holds an OS lock on `path`. The OS drops the lock when the
    process ends, so a reboot or a kill never leaves the queue locked."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        try:
            if sys.platform == "win32":
                # Locks one byte at the file position; LK_LOCK gives up after 10 tries a second
                # apart.
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_LOCK if wait else msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(f, fcntl.LOCK_EX if wait else fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            if wait:
                raise
            yield False
            return
        try:
            yield True
        finally:
            if sys.platform == "win32":
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)


@contextmanager
def edit(queue_dir: Path) -> Iterator[list[dict[str, Any]]]:
    """The queue's runs, written back when the block ends without an exception. `submit` and
    `run` can edit at the same time, so every edit holds the queue lock."""
    path = queue_dir / "queue.json"
    with locked(queue_dir / "queue.lock", wait=True):
        runs = json.loads(path.read_text()) if path.exists() else []
        yield runs
        # Write then rename, so a crash mid-write leaves the old file whole.
        new = path.with_suffix(".json.new")
        new.write_text(json.dumps(runs, indent=1) + "\n")
        new.replace(path)


def load(queue_dir: Path) -> list[dict[str, Any]]:
    with edit(queue_dir) as runs:
        return runs


def submit(path: Path, queue_dir: Path, *, dry_run: bool) -> None:
    configs = load_experiment(path)
    for config in configs:
        print(
            f"{run_name(config)}  {config.env_id} {config.start_state} "
            f"{config.total_timesteps:,} steps"
        )
    if dry_run:
        print(f"{len(configs)} runs. Dry run: nothing queued.")
        return
    with edit(queue_dir) as runs:
        names = Counter([r["name"] for r in runs] + [run_name(c) for c in configs])
        repeated = sorted(name for name, n in names.items() if n > 1)
        if repeated:
            raise ValueError(f"already in the queue or repeated in {path}: {repeated}")
        (queue_dir / "configs").mkdir(exist_ok=True)
        for config in configs:
            name = run_name(config)
            save_config(config, queue_dir / "configs" / f"{name}.json")
            runs.append(
                {
                    "name": name,
                    "experiment_id": config.experiment_id,
                    "arm": config.arm,
                    "seed": config.seed,
                    "status": "queued",
                    "attempts": 0,
                    "started": "",
                    "ended": "",
                }
            )
    print(f"{len(configs)} runs queued in {queue_dir / 'queue.json'}.")


def run(queue_dir: Path) -> None:
    """Train queued runs in submit order until none is left, each attempt in its own process."""
    with locked(queue_dir / "run.lock", wait=False) as held:
        if not held:
            print("Another `run` is working through the queue. Exiting.")
            return
        with edit(queue_dir) as runs:
            # No process can still be training these: the run lock was free.
            stale = [r for r in runs if r["status"] == "running"]
            for entry in stale:
                settle(entry, ok=False)
        for entry in stale:
            report(entry, queue_dir)
        while attempt(queue_dir):
            pass


def attempt(queue_dir: Path) -> bool:
    """Train the first queued run once. False when nothing is queued."""
    with edit(queue_dir) as runs:
        entry = next((r for r in runs if r["status"] == "queued"), None)
        if entry is None:
            return False
        entry |= {
            "status": "running",
            "attempts": entry["attempts"] + 1,
            "started": now(),
            "ended": "",
        }
    name = entry["name"]
    log = log_path(queue_dir, entry)
    log.parent.mkdir(exist_ok=True)
    print(f"{now()} {name}: attempt {entry['attempts']}, log {log}")
    config = queue_dir / "configs" / f"{name}.json"
    try:
        with log.open("wb") as out:
            code = subprocess.run(
                [sys.executable, "-m", "algoarcade.runqueue.train", str(config)],
                stdout=out,
                stderr=subprocess.STDOUT,
                check=False,
            ).returncode
    except KeyboardInterrupt:
        # Stopping the queue by hand is not the run's fault: it keeps its attempt.
        with edit(queue_dir) as runs:
            find(runs, name).update(status="queued", attempts=entry["attempts"] - 1)
        raise
    with edit(queue_dir) as runs:
        entry = find(runs, name)
        settle(entry, ok=code == 0)
    print(f"{now()} {name}: {entry['status']} (exit code {code})")
    report(entry, queue_dir)
    return True


def settle(entry: dict[str, Any], *, ok: bool) -> None:
    entry["ended"] = now()
    if ok:
        entry["status"] = "done"
    else:
        entry["status"] = "queued" if entry["attempts"] < MAX_ATTEMPTS else "failed"


def report(entry: dict[str, Any], queue_dir: Path) -> None:
    """Post a failed run to the team channel."""
    if entry["status"] != "failed":
        return
    url = os.environ.get(WEBHOOK_ENV)
    if not url:
        print(f"{WEBHOOK_ENV} is not set: no message sent.")
        return
    log = log_path(queue_dir, entry)
    # Missing when the box stopped between marking the run running and opening its log.
    lines = log.read_text(encoding="utf-8", errors="replace").splitlines() if log.exists() else []
    # A worker that dies prints the cause in its own traceback, above the parent's EOFError, so
    # the tail alone can miss it.
    head = "\n".join(
        [
            f"Run failed on {socket.gethostname()} after {entry['attempts']} attempts. "
            f"Experiment {entry['experiment_id']}, run {entry['name']}, log {log}",
            *exceptions(lines),
        ]
    )
    tail = "\n".join(lines[-TAIL_LINES:])
    text = head + "\n```\n" + tail[-(MESSAGE_CHARS - len(head)) :] + "\n```"
    # Discord reads `content` and Slack reads `text`. Discord refuses urllib's default
    # User-Agent.
    request = urllib.request.Request(
        url,
        data=json.dumps({"content": text, "text": text}).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "algoarcade-runqueue"},
    )
    try:
        urllib.request.urlopen(request, timeout=30).close()
    except OSError as error:
        # The error never includes the URL, which is a secret.
        print(f"Webhook post failed: {error}")


def exceptions(lines: list[str]) -> list[str]:
    """The line that ends each traceback: the exception and its message."""
    found = []
    inside = False
    for line in lines:
        if line.startswith("Traceback (most recent call last):"):
            inside = True
        elif inside and line and not line.startswith(" "):
            found.append(line)
            inside = False
    return found


def status(queue_dir: Path) -> None:
    runs = load(queue_dir)
    if not runs:
        print("The queue is empty.")
        return
    width = max(len(r["name"]) for r in runs)
    print(f"{'run':<{width}}  {'status':<7}  attempts  started           ended")
    for r in runs:
        print(
            f"{r['name']:<{width}}  {r['status']:<7}  {r['attempts']:>8}  "
            f"{r['started']:<16}  {r['ended']}"
        )
    for experiment in dict.fromkeys(r["experiment_id"] for r in runs):
        counts = Counter(r["status"] for r in runs if r["experiment_id"] == experiment)
        print(f"{experiment}: " + ", ".join(f"{counts[s]} {s}" for s in STATUSES))


def find(runs: list[dict[str, Any]], name: str) -> dict[str, Any]:
    return next(r for r in runs if r["name"] == name)


def log_path(queue_dir: Path, entry: dict[str, Any]) -> Path:
    return queue_dir / "logs" / f"{entry['name']}-{entry['attempts']}.log"


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M")


def main(argv: list[str], queue_dir: Path = QUEUE_DIR) -> None:
    parser = argparse.ArgumentParser(prog="python -m algoarcade.runqueue")
    commands = parser.add_subparsers(dest="command", required=True)
    submit_parser = commands.add_parser("submit", help="Queue one run per arm and seed.")
    submit_parser.add_argument("experiment", type=Path)
    submit_parser.add_argument(
        "--dry-run", action="store_true", help="Print the runs and queue nothing."
    )
    commands.add_parser("run", help="Train queued runs until none is left.")
    commands.add_parser("status", help="Print every run and its status.")
    args = parser.parse_args(argv)
    if args.command == "submit":
        submit(args.experiment, queue_dir, dry_run=args.dry_run)
    elif args.command == "run":
        run(queue_dir)
    else:
        status(queue_dir)
