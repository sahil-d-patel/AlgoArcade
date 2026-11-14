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
import contextlib
import dataclasses
import importlib.util
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any

from algoarcade.training.config import TrainConfig, load_config, save_config

if sys.platform == "win32":
    import ctypes
    import msvcrt
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
    kernel32.SetInformationJobObject.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    )
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    kernel32.TerminateJobObject.argtypes = (wintypes.HANDLE, wintypes.UINT)
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)

    class IoCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in ("r", "w", "o", "rb", "wb", "ob")]

    class BasicLimits(ctypes.Structure):
        _fields_ = (
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        )

    # JOBOBJECT_EXTENDED_LIMIT_INFORMATION, winnt.h.
    class ExtendedLimits(ctypes.Structure):
        _fields_ = (
            ("BasicLimitInformation", BasicLimits),
            ("IoInfo", IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        )

    JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
    PROCESS_SET_QUOTA_AND_TERMINATE = 0x0101
else:
    import fcntl

QUEUE_DIR = Path("data/runs/queue")
TRAIN = [sys.executable, "-m", "algoarcade.runqueue.train"]
# An attempt still running after STARTUP_S + total_timesteps / min_steps_per_s is killed. The
# floor is a fifth of the 500 steps/s planning figure; it allows 43 hours for a 15M-step run.
STARTUP_S = 3600.0
MIN_STEPS_PER_S = 100.0
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
        runs = read(path)
        yield runs
        # Write then rename, so a crash mid-write leaves the old file whole. On Windows the
        # rename fails while another program (a virus scanner, the indexer) has the file open.
        new = path.with_suffix(".json.new")
        new.write_text(json.dumps(runs, indent=1) + "\n")
        for delay in (0.1, 0.2, 0.5, 1.0, 2.0):
            try:
                new.replace(path)
                break
            except PermissionError:
                time.sleep(delay)
        else:
            new.replace(path)


def load(queue_dir: Path) -> list[dict[str, Any]]:
    with locked(queue_dir / "queue.lock", wait=True):
        return read(queue_dir / "queue.json")


def read(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text()) if path.exists() else []


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


def run(queue_dir: Path, min_steps_per_s: float = MIN_STEPS_PER_S) -> None:
    """Train queued runs in submit order until none is left, each attempt in its own process."""
    with locked(queue_dir / "run.lock", wait=False) as held:
        if not held:
            print("Another `run` is working through the queue. Exiting.")
            return
        with edit(queue_dir) as runs:
            # The run lock was free, so the `run` that started these has exited. On Windows its
            # trainer died with it (see start). On macOS and Linux a trainer outlives a `run`
            # killed with SIGKILL.
            stale = [r for r in runs if r["status"] == "running"]
            for entry in stale:
                settle(entry, ok=False)
        for entry in stale:
            report(entry, queue_dir)
        while attempt(queue_dir, min_steps_per_s):
            pass


def attempt(queue_dir: Path, min_steps_per_s: float) -> bool:
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
    limit = STARTUP_S + load_config(config).total_timesteps / min_steps_per_s
    with log.open("wb") as out:
        process, job = start([*TRAIN, str(config)], out)
        try:
            code = process.wait(timeout=limit)
        except subprocess.TimeoutExpired:
            code = None
        except KeyboardInterrupt:
            # Stopping the queue by hand is not the run's fault: it keeps its attempt.
            with edit(queue_dir) as runs:
                find(runs, name).update(status="queued", attempts=entry["attempts"] - 1)
            raise
        finally:
            kill(process, job)
        if code is None:
            out.write(
                f"\nrun queue: no exit after {limit:,.0f} s; killed it and its workers\n".encode()
            )
    with edit(queue_dir) as runs:
        entry = find(runs, name)
        settle(entry, ok=code == 0)
    ending = "timed out" if code is None else f"exit code {code}"
    print(f"{now()} {name}: {entry['status']} ({ending})")
    report(entry, queue_dir)
    return True


def start(command: list[str], out: IO[bytes]) -> tuple[subprocess.Popen[bytes], Any]:
    """Start a trainer so that `kill` also ends the SubprocVecEnv workers it starts. On Windows
    it goes in a job object the OS ends when `run` exits, however `run` exits. The webhook URL is
    a secret the trainer does not need."""
    env = {k: v for k, v in os.environ.items() if k != WEBHOOK_ENV}
    if sys.platform == "win32":
        job = kernel32.CreateJobObjectW(None, None)
        limits = ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not job or not kernel32.SetInformationJobObject(
            job, JOB_OBJECT_EXTENDED_LIMIT_INFORMATION, ctypes.byref(limits), ctypes.sizeof(limits)
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        process = subprocess.Popen(command, stdout=out, stderr=subprocess.STDOUT, env=env)
        # The trainer imports torch for seconds before it starts a worker, so it joins the job
        # before any child exists.
        handle = kernel32.OpenProcess(PROCESS_SET_QUOTA_AND_TERMINATE, False, process.pid)
        assigned = handle and kernel32.AssignProcessToJobObject(job, handle)
        error = ctypes.get_last_error()
        if handle:
            kernel32.CloseHandle(handle)
        if not assigned:
            process.kill()
            raise ctypes.WinError(error)
        return process, job
    process = subprocess.Popen(
        command, stdout=out, stderr=subprocess.STDOUT, env=env, start_new_session=True
    )
    return process, None


def kill(process: subprocess.Popen[bytes], job: Any) -> None:
    """End the trainer and every process it started, if any is left."""
    if sys.platform == "win32":
        kernel32.TerminateJobObject(job, 1)
        kernel32.CloseHandle(job)
    else:
        # The trainer's process group outlives it while one of its workers is alive.
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
    process.wait()


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
    # the tail alone can miss it. Sixteen workers dying on one bad config print it sixteen times.
    head = "\n".join(
        [
            f"Run failed on {socket.gethostname()} after {entry['attempts']} attempts. "
            f"Experiment {entry['experiment_id']}, run {entry['name']}, log {log}",
            *dict.fromkeys(exceptions(lines)),
        ]
    )[: MESSAGE_CHARS // 2]
    tail = "\n".join(lines[-TAIL_LINES:])
    text = head + "\n```\n" + tail[-(MESSAGE_CHARS - len(head) - 9) :] + "\n```"
    try:
        # Discord reads `content` and Slack reads `text`. Discord refuses urllib's default
        # User-Agent.
        request = urllib.request.Request(
            url,
            data=json.dumps({"content": text, "text": text}).encode(),
            headers={"Content-Type": "application/json", "User-Agent": "algoarcade-runqueue"},
        )
        urllib.request.urlopen(request, timeout=30).close()
    except Exception as error:  # noqa: BLE001 - a bad URL or a network error must not stop the queue
        # The type and status only: urllib's and http.client's messages can quote the URL, and
        # the URL holds the webhook's token.
        status = f" {error.code}" if isinstance(error, urllib.error.HTTPError) else ""
        print(f"Webhook post failed: {type(error).__name__}{status}")


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
    run_parser = commands.add_parser("run", help="Train queued runs until none is left.")
    run_parser.add_argument(
        "--min-steps-per-s",
        type=float,
        default=MIN_STEPS_PER_S,
        help="Kill an attempt still running after 1 h + total_timesteps / this.",
    )
    commands.add_parser("status", help="Print every run and its status.")
    args = parser.parse_args(argv)
    if args.command == "submit":
        submit(args.experiment, queue_dir, dry_run=args.dry_run)
    elif args.command == "run":
        run(queue_dir, args.min_steps_per_s)
    else:
        status(queue_dir)
