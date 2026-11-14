"""The run queue: submit, run and status on the fake game."""

import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import ClassVar

import pyarrow.parquet as pq
import pytest

pytest.importorskip("stable_baselines3")

from algoarcade.runqueue import queue

# One 64-step rollout on one env: the fake game's step budget, so each run is one episode.
EXPERIMENT = """
import dataclasses
from pathlib import Path

from algoarcade.training.config import PRESETS

FIRERED = PRESETS["firered"][1]
EXPERIMENT = {experiment!r}
BASE = dataclasses.replace(
    FIRERED,
    env_id="fake",
    start_state="",
    total_timesteps=64,
    n_envs=1,
    checkpoint_every=64,
    eval_episodes=1,
    reward_weights={{"events": 1.0, "explore": 0.1}},
    ppo=dataclasses.replace(FIRERED.ppo, n_steps=64, batch_size=32),
    runs_dir=Path({runs_dir!r}),
)
ARMS = {arms}
SEEDS = {seeds!r}
"""


def write_experiment(
    tmp_path: Path, experiment: str, arms: str, seeds: list[int], name: str = "exp.py"
) -> Path:
    path = tmp_path / name
    runs_dir = str(tmp_path / "runs")
    path.write_text(
        EXPERIMENT.format(experiment=experiment, arms=arms, seeds=seeds, runs_dir=runs_dir)
    )
    return path


def no_process(*args: object, **kwargs: object) -> None:
    raise AssertionError("started a process")


class Webhook(BaseHTTPRequestHandler):
    posts: ClassVar[list[dict[str, str]]] = []

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers["Content-Length"]))
        self.posts.append(json.loads(body))
        self.send_response(204)
        self.end_headers()


@pytest.fixture
def webhook(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[dict[str, str]]]:
    server = HTTPServer(("127.0.0.1", 0), Webhook)
    Webhook.posts = []
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv(queue.WEBHOOK_ENV, f"http://127.0.0.1:{server.server_port}/hook")
    yield Webhook.posts
    server.shutdown()
    server.server_close()


def test_dry_run_prints_ten_runs_and_starts_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write_experiment(tmp_path, "E0", '{"off": {}, "on": {"n_envs": 2}}', [0, 1, 2, 3, 4])
    monkeypatch.setattr(subprocess, "Popen", no_process)
    queue_dir = tmp_path / "queue"

    queue.main(["submit", str(path), "--dry-run"], queue_dir)

    lines = capsys.readouterr().out.splitlines()
    runs = [line.split()[0] for line in lines if line.startswith("E0-")]
    assert runs == [f"E0-{arm}-seed{seed}" for seed in range(5) for arm in ("off", "on")]
    assert len(lines) == 11
    assert not queue_dir.exists()


def test_submit_refuses_a_name_already_in_the_queue(tmp_path: Path) -> None:
    queue_dir = tmp_path / "queue"
    queue.main(["submit", str(write_experiment(tmp_path, "E0", '{"a": {}}', [0, 1]))], queue_dir)
    again = write_experiment(tmp_path, "E0", '{"a": {}}', [1, 2], name="again.py")

    with pytest.raises(ValueError, match="E0-a-seed1"):
        queue.main(["submit", str(again)], queue_dir)
    assert [r["name"] for r in queue.load(queue_dir)] == ["E0-a-seed0", "E0-a-seed1"]


def test_crash_does_not_stop_queue(
    tmp_path: Path, webhook: list[dict[str, str]], capsys: pytest.CaptureFixture[str]
) -> None:
    # The second arm names a reward component that does not exist, so it raises on every attempt.
    arms = '{"first": {}, "broken": {"reward_weights": {"no_such_component": 1.0}}, "third": {}}'
    queue_dir = tmp_path / "queue"
    queue.main(["submit", str(write_experiment(tmp_path, "crash", arms, [0]))], queue_dir)

    queue.main(["run"], queue_dir)

    runs = queue.load(queue_dir)
    assert [(r["name"], r["status"], r["attempts"]) for r in runs] == [
        ("crash-first-seed0", "done", 1),
        ("crash-broken-seed0", "failed", 2),
        ("crash-third-seed0", "done", 1),
    ]
    assert sorted(p.name for p in (queue_dir / "logs").iterdir()) == [
        "crash-broken-seed0-1.log",
        "crash-broken-seed0-2.log",
        "crash-first-seed0-1.log",
        "crash-third-seed0-1.log",
    ]
    assert {p.name.rsplit("-", 2)[0] for p in (tmp_path / "runs").iterdir()} == {
        "run_id=crash-first-seed0",
        "run_id=crash-broken-seed0",
        "run_id=crash-third-seed0",
    }
    rows = [
        row
        for path in sorted((tmp_path / "runs").glob("*/runs/*.parquet"))
        for row in pq.read_table(path, columns=["experiment_id", "arm", "status"]).to_pylist()
    ]
    assert {r["experiment_id"] for r in rows} == {"crash"}
    assert sorted((r["arm"], r["status"]) for r in rows) == [
        ("first", "finished"),
        ("first", "running"),
        ("third", "finished"),
        ("third", "running"),
    ]
    assert len(webhook) == 1
    message = webhook[0]["content"]
    assert "crash" in message
    assert "crash-broken-seed0" in message
    assert "no_such_component" in message

    capsys.readouterr()
    queue.main(["status"], queue_dir)
    out = capsys.readouterr().out
    assert "crash: 0 queued, 0 running, 2 done, 1 failed" in out


def queued(tmp_path: Path, status: str, attempts: int) -> Path:
    queue_dir = tmp_path / "queue"
    queue.main(["submit", str(write_experiment(tmp_path, "E0", '{"a": {}}', [0]))], queue_dir)
    with queue.edit(queue_dir) as runs:
        runs[0] |= {"status": status, "attempts": attempts}
    return queue_dir


def train_with(monkeypatch: pytest.MonkeyPatch, code: str) -> None:
    """Replace the trainer with a Python snippet; the run's config path stays the last argument."""
    monkeypatch.setattr(queue, "TRAIN", [sys.executable, "-c", code])


def statuses(queue_dir: Path) -> list[tuple[str, int]]:
    return [(r["status"], r["attempts"]) for r in queue.load(queue_dir)]


def test_a_run_left_running_counts_as_a_crashed_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, webhook: list[dict[str, str]]
) -> None:
    # A reboot or a killed `run` leaves its run marked running with no process behind it.
    queue_dir = queued(tmp_path, "running", 1)
    train_with(monkeypatch, "pass")

    queue.main(["run"], queue_dir)

    assert statuses(queue_dir) == [("done", 2)]
    assert webhook == []


def test_a_second_interrupted_attempt_fails_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, webhook: list[dict[str, str]]
) -> None:
    queue_dir = queued(tmp_path, "running", 2)
    monkeypatch.setattr(subprocess, "Popen", no_process)

    queue.main(["run"], queue_dir)

    assert statuses(queue_dir) == [("failed", 2)]
    assert len(webhook) == 1


def test_ctrl_c_puts_the_run_back_without_using_an_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue_dir = queued(tmp_path, "queued", 0)
    train_with(monkeypatch, "import time; time.sleep(600)")
    wait = subprocess.Popen.wait

    def interrupted(self: subprocess.Popen[bytes], timeout: float | None = None) -> int:
        if timeout is not None:
            raise KeyboardInterrupt
        return wait(self)

    monkeypatch.setattr(subprocess.Popen, "wait", interrupted)

    with pytest.raises(KeyboardInterrupt):
        queue.main(["run"], queue_dir)
    assert statuses(queue_dir) == [("queued", 0)]


def test_run_leaves_the_queue_to_a_runner_already_on_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The nightly task fires while last night's `run` is still working through the queue.
    queue_dir = queued(tmp_path, "queued", 0)
    monkeypatch.setattr(subprocess, "Popen", no_process)

    with queue.locked(queue_dir / "run.lock", wait=False) as held:
        assert held
        queue.main(["run"], queue_dir)

    assert statuses(queue_dir) == [("queued", 0)]


def test_a_hung_run_is_killed_with_its_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The trainer starts a worker that writes a heartbeat, then hangs, like a stuck SubprocVecEnv.
    heartbeat = tmp_path / "heartbeat"
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import time\nfrom pathlib import Path\n"
        f"beat = Path({str(heartbeat)!r})\n"
        "for n in range(100_000):\n    beat.write_text(str(n))\n    time.sleep(0.02)\n"
    )
    queue_dir = queued(tmp_path, "queued", 0)
    train_with(
        monkeypatch,
        f"import subprocess, sys, time; subprocess.Popen([sys.executable, {str(worker)!r}]); "
        "time.sleep(600)",
    )
    monkeypatch.setattr(queue, "STARTUP_S", 1.5)

    queue.main(["run", "--min-steps-per-s", "1e9"], queue_dir)

    assert statuses(queue_dir) == [("failed", 2)]
    assert "no exit after" in (queue_dir / "logs" / "E0-a-seed0-2.log").read_text()
    beat = heartbeat.read_text()
    time.sleep(0.5)
    assert heartbeat.read_text() == beat


def test_the_trainer_does_not_see_the_webhook_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, webhook: list[dict[str, str]]
) -> None:
    queue_dir = queued(tmp_path, "queued", 0)
    train_with(monkeypatch, f"import os, sys; sys.exit({queue.WEBHOOK_ENV!r} in os.environ)")

    queue.main(["run"], queue_dir)

    assert statuses(queue_dir) == [("done", 1)]


def test_status_reads_without_rewriting_the_queue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue_dir = queued(tmp_path, "queued", 0)
    monkeypatch.setattr(Path, "replace", no_process)

    queue.main(["status"], queue_dir)


def test_a_write_retries_while_another_program_holds_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # On Windows a virus scanner or indexer can hold queue.json open for a moment.
    replace = Path.replace
    calls = []

    def busy_twice(self: Path, target: Path) -> Path:
        calls.append(target)
        if len(calls) < 3:
            raise PermissionError
        return replace(self, target)

    monkeypatch.setattr(Path, "replace", busy_twice)

    queue_dir = queued(tmp_path, "queued", 0)

    assert len(calls) >= 3
    assert statuses(queue_dir) == [("queued", 0)]


# 16 workers dying on one bad config print 16 copies of the same exception.
WORKER_DEATH = [
    *(
        line
        for i in range(16)
        for line in (
            f"Process SpawnProcess-{i}:",
            "Traceback (most recent call last):",
            '  File "games.py", line 1, in fake_env',
            "ValueError: unknown reward component(s) ['no_such_component']; valid names: "
            + "x" * 120,
        )
    ),
    "Traceback (most recent call last):",
    '  File "subproc_vec_env.py", line 127, in __init__',
    "EOFError",
]


def test_a_16_worker_failure_fits_one_discord_message(
    tmp_path: Path, webhook: list[dict[str, str]]
) -> None:
    entry = {"name": "E0-a-seed0", "experiment_id": "E0", "attempts": 2, "status": "failed"}
    log = queue.log_path(tmp_path, entry)
    log.parent.mkdir()
    log.write_text("\n".join(WORKER_DEATH) + "\n")

    queue.report(entry, tmp_path)

    [message] = webhook
    assert len(message["content"]) <= queue.MESSAGE_CHARS
    head = message["content"].split("```")[0]
    assert head.count("ValueError: unknown reward component") == 1
    assert "EOFError" in head


@pytest.mark.parametrize(
    "url",
    [
        "discord.example/api/webhooks/123/SECRETTOKEN",
        "https://discord.example/api/webhooks/123/SECRET TOKEN",
    ],
)
def test_a_malformed_webhook_url_neither_stops_the_queue_nor_prints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    url: str,
) -> None:
    monkeypatch.setenv(queue.WEBHOOK_ENV, url)
    entry = {"name": "E0-a-seed0", "experiment_id": "E0", "attempts": 2, "status": "failed"}

    queue.report(entry, tmp_path)

    out = capsys.readouterr().out
    assert "Webhook post failed" in out
    assert "SECRET" not in out


@pytest.mark.skipif(sys.platform == "win32", reason="Windows ends the trainer through its job")
@pytest.mark.parametrize("name", ["SIGTERM", "SIGHUP"])
def test_a_terminated_run_takes_its_trainer_with_it(tmp_path: Path, name: str) -> None:
    queue_dir = queued(tmp_path, "queued", 0)
    pid_file = tmp_path / "trainer.pid"
    trainer = (
        f"import os, time; open({str(pid_file)!r}, 'w').write(str(os.getpid())); time.sleep(120)"
    )
    driver = (
        "import sys\nfrom pathlib import Path\nfrom algoarcade.runqueue import queue\n"
        f"queue.TRAIN = [sys.executable, '-c', {trainer!r}]\n"
        f"queue.run(Path({str(queue_dir)!r}))\n"
    )
    run = subprocess.Popen([sys.executable, "-c", driver])
    deadline = time.monotonic() + 30
    while not (pid_file.exists() and pid_file.read_text()):
        assert time.monotonic() < deadline, "the trainer never started"
        time.sleep(0.05)
    trainer_pid = int(pid_file.read_text())

    run.send_signal(getattr(signal, name))
    run.wait(timeout=30)

    with pytest.raises(ProcessLookupError):
        os.kill(trainer_pid, 0)
    assert statuses(queue_dir) == [("running", 1)]
