"""Exercise container dispatch and the runner's process/locking boundaries."""

import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "apps/stormsense/runner"


def wait_for(predicate):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("Timed out waiting for runner process")


def test_entrypoint_preserves_arguments_and_exit_status(tmp_path):
    executable = tmp_path / "geo2wf-train"
    executable.write_text(
        f"#!{sys.executable}\nimport json, sys\n"
        "print(json.dumps(sys.argv[1:]))\nsys.exit(7)\n"
    )
    executable.chmod(0o755)
    result = subprocess.run(
        [
            "bash",
            str(RUNNER / "entrypoint.sh"),
            "train",
            "data.root=/data/with spaces",
            "--config",
            "custom.yaml",
        ],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "STORMSENSE_ROOT": str(ROOT),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 7
    assert json.loads(result.stdout) == [
        "data.root=/data/with spaces",
        "--config",
        "custom.yaml",
    ]


def test_cycle_lock_covers_all_steps_and_workers_are_configurable(tmp_path):
    db = tmp_path / "state.sqlite"
    calls = tmp_path / "calls.jsonl"
    stub = tmp_path / "python-stub"
    stub.write_text(
        f"#!{sys.executable}\nimport json, sys\n"
        f"with open({str(calls)!r}, 'a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n"
    )
    stub.chmod(0o755)
    env = {
        **os.environ,
        "STORMSENSE_PYTHON": str(stub),
        "STORMSENSE_DB": str(db),
        "STORMSENSE_WORKERS": "1",
        "STORMSENSE_IMAGERY_WORKERS": "3",
        "STORMSENSE_PUBLISH": "0",
    }
    with Path(str(db) + ".cycle.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        result = subprocess.run(["bash", str(RUNNER / "cycle.sh")], env=env)
        assert result.returncode == 75
        assert not calls.exists()
    subprocess.run(["bash", str(RUNNER / "cycle.sh")], env=env, check=True)
    commands = [json.loads(line) for line in calls.read_text().splitlines()]
    assert commands[0][-3:] == ["update", "--workers", "1"]
    assert commands[1][-3:] == ["imagery", "--workers", "3"]
    assert not any("publish" in command for command in commands)


def test_schedule_retries_failed_cycle_and_stops_process_group(tmp_path):
    runner = tmp_path / "apps/stormsense/runner"
    runner.mkdir(parents=True)
    (runner / "cycle.sh").write_text(
        f'#!/bin/sh\nexec "{sys.executable}" "{tmp_path}/cycle.py"\n'
    )
    (tmp_path / "cycle.py").write_text(
        "from pathlib import Path\nimport os, subprocess, sys, time\n"
        f"root=Path({str(tmp_path)!r})\n"
        "if not (root/'attempted').exists():\n"
        "    (root/'attempted').touch()\n    sys.exit(7)\n"
        "child=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "(root/'pids').write_text(f'{os.getpid()} {child.pid}')\n"
        "child.wait()\n"
    )
    process = subprocess.Popen(
        [sys.executable, str(RUNNER / "schedule.py")],
        env={
            **os.environ,
            "STORMSENSE_ROOT": str(tmp_path),
            "STORMSENSE_DB": str(tmp_path / "state.sqlite"),
            "STORMSENSE_INTERVAL_SECONDS": "1",
        },
    )
    pids = []
    try:
        wait_for(lambda: (tmp_path / "pids").exists())
        pids = [int(pid) for pid in (tmp_path / "pids").read_text().split()]
        report = json.loads((tmp_path / "state.scheduler.json").read_text())
        assert report["exit_code"] == 7
        process.terminate()
        assert process.wait(timeout=5) == 0

        def stopped(pid):
            stat = Path(f"/proc/{pid}/stat")
            return not stat.exists() or stat.read_text().split()[2] == "Z"

        wait_for(lambda: all(stopped(pid) for pid in pids))
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
