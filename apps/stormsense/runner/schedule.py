"""Run finite cycles serially, survive failures, and forward shutdown signals."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--interval",
        type=int,
        default=int(os.environ.get("STORMSENSE_INTERVAL_SECONDS", "900")),
        help="Seconds to wait after each completed cycle (default: 900)",
    )
    interval = parser.parse_args().interval
    if interval < 1:
        parser.error("interval must be positive")
    root = Path(os.environ.get("STORMSENSE_ROOT", Path(__file__).resolve().parents[3]))
    root = root.resolve()
    os.chdir(root)
    state_path = Path(os.environ.get("STORMSENSE_DB", "var/stormsense/state.sqlite"))
    status_path = state_path.with_suffix(".scheduler.json")
    status_path.parent.mkdir(parents=True, exist_ok=True)
    stopping = threading.Event()
    child = None

    def stop(signum, _frame):
        stopping.set()
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while not stopping.is_set():
        started = time.time()
        print("Starting StormSense cycle", flush=True)
        child = subprocess.Popen(
            ["bash", str(root / "apps/stormsense/runner/cycle.sh")],
            cwd=root,
            start_new_session=True,
        )
        if stopping.is_set() and child.poll() is None:
            # Cover a stop arriving between the loop check and Popen returning.
            stop(signal.SIGTERM, None)
        code = child.wait()
        child = None
        report = {"started_at": started, "finished_at": time.time(), "exit_code": code}
        temporary = status_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report) + "\n")
        temporary.replace(status_path)
        print(f"StormSense cycle finished with exit code {code}", flush=True)
        # Wait after completion so a large catch-up never overlaps or causes a
        # tight retry loop. SQLite resumes committed work after interruption.
        stopping.wait(interval)


if __name__ == "__main__":
    main()
