"""Process-isolated satellite reads with bounded concurrency and wall-clock limits.

NetCDF/HDF5 libraries and HTTP file handles must not share a process across
concurrent crop reads. A fresh spawned process also avoids inheriting library
locks. The parent alone writes collection state; a stuck child is killable.
"""

from __future__ import annotations

import multiprocessing as mp
import time


def _collect(root, row):
    from .dataset import collect_one

    return collect_one(root, row)


def _child(root, row, connection, worker):
    try:
        try:
            result = worker(root, row)
        except Exception as exc:
            result = (
                "failed",
                {"reason": type(exc).__name__, "detail": str(exc), "retryable": True},
            )
        connection.send(result)
    finally:
        connection.close()


def _stop(process):
    if process.is_alive():
        process.terminate()
    process.join(timeout=2)
    if process.is_alive():
        process.kill()
        process.join(timeout=2)
    process.close()


def acquire_rows(
    root, rows, workers=4, task_timeout=600, heartbeat=None, worker=_collect
):
    """Yield completed (row, result) pairs immediately, including timed-out reads."""
    if workers < 1 or task_timeout <= 0:
        raise ValueError("workers and task_timeout must be positive")
    context = mp.get_context("spawn")
    pending = iter(rows)
    active = []
    exhausted = False
    last_heartbeat = 0.0
    try:
        while active or not exhausted:
            while len(active) < workers and not exhausted:
                try:
                    row = next(pending)
                except StopIteration:
                    exhausted = True
                    break
                receiver, sender = context.Pipe(duplex=False)
                process = context.Process(
                    target=_child, args=(root, row, sender, worker), daemon=True
                )
                process.start()
                sender.close()
                active.append((row, process, receiver, time.monotonic()))
            now = time.monotonic()
            if heartbeat is not None and now - last_heartbeat >= 5:
                heartbeat(
                    [
                        {
                            "sample_id": r["sample_id"],
                            "pid": p.pid,
                            "elapsed_seconds": round(now - started, 1),
                        }
                        for r, p, _, started in active
                    ]
                )
                last_heartbeat = now
            completed = False
            for entry in list(active):
                row, process, receiver, started = entry
                result = None
                if receiver.poll():
                    try:
                        result = receiver.recv()
                    except EOFError:
                        result = (
                            "failed",
                            {
                                "reason": "worker_exit",
                                "detail": f"Worker exited without a result ({process.exitcode})",
                                "retryable": True,
                            },
                        )
                elif now - started >= task_timeout:
                    result = (
                        "failed",
                        {
                            "reason": "acquisition_timeout",
                            "detail": f"Exceeded {task_timeout}s wall-clock deadline",
                            "retryable": True,
                        },
                    )
                elif not process.is_alive():
                    result = (
                        "failed",
                        {
                            "reason": "worker_exit",
                            "detail": f"Worker exited with code {process.exitcode}",
                            "retryable": True,
                        },
                    )
                if result is not None:
                    active.remove(entry)
                    receiver.close()
                    _stop(process)
                    completed = True
                    yield row, result
            if not completed:
                time.sleep(0.1)
    finally:
        for _, process, receiver, _ in active:
            receiver.close()
            _stop(process)
