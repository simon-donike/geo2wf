import os
import subprocess
import sys
import time

from geo2wf.historical.acquisition import acquire_rows


def test_acquisition_import_does_not_load_training_stack():
    # Use a fresh interpreter: other tests import the training modules during
    # collection, hiding the dependency cost paid by every spawned worker.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import geo2wf.historical.acquisition; "
            "assert not {'torch', 'pytorch_lightning', 'matplotlib', 'rasterio'} "
            "& sys.modules.keys()",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def fake_acquire(root, row):
    if row["sample_id"] == "hung":
        time.sleep(60)
    if row["sample_id"] == "crashed":
        os._exit(3)
    return "ready", {"id": row["sample_id"]}


def test_hung_read_does_not_block_other_reads(tmp_path):
    rows = [{"sample_id": x} for x in ("hung", "first", "second")]
    started = time.monotonic()
    results = list(
        acquire_rows(tmp_path, rows, workers=2, task_timeout=8, worker=fake_acquire)
    )
    assert [r["sample_id"] for r, _ in results][:2] == ["first", "second"]
    assert results[-1][1][0] == "failed"
    assert results[-1][1][1]["reason"] == "acquisition_timeout"
    assert time.monotonic() - started < 20


def test_crashed_worker_is_recorded_and_replaced(tmp_path):
    rows = [{"sample_id": x} for x in ("crashed", "next")]
    results = list(
        acquire_rows(tmp_path, rows, workers=1, task_timeout=5, worker=fake_acquire)
    )
    assert results[0][1][0] == "failed"
    assert results[0][1][1]["reason"] == "worker_exit"
    assert results[1][1][0] == "ready"
