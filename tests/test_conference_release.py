from __future__ import annotations

import copy
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from scripts.conference_release import ROOT, paper_rows, portable, registry, tables


def test_exact_publication_tables(tmp_path):
    tables(registry(), tmp_path)
    for name in paper_rows(registry()):
        assert (tmp_path / f"{name}.tex").read_bytes() == (
            ROOT / "release/paper" / f"{name}.tex"
        ).read_bytes()


def test_joint_models_are_not_substituted_between_tables():
    reg = registry()
    for regime, latent in (
        ("with-era5", "latent_sar_era5_max_wind"),
        ("without-era5", "latent_sar_no_era5_max_wind"),
    ):
        joint = reg["architecture"][regime]["models"]["joint"]
        assert reg["models"][joint]["sha256"] != reg["models"][latent]["sha256"]


def test_portability_relocates_checkpoint_data_and_run_independently(tmp_path):
    reg = registry()
    ref = reg["models"]["latent_sar_era5_max_wind"]
    original = {
        "checkpoint": ref["original_checkpoint"],
        "data": "data/geotiff/paired",
        "run": reg["source_root"] + "/logs/run",
    }
    saved = copy.deepcopy(original)
    result = portable(
        original,
        reg,
        tmp_path / "artifacts",
        tmp_path / "observations",
        tmp_path / "runtime",
    )
    assert original == saved
    assert result == {
        "checkpoint": str(tmp_path / "artifacts" / ref["checkpoint"]),
        "data": str(tmp_path / "observations/geotiff/paired"),
        "run": str(tmp_path / "runtime/logs/run"),
    }


def test_provenance_and_compressed_manifests_are_intact():
    for relative, info in registry()["files"].items():
        content = (ROOT / relative).read_bytes()
        assert hashlib.sha256(content).hexdigest() == info["sha256"], relative
        if "uncompressed_sha256" in info:
            assert (
                hashlib.sha256(gzip.decompress(content)).hexdigest()
                == info["uncompressed_sha256"]
            )


def test_all_retained_dependencies_are_registered():
    reg = registry()
    for ref in reg["models"].values():
        assert set(ref["dependencies"]).issubset(reg["models"])
    assert reg["models"]["structure-cache-field"]["include_test_in_train"] is True


def test_wrong_source_metrics_fail_instead_of_overwriting_publication(tmp_path):
    reg = registry()
    report = json.loads((ROOT / reg["architecture"]["with-era5"]["report"]).read_text())
    report["table"][0]["intensity_mae_ms"] = 999
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(report))
    reg["architecture"]["with-era5"]["report"] = str(changed)
    with pytest.raises(ValueError, match="do not reproduce"):
        tables(reg, tmp_path / "output")
