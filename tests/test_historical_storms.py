import json
from pathlib import Path

import numpy as np
import pytest
import torch

from geo2wf.historical import dataset as d
from geo2wf.historical.training import ScalarAdapter, HistoricalDataset, metrics
from geo2wf.operational.common import KNOT, file_hash


@pytest.fixture(autouse=True)
def inline_mocked_acquisition(monkeypatch):
    """Exercise collection state in-process; process isolation has separate tests."""

    def acquire(root, rows, workers=4, heartbeat=None):
        for row in rows:
            yield row, d.collect_one(root, row)

    monkeypatch.setattr(d, "acquire_rows", acquire)


def atcf(
    wind=70, threshold=34, quadrants=("100", "0", "50", "0"), rmw="20", code="NEQ"
):
    f = [
        "AL",
        "01",
        "2020080100",
        "",
        "BEST",
        "0",
        "200N",
        "700W",
        str(wind),
        "980",
        "HU",
        str(threshold),
        code,
        *quadrants,
        "",
        "",
        "" + rmw,
    ]
    return ", ".join(f).encode()


def test_radii_missing_zero_and_units():
    r = d.parse_labels(atcf(), "AL012020")[0]
    assert r["labels"]["vmax"] == 70 * KNOT
    assert r["labels"]["rmw"] == 20 * 1.852
    assert r["labels"]["r34"] == pytest.approx(np.sqrt((100**2 + 50**2) / 4) * 1.852)
    assert not r["label_valid"]["eye"]
    assert not r["label_valid"]["r50"]
    r = d.parse_labels(atcf(quadrants=("0",) * 4, rmw="0"), "AL012020")[0]
    assert r["labels"]["r34"] is None and r["labels"]["rmw"] is None
    r = d.parse_labels(atcf(wind=30, quadrants=("0",) * 4), "AL012020")[0]
    assert r["labels"]["r34"] == 0 and r["label_valid"]["r34"]
    assert r["label_origin"]["r34"] == "below_intensity_threshold"
    r = d.parse_labels(atcf(quadrants=("100", "-999", "0", "0")), "AL012020")[0]
    assert r["labels"]["r34"] is None
    r = d.parse_labels(atcf(quadrants=("100", "", "", ""), code="AAA"), "AL012020")[0]
    # Malformed AAA fields are safely missing, never treated as zero.
    assert r["labels"]["r34"] is None


def test_splits_deterministic_whole_storm_exclusions():
    ids = [f"AL{i:02d}2020" for i in range(1, 21)]
    a = d.split_storms(ids, {"AL012020"})
    assert a == d.split_storms(reversed(ids), {"AL012020"})
    assert "AL012020" not in a
    assert list(a.values()).count("val") == 3


def model(mode):
    return ScalarAdapter(
        mode=mode,
        architecture={
            "in_channels": 15,
            "base_channels": 4,
            "channel_mults": (1, 2),
            "intensity_hidden_features": 8,
            "intensity_dropout": 0.0,
            "structure_outputs": 5,
        },
    )


def batch():
    return {
        "condition": torch.randn(2, 14, 16, 16),
        "condition_mask": torch.ones(2, 1, 16, 16),
        "intensity": torch.tensor([45.0, 50.0]),
        "structure": torch.tensor([[float("nan"), 20.0, 0.0, 0.0, 0.0]] * 2),
        "structure_valid": torch.tensor([[False, True, True, True, True]] * 2),
    }


@pytest.mark.parametrize("mode", ["heads", "encoder"])
def test_scalar_updates_without_sar_or_decoder(mode):
    torch.set_num_threads(1)
    m = model(mode)
    before = {k: v.clone() for k, v in m.model.state_dict().items()}
    optimizer = m.configure_optimizers()
    isum, n, ssum, sn = m.loss_terms(batch())
    loss = isum / n + 0.25 * ssum / sn
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    changed = [
        k for k, v in m.model.state_dict().items() if not torch.equal(v, before[k])
    ]
    assert changed and any(m.is_head(k) for k in changed)
    assert not any("decoder" in k for k in before)
    assert any(not m.is_head(k) for k in changed) == (mode == "encoder")
    assert (
        torch.equal(
            before["structure_head.weight"][0], m.model.structure_head.weight[0]
        )
        or mode == "encoder"
    )


def test_no_structure_labels_finite():
    b = batch()
    b["structure_valid"][:] = False
    b["structure"][:] = float("nan")
    isum, n, ssum, sn = model("heads").loss_terms(b)
    assert torch.isfinite(isum) and ssum == 0 and sn == 0


def fixture_manifest(root):
    row = d.parse_labels(atcf(), "AL012020")[0]
    row.update(
        sample_id="AL012020_20200801T000000Z",
        storm_id="AL012020",
        split="train",
        basin="AL",
    )
    manifest = {
        "samples": [row],
        "stats_sha256": None,
        "sources": [],
        "holdout_storms": [],
        "research_excluded_storms": [],
    }
    (root / "stats.json").write_text("{}")
    manifest["stats_sha256"] = file_hash(root / "stats.json")
    d.write_json(root / "manifest.json", manifest)
    return manifest


def test_collection_resume_and_checksum(tmp_path, monkeypatch):
    manifest = fixture_manifest(tmp_path)
    calls = []

    def acquire(at, row):
        calls.append(at)
        return (
            np.ones((10, 256, 256), np.float32) * 250,
            np.ones((256, 256), bool),
            {"end": "2020-07-31T23:50:00Z"},
        )

    monkeypatch.setattr(d.satellite, "acquire", acquire)
    d.collect(tmp_path, manifest, workers=1)
    d.collect(tmp_path, manifest, workers=1)
    assert len(calls) == 1
    assert not d.verify(tmp_path, manifest)["errors"]
    path = next((tmp_path / "crops").rglob("*.npz"))
    with path.open("ab") as f:
        f.write(b"corrupt")
    assert "checksum" in d.verify(tmp_path, manifest)["errors"][0]


def test_gap_retry_and_pending(tmp_path, monkeypatch):
    manifest = fixture_manifest(tmp_path)
    assert d.coverage(tmp_path, manifest)["counts"]["train/pending"] == 1

    def fail(*args):
        raise d.satellite.DataGap("missing_scan")

    monkeypatch.setattr(d.satellite, "acquire", fail)
    d.collect(tmp_path, manifest, workers=1)
    assert d.coverage(tmp_path, manifest)["counts"]["train/gap"] == 1
    monkeypatch.setattr(
        d.satellite,
        "acquire",
        lambda *a: (
            np.zeros((10, 256, 256), np.float32),
            np.ones((256, 256), bool),
            {"end": "2020-07-31T23:50:00Z"},
        ),
    )
    d.collect(tmp_path, manifest, workers=1, retry=True)
    assert d.coverage(tmp_path, manifest)["counts"]["train/ready"] == 1


def test_metrics_missing_zero_and_storm_macro():
    rows = [
        {
            "storm_id": s,
            "target": dict(vmax=t, rmw=None, r34=0, r50=None, r64=None),
            "prediction": dict(vmax=p, r34=1),
        }
        for s, t, p in [("a", 10, 12), ("a", 10, 12), ("b", 10, 16)]
    ]
    m = metrics(rows)
    assert m["vmax"]["storm_macro_mae"] == 4
    assert m["r34"]["n"] == 3 and m["rmw"]["n"] == 0


def test_pinned_encoder_matches_operational_and_preprocessing(tmp_path):
    from geo2wf.historical.training import load_encoder
    from geo2wf.operational.models import prepare, MANIFEST, resolve_file, checkpoint
    from geo2wf.models.bottleneck_unet_mlp import BottleneckUNetMLPRegressor

    try:
        path = checkpoint("nowcast", "downloads/models")
    except FileNotFoundError:
        pytest.skip("Pinned checkpoint unavailable")
    torch.set_num_threads(1)
    manifest = fixture_manifest(tmp_path)
    stats = resolve_file(MANIFEST["models"]["nowcast"]["stats"], "downloads/models")
    (tmp_path / "stats.json").write_bytes(stats.read_bytes())
    manifest["stats_sha256"] = file_hash(tmp_path / "stats.json")
    d.write_json(tmp_path / "manifest.json", manifest)
    array = np.full((10, 256, 256), 250, dtype=np.float32)
    valid = np.ones((256, 256), bool)
    np.savez_compressed(tmp_path / "crop.npz", array=array, valid=valid)
    with d.database(tmp_path) as db:
        db.execute(
            "INSERT INTO observations VALUES (?,?,?)",
            (
                manifest["samples"][0]["sample_id"],
                "ready",
                json.dumps({"path": "crop.npz"}),
            ),
        )
    item = HistoricalDataset(tmp_path, "train")[0]
    row = manifest["samples"][0]
    prepared = prepare(
        array, valid, row["lat"], row["lon"], row["time"], json.loads(stats.read_text())
    )
    assert torch.equal(item["condition"], prepared["condition"][0])
    assert "target_physical" not in item
    source = BottleneckUNetMLPRegressor.load_from_checkpoint(
        path, map_location="cpu"
    ).eval()
    encoder = load_encoder("downloads/models").eval()
    with torch.inference_mode():
        expected = source(prepared["condition"], prepared["condition_mask"])
        inputs = torch.cat(
            [
                prepared["condition"] * prepared["condition_mask"],
                prepared["condition_mask"],
            ],
            dim=1,
        )
        actual = encoder(inputs)
    torch.testing.assert_close(
        actual.intensity_prediction_ms, expected.intensity_prediction_ms
    )
    torch.testing.assert_close(
        actual.structure_prediction_km, expected.structure_prediction_km
    )


def test_lightning_train_checkpoint_restore(tmp_path):
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader

    torch.set_num_threads(1)
    m = model("encoder")
    b = batch()
    rows = [{k: v[i] for k, v in b.items()} for i in range(2)]
    loader = DataLoader(rows, batch_size=2)
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(m, loader, loader)
    assert torch.isfinite(trainer.callback_metrics["val/loss"])
    path = tmp_path / "scalar.ckpt"
    trainer.save_checkpoint(path)
    restored = ScalarAdapter.load_from_checkpoint(path).eval()
    m.eval()
    with torch.inference_mode():
        for a, bv in zip(m(b), restored(b)):
            torch.testing.assert_close(a, bv)


def test_verify_rejects_future_scan(tmp_path, monkeypatch):
    manifest = fixture_manifest(tmp_path)
    monkeypatch.setattr(
        d.satellite,
        "acquire",
        lambda *a: (
            np.zeros((10, 256, 256), np.float32),
            np.ones((256, 256), bool),
            {"end": "2020-08-01T00:01:00Z"},
        ),
    )
    d.collect(tmp_path, manifest, workers=1)
    assert "scan timing" in d.verify(tmp_path, manifest)["errors"][0]


def test_discovery_fails_closed_and_whole_storm_holdout(tmp_path, monkeypatch):
    import sqlite3

    root = tmp_path / "dataset"
    root.mkdir()
    state = tmp_path / "state.sqlite"
    with sqlite3.connect(state) as db:
        db.execute("CREATE TABLE storms(body TEXT)")
        db.execute(
            "INSERT INTO storms VALUES (?)",
            (
                json.dumps(
                    {"id": "AL012025", "track": [{"time": "2025-10-03T00:00:00Z"}]}
                ),
            ),
        )
    stats = tmp_path / "stats.json"
    stats.write_text("{}")
    monkeypatch.setattr(d, "research_exclusions", lambda: ({"AL022020"}, []))
    monkeypatch.setattr(d, "training_storms", lambda _: {"AL012025"})
    monkeypatch.setattr(d, "resolve_file", lambda *_: stats)
    monkeypatch.setitem(
        d.MANIFEST["models"]["nowcast"], "stats_sha256", file_hash(stats)
    )

    def urls(year):
        return (
            [("AL012025", "https://test/holdout")]
            if year == 2025
            else (
                [
                    ("AL022020", "https://test/research"),
                    ("AL032020", "https://test/train"),
                ]
                if year == 2020
                else []
            )
        )

    monkeypatch.setattr(d.tracks, "track_urls", urls)

    def fetch(url):
        if url.endswith("holdout"):
            return b"\n".join(
                atcf().replace(b"2020080100", t) for t in (b"2025093000", b"2025100300")
            )
        return atcf()

    monkeypatch.setattr(d, "fetch", fetch)
    m = d.discover(root, state, "unused")
    assert not any(r["storm_id"] == "AL022020" for r in m["samples"])
    holdout = [r for r in m["samples"] if r["storm_id"] == "AL012025"]
    assert len(holdout) == 1 and holdout[0]["split"] == "test"
    assert holdout[0]["pretraining_membership"] == "seen"
    assert all(
        r["storm_id"] != "AL012025" for r in m["samples"] if r["split"] == "train"
    )
