"""Canonical collation for tensor batches with per-sample metadata."""

from __future__ import annotations

from typing import Any

from torch.utils.data._utils.collate import default_collate


def collate_wind_field_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate tensors while keeping descriptive metadata sample-oriented."""

    if not samples:
        raise ValueError("cannot collate an empty sample list")
    if not isinstance(samples[0], dict):
        return default_collate(samples)
    metadata = [sample.get("meta", {}) for sample in samples]
    ibtracs = [sample.get("ibtracs") for sample in samples]
    tensor_payload = [
        {key: value for key, value in sample.items() if key not in {"meta", "ibtracs"}}
        for sample in samples
    ]
    batch = default_collate(tensor_payload)
    batch["meta"] = metadata
    if any(value is not None for value in ibtracs):
        batch["ibtracs"] = ibtracs
    return batch
