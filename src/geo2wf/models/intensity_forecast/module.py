"""Scalar residual MLP for six-hour maximum-wind forecasts."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from geo2wf.data.intensity_forecast import (
    FORECAST_FEATURE_NAMES,
    IntensityForecastDataSpec,
)
from geo2wf.tracking.forecast_media import log_wandb_ri_forecasts


@dataclass(frozen=True)
class IntensityForecastPrediction:
    anchor_wind_ms: torch.Tensor
    predicted_delta_ms: torch.Tensor
    output_wind_ms: torch.Tensor


def _metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = prediction - target
    return {
        "mae_ms": float(np.mean(np.abs(error))),
        "rmse_ms": float(np.sqrt(np.mean(np.square(error)))),
        "bias_ms": float(np.mean(error)),
    }


def summarize_forecast_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize empty forecast rows")
    prediction = np.asarray([row["prediction_ms"] for row in rows], dtype=float)
    target = np.asarray([row["target_ms"] for row in rows], dtype=float)
    persistence = np.asarray([row["persistence_ms"] for row in rows], dtype=float)
    trend = np.asarray([row["trend_ms"] for row in rows], dtype=float)
    if not all(
        np.isfinite(value).all() for value in (prediction, target, persistence, trend)
    ):
        raise ValueError("forecast evaluation rows must be finite")
    per_storm = {}
    for storm_id in sorted({str(row["storm_id"]) for row in rows}):
        selected = np.asarray(
            [str(row["storm_id"]) == storm_id for row in rows], dtype=bool
        )
        per_storm[storm_id] = {
            "samples": int(selected.sum()),
            **_metrics(prediction[selected], target[selected]),
            "persistence_mae_ms": _metrics(persistence[selected], target[selected])[
                "mae_ms"
            ],
            "trend_mae_ms": _metrics(trend[selected], target[selected])["mae_ms"],
        }
    return {
        "samples": len(rows),
        "storms": len(per_storm),
        "regression": _metrics(prediction, target),
        "persistence_baseline": _metrics(persistence, target),
        "recent_trend_baseline": _metrics(trend, target),
        "storm_macro_mae_ms": float(
            np.mean([item["mae_ms"] for item in per_storm.values()])
        ),
        "persistence_storm_macro_mae_ms": float(
            np.mean([item["persistence_mae_ms"] for item in per_storm.values()])
        ),
        "trend_storm_macro_mae_ms": float(
            np.mean([item["trend_mae_ms"] for item in per_storm.values()])
        ),
        "per_storm": per_storm,
    }


class IntensityForecastMLP(pl.LightningModule):
    checkpoint_monitor = "val/storm_macro_mae_ms"
    checkpoint_mode = "min"

    def __init__(
        self,
        feature_count: int = len(FORECAST_FEATURE_NAMES),
        hidden_features: Sequence[int] = (32, 16),
        dropout: float = 0.1,
        huber_delta_ms: float = 5.0,
        lr: float = 3.0e-4,
        weight_decay: float = 1.0e-4,
        lr_scheduler_factor: float = 0.5,
        lr_scheduler_patience: int = 10,
        lr_scheduler_min_lr: float = 1.0e-6,
        log_wandb_ri_media: bool = True,
    ) -> None:
        super().__init__()
        widths = tuple(int(value) for value in hidden_features)
        if feature_count <= 0 or not widths or any(value <= 0 for value in widths):
            raise ValueError("forecast MLP dimensions must be positive")
        if not 0.0 <= dropout < 1.0 or huber_delta_ms <= 0:
            raise ValueError("invalid forecast dropout or Huber delta")
        self.save_hyperparameters()
        self.feature_count = int(feature_count)
        self.huber_delta_ms = float(huber_delta_ms)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.lr_scheduler_factor = float(lr_scheduler_factor)
        self.lr_scheduler_patience = int(lr_scheduler_patience)
        self.lr_scheduler_min_lr = float(lr_scheduler_min_lr)
        self.log_wandb_ri_media = bool(log_wandb_ri_media)
        self.register_buffer("feature_mean", torch.zeros(self.feature_count))
        self.register_buffer("feature_std", torch.ones(self.feature_count))
        layers: list[nn.Module] = []
        input_width = self.feature_count
        for index, width in enumerate(widths):
            layers.extend(
                [nn.Linear(input_width, width), nn.LayerNorm(width), nn.SiLU()]
            )
            if index == 0 and dropout:
                layers.append(nn.Dropout(dropout))
            input_width = width
        self.encoder = nn.Sequential(*layers)
        self.delta_head = nn.Linear(input_width, 1)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        self._evaluation_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def validate_data_spec(self, spec: IntensityForecastDataSpec) -> None:
        if not isinstance(spec, IntensityForecastDataSpec):
            raise TypeError("IntensityForecastMLP requires IntensityForecastDataSpec")
        if spec.feature_names != FORECAST_FEATURE_NAMES:
            raise ValueError(
                f"forecast feature names are incompatible: {spec.feature_names}"
            )
        if spec.feature_count != self.feature_count:
            raise ValueError(
                f"model expects {self.feature_count} forecast features, "
                f"data provides {spec.feature_count}"
            )
        mean = torch.tensor(spec.feature_mean, dtype=self.feature_mean.dtype)
        std = torch.tensor(spec.feature_std, dtype=self.feature_std.dtype)
        if (
            not torch.isfinite(mean).all()
            or not torch.isfinite(std).all()
            or (std <= 0).any()
        ):
            raise ValueError(
                "forecast scaler must be finite with positive standard deviations"
            )
        self.feature_mean.copy_(mean)
        self.feature_std.copy_(std)

    @staticmethod
    def make_features(
        anchor: torch.Tensor, minus_6: torch.Tensor, minus_12: torch.Tensor
    ) -> torch.Tensor:
        return torch.stack(
            [
                anchor,
                minus_6,
                minus_12,
                anchor - minus_6,
                minus_6 - minus_12,
            ],
            dim=-1,
        )

    def predict_forecast(
        self, features: torch.Tensor, anchor_wind_ms: torch.Tensor
    ) -> IntensityForecastPrediction:
        if features.ndim != 2 or features.shape[1] != self.feature_count:
            raise ValueError(
                f"forecast features must have shape [B,{self.feature_count}]"
            )
        anchor = anchor_wind_ms.to(features).reshape(-1)
        if anchor.shape[0] != features.shape[0]:
            raise ValueError("forecast anchor batch size disagrees with features")
        if not torch.isfinite(features).all() or not torch.isfinite(anchor).all():
            raise ValueError("forecast inputs must be finite")
        normalized = (features - self.feature_mean) / self.feature_std
        delta = self.delta_head(self.encoder(normalized)).squeeze(1)
        output = (anchor + delta).clamp_min(0.0)
        return IntensityForecastPrediction(anchor, delta, output)

    def forward(
        self, features: torch.Tensor, anchor_wind_ms: torch.Tensor
    ) -> torch.Tensor:
        return self.predict_forecast(features, anchor_wind_ms).output_wind_ms

    def predict_two_steps(
        self,
        anchor_wind_ms: torch.Tensor,
        wind_minus_6h_ms: torch.Tensor,
        wind_minus_12h_ms: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        anchor = anchor_wind_ms.reshape(-1)
        minus_6 = wind_minus_6h_ms.to(anchor).reshape(-1)
        minus_12 = wind_minus_12h_ms.to(anchor).reshape(-1)
        first_features = self.make_features(anchor, minus_6, minus_12)
        plus_6 = self.predict_forecast(first_features, anchor).output_wind_ms
        second_features = self.make_features(plus_6, anchor, minus_6)
        plus_12 = self.predict_forecast(second_features, plus_6).output_wind_ms
        return plus_6, plus_12

    def _loss(
        self, batch: Mapping[str, Any]
    ) -> tuple[torch.Tensor, IntensityForecastPrediction]:
        prediction = self.predict_forecast(batch["features"], batch["anchor_wind_ms"])
        target = batch["target_wind_ms"].to(prediction.output_wind_ms).reshape(-1)
        weights = batch.get("sample_weight", torch.ones_like(target)).to(target)
        element = F.smooth_l1_loss(
            prediction.output_wind_ms,
            target,
            reduction="none",
            beta=self.huber_delta_ms,
        )
        return (element * weights).sum() / weights.sum().clamp_min(1e-12), prediction

    def training_step(self, batch: Mapping[str, Any], batch_idx: int) -> torch.Tensor:
        del batch_idx
        loss, prediction = self._loss(batch)
        target = batch["target_wind_ms"].to(prediction.output_wind_ms)
        error = prediction.output_wind_ms - target
        for name, value in {
            "loss": loss,
            "mae_ms": error.abs().mean(),
            "rmse_ms": error.square().mean().sqrt(),
            "bias_ms": error.mean(),
        }.items():
            self.log(
                f"train/{name}",
                value,
                on_step=name == "loss",
                on_epoch=True,
                prog_bar=name in {"loss", "mae_ms"},
                batch_size=int(target.numel()),
            )
        return loss

    @staticmethod
    def _rows(batch: Mapping[str, Any], prediction: IntensityForecastPrediction):
        target = batch["target_wind_ms"].to(prediction.output_wind_ms)
        minus_6 = batch["wind_minus_6h_ms"].to(target)
        minus_12 = batch["wind_minus_12h_ms"].to(target)
        trend = (prediction.anchor_wind_ms + minus_6 - minus_12).clamp_min(0.0)
        arrays = [
            value.detach().cpu().reshape(-1).tolist()
            for value in (
                prediction.output_wind_ms,
                target,
                prediction.anchor_wind_ms,
                trend,
                prediction.predicted_delta_ms,
            )
        ]
        return [
            {
                "sample_id": str(sample_id),
                "storm_id": str(storm_id),
                "init_timestamp": str(timestamp),
                "prediction_ms": float(predicted),
                "target_ms": float(observed),
                "persistence_ms": float(anchor),
                "trend_ms": float(trend_value),
                "predicted_delta_ms": float(delta),
            }
            for sample_id, storm_id, timestamp, predicted, observed, anchor, trend_value, delta in zip(
                batch["sample_id"],
                batch["storm_id"],
                batch["init_timestamp"],
                *arrays,
            )
        ]

    def _evaluation_step(self, batch: Mapping[str, Any], prefix: str) -> None:
        loss, prediction = self._loss(batch)
        self.log(
            f"{prefix}/loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=int(prediction.output_wind_ms.numel()),
        )
        self._evaluation_rows[prefix].extend(self._rows(batch, prediction))

    def validation_step(self, batch: Mapping[str, Any], batch_idx: int) -> None:
        del batch_idx
        self._evaluation_step(batch, "val")

    def test_step(self, batch: Mapping[str, Any], batch_idx: int) -> None:
        del batch_idx
        self._evaluation_step(batch, "test")

    def on_validation_epoch_start(self) -> None:
        self._evaluation_rows["val"] = []

    def on_test_epoch_start(self) -> None:
        self._evaluation_rows["test"] = []

    @staticmethod
    def _distributed_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
        ):
            return rows
        gathered: list[list[dict[str, Any]] | None] = [
            None for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather_object(gathered, rows)
        return [row for part in gathered if part is not None for row in part]

    def _log_evaluation(self, prefix: str) -> None:
        rows = self._distributed_rows(self._evaluation_rows[prefix])
        summary = summarize_forecast_rows(rows)
        metrics = {
            **summary["regression"],
            "storm_macro_mae_ms": summary["storm_macro_mae_ms"],
            "persistence_mae_ms": summary["persistence_baseline"]["mae_ms"],
            "persistence_storm_macro_mae_ms": summary["persistence_storm_macro_mae_ms"],
            "trend_mae_ms": summary["recent_trend_baseline"]["mae_ms"],
            "trend_storm_macro_mae_ms": summary["trend_storm_macro_mae_ms"],
        }
        for name, value in metrics.items():
            self.log(
                f"{prefix}/{name}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=name in {"mae_ms", "storm_macro_mae_ms"},
                sync_dist=False,
            )
        if prefix == "val" and self.log_wandb_ri_media:
            datamodule = getattr(self.trainer, "datamodule", None)
            cases = getattr(datamodule, "ri_rollout_cases", [])
            log_wandb_ri_forecasts(self, cases, prefix=prefix)

    def on_validation_epoch_end(self) -> None:
        self._log_evaluation("val")

    def on_test_epoch_end(self) -> None:
        self._log_evaluation("test")

    @torch.no_grad()
    def predict_step(
        self, batch: Mapping[str, Any], batch_idx: int, dataloader_idx: int = 0
    ) -> dict[str, Any]:
        del batch_idx, dataloader_idx
        prediction = self.predict_forecast(batch["features"], batch["anchor_wind_ms"])
        return {
            "sample_id": list(batch["sample_id"]),
            "storm_id": list(batch["storm_id"]),
            "init_timestamp": list(batch["init_timestamp"]),
            "anchor_wind_ms": prediction.anchor_wind_ms,
            "predicted_delta_ms": prediction.predicted_delta_ms,
            "output_wind_ms": prediction.output_wind_ms,
        }

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.lr_scheduler_factor,
            patience=self.lr_scheduler_patience,
            min_lr=self.lr_scheduler_min_lr,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self.checkpoint_monitor,
            },
        }


__all__ = [
    "IntensityForecastMLP",
    "IntensityForecastPrediction",
    "summarize_forecast_rows",
]
