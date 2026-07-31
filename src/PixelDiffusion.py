import copy
import math
import hashlib
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/dif_img_rec_matplotlib")

import pytorch_lightning as pl
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_erosion
from torch.optim.lr_scheduler import ReduceLROnPlateau
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from .DenoisingDiffusionProcess import *
from .DenoisingDiffusionProcess.samplers import DDIM_Sampler, DDPM_Sampler
from .reconstruction_logging import log_wandb_reconstruction
from .wind_metrics import RADIAL_METRIC_NAMES, radial_wind_metric_statistics

STORM_METRIC_NAMES = ("high_wind_mae_ms",) + RADIAL_METRIC_NAMES


class PixelDiffusionConditional(pl.LightningModule):
    """Conditional pixel-space diffusion Lightning module.

    Expects each batch to provide:
    - `condition`: conditional tensor
    - `target`: target tensor to reconstruct/generate
    - `target_mask`: optional target-validity mask
    """

    checkpoint_monitor = "val/eye_structure_score"
    checkpoint_mode = "min"

    def __init__(
        self,
        condition_channels=3,
        generated_channels=3,
        num_timesteps=1000,
        schedule="linear",
        model_dim=64,
        model_dim_mults=(1, 2, 4, 8),
        model_channels=None,
        model_out_dim=None,
        lr=1e-3,
        lr_scheduler_factor=0.5,
        lr_scheduler_patience=25,
        lr_scheduler_monitor="val/eye_structure_score",
        lr_scheduler_cooldown=0,
        lr_scheduler_min_lr=0.0,
        sampling_method="ddpm",
        sampling_timesteps=None,
        sampling_eta=0.0,
        clip_sample=True,
        sparse_target_fill=None,
        unobserved_loss_weight=0.0,
        validation_reconstruction_batches=1,
        validation_seed=42,
        ema_decay=None,
        ema_update_after_step=0,
        ema_use_for_eval=True,
        min_snr_gamma=None,
        condition_dropout_probability=0.0,
        guidance_scale=1.0,
        validation_ensemble_size=1,
        validation_ensemble_batches=1,
        probabilistic_score_sharpness_weight=2.0,
        probabilistic_score_target_sharpness_ratio=1.0,
        probabilistic_score_peak_weight=0.0,
        probabilistic_peak_fraction=0.005,
        log_reconstruction_images=True,
    ):
        super().__init__()
        if not 0.0 <= float(unobserved_loss_weight) <= 1.0:
            raise ValueError("unobserved_loss_weight must be in [0, 1]")
        if validation_reconstruction_batches < 1:
            raise ValueError("validation_reconstruction_batches must be positive")
        if validation_ensemble_size < 1:
            raise ValueError("validation_ensemble_size must be positive")
        if validation_ensemble_batches < 1:
            raise ValueError("validation_ensemble_batches must be positive")
        if ema_decay is not None and not 0.0 < float(ema_decay) < 1.0:
            raise ValueError("ema_decay must be in (0, 1)")
        if min_snr_gamma is not None and float(min_snr_gamma) <= 0:
            raise ValueError("min_snr_gamma must be positive")
        if not 0.0 <= float(condition_dropout_probability) < 1.0:
            raise ValueError("condition_dropout_probability must be in [0, 1)")
        if float(guidance_scale) < 0:
            raise ValueError("guidance_scale must be non-negative")
        if float(probabilistic_score_sharpness_weight) < 0:
            raise ValueError(
                "probabilistic_score_sharpness_weight must be non-negative"
            )
        if float(probabilistic_score_target_sharpness_ratio) <= 0:
            raise ValueError(
                "probabilistic_score_target_sharpness_ratio must be positive"
            )
        if float(probabilistic_score_peak_weight) < 0:
            raise ValueError("probabilistic_score_peak_weight must be non-negative")
        if not 0.0 < float(probabilistic_peak_fraction) <= 1.0:
            raise ValueError("probabilistic_peak_fraction must be in (0, 1]")
        self.lr = lr
        self.lr_scheduler_factor = lr_scheduler_factor
        self.lr_scheduler_patience = lr_scheduler_patience
        self.lr_scheduler_monitor = str(lr_scheduler_monitor)
        self.lr_scheduler_cooldown = int(lr_scheduler_cooldown)
        self.lr_scheduler_min_lr = float(lr_scheduler_min_lr)
        self.schedule = str(schedule)
        self._backward_steps = 0
        self.sparse_target_fill = sparse_target_fill
        self.unobserved_loss_weight = float(unobserved_loss_weight)
        self.validation_reconstruction_batches = int(validation_reconstruction_batches)
        self.log_reconstruction_images = bool(log_reconstruction_images)
        self.validation_ensemble_size = int(validation_ensemble_size)
        self.validation_ensemble_batches = int(validation_ensemble_batches)
        self.validation_seed = int(validation_seed)
        self.ema_decay = None if ema_decay is None else float(ema_decay)
        self.ema_update_after_step = int(ema_update_after_step)
        self.ema_use_for_eval = bool(ema_use_for_eval)
        self.min_snr_gamma = None if min_snr_gamma is None else float(min_snr_gamma)
        self.condition_dropout_probability = float(condition_dropout_probability)
        self.guidance_scale = float(guidance_scale)
        self.probabilistic_score_sharpness_weight = float(
            probabilistic_score_sharpness_weight
        )
        self.probabilistic_score_target_sharpness_ratio = float(
            probabilistic_score_target_sharpness_ratio
        )
        self.probabilistic_score_peak_weight = float(probabilistic_score_peak_weight)
        self.probabilistic_peak_fraction = float(probabilistic_peak_fraction)

        sampling_method = str(sampling_method).strip().lower()
        sampling_timesteps = int(sampling_timesteps or num_timesteps)
        if sampling_method == "ddpm":
            if sampling_timesteps != num_timesteps:
                raise ValueError(
                    "DDPM sampling_timesteps must equal the training timesteps; "
                    "use DDIM for a reduced reverse schedule"
                )
            sampler = DDPM_Sampler(
                num_timesteps=num_timesteps,
                schedule=schedule,
                clip_sample=clip_sample,
            )
        elif sampling_method == "ddim":
            sampler = DDIM_Sampler(
                num_timesteps=sampling_timesteps,
                train_timesteps=num_timesteps,
                schedule=schedule,
                clip_sample=clip_sample,
                eta=sampling_eta,
            )
        else:
            raise ValueError("sampling_method must be 'ddpm' or 'ddim'")

        # Core conditional diffusion process used by training, validation, and prediction.
        self.model = DenoisingDiffusionConditionalProcess(
            generated_channels=generated_channels,
            condition_channels=condition_channels,
            schedule=schedule,
            num_timesteps=num_timesteps,
            model_dim=model_dim,
            model_dim_mults=model_dim_mults,
            model_channels=model_channels,
            model_out_dim=model_out_dim,
            sampler=sampler,
        )
        self.ema_model = None
        self.register_buffer("_ema_updates", torch.zeros((), dtype=torch.long))
        if self.ema_decay is not None:
            self.ema_model = copy.deepcopy(self.model)
            self.ema_model.requires_grad_(False)
            self.ema_model.eval()
        self.register_buffer(
            "_validation_storm_statistics",
            torch.zeros((len(STORM_METRIC_NAMES), 2), dtype=torch.float64),
            persistent=False,
        )
        self.register_buffer(
            "_test_storm_statistics",
            torch.zeros((len(STORM_METRIC_NAMES), 2), dtype=torch.float64),
            persistent=False,
        )
        self.register_buffer(
            "_validation_physical_statistics",
            torch.zeros(6, dtype=torch.float64),
            persistent=False,
        )
        self.register_buffer(
            "_test_physical_statistics",
            torch.zeros(6, dtype=torch.float64),
            persistent=False,
        )

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        """Lightning inference helper; returns output mapped back to [0, 1]."""
        process = (
            self.ema_model
            if not self.training
            and self.ema_model is not None
            and self.ema_use_for_eval
            else self.model
        )
        return self.output_T(process(*args, **kwargs))

    def input_T(self, input):
        # Model internally expects values in [-1, 1].
        return input.clamp(0, 1).mul(2).sub(1)

    def output_T(self, input):
        # Inverse mapping from [-1, 1] back to [0, 1] for visualization/metrics.
        return input.add(1).div(2).clamp(0, 1)

    def _condition_for_training(self, condition):
        """Drop complete conditions per sample for classifier-free guidance."""
        probability = self.condition_dropout_probability
        if probability <= 0:
            return condition
        keep = (
            torch.rand(condition.shape[0], 1, 1, 1, device=condition.device)
            >= probability
        )
        return condition * keep.to(condition.dtype)

    def _diffusion_loss(self, model_target, condition, loss_weight, batch, *, stage):
        """Hook for task-specific diffusion objectives."""
        del batch
        if stage == "train":
            condition = self._condition_for_training(condition)
        return self.model.p_loss(
            model_target,
            condition,
            mask=loss_weight,
            min_snr_gamma=self.min_snr_gamma,
        )

    def training_step(self, batch, batch_idx):
        """Lightning train hook for conditional diffusion."""
        input, output, target_mask = self._unpack_batch(batch)
        model_target, condition, loss_weight = self._prepare_training_inputs(
            input, output, target_mask, batch
        )
        batch_size = int(output.shape[0])
        loss = self._diffusion_loss(
            model_target, condition, loss_weight, batch, stage="train"
        )

        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
        )

        return loss

    def on_after_backward(self):
        """Count and log completed backward passes, including accumulated ones."""
        self._backward_steps += 1
        self.log(
            "train/backward_steps",
            self._backward_steps,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            sync_dist=False,
        )

    def optimizer_step(self, *args, **kwargs):
        """Run the optimizer first, then update EMA from the new parameters."""
        super().optimizer_step(*args, **kwargs)
        if self.ema_model is None:
            return
        if int(self.global_step) < self.ema_update_after_step:
            self._copy_online_to_ema()
        else:
            self._update_ema()

    @torch.no_grad()
    def _copy_online_to_ema(self):
        if self.ema_model is None:
            return
        self.ema_model.load_state_dict(self.model.state_dict(), strict=True)
        self._ema_updates.add_(1)

    @torch.no_grad()
    def _update_ema(self):
        """Update the stored inference model from current trainable weights."""
        if self.ema_model is None or self.ema_decay is None:
            return
        decay = self.ema_decay
        for ema_parameter, parameter in zip(
            self.ema_model.parameters(), self.model.parameters()
        ):
            ema_parameter.lerp_(parameter.detach(), 1.0 - decay)
        for ema_buffer, buffer in zip(self.ema_model.buffers(), self.model.buffers()):
            ema_buffer.copy_(buffer.detach())
        self._ema_updates.add_(1)

    def on_save_checkpoint(self, checkpoint):
        """Persist the backward-pass counter across resumed training runs."""
        checkpoint["backward_steps"] = self._backward_steps

    def on_load_checkpoint(self, checkpoint):
        """Restore the counter while remaining compatible with older checkpoints."""
        self._backward_steps = int(checkpoint.get("backward_steps", 0))
        state_dict = checkpoint.get("state_dict", {})
        checkpoint_betas = state_dict.get("model.forward_process.betas")
        configured_betas = self.model.forward_process.betas
        if checkpoint_betas is not None and (
            checkpoint_betas.shape != configured_betas.shape
            or not torch.allclose(
                checkpoint_betas.to(configured_betas),
                configured_betas,
                rtol=1e-6,
                atol=1e-8,
            )
        ):
            raise ValueError(
                "Checkpoint diffusion coefficients do not match the configured "
                f"{self.schedule!r} schedule/timestep count. Resume with the "
                "checkpoint's original schedule, or start a fresh run for the "
                "new schedule."
            )
        if "_ema_updates" not in state_dict:
            state_dict["_ema_updates"] = self._ema_updates.clone()
        if self.ema_model is None:
            for key in list(state_dict):
                if key.startswith("ema_model."):
                    state_dict.pop(key)
        if self.ema_model is not None and not any(
            key.startswith("ema_model.") for key in state_dict
        ):
            # Allow EMA to be enabled when resuming a pre-EMA checkpoint. The
            # first EMA state is an exact copy of the trained online process.
            for key, value in list(state_dict.items()):
                if key.startswith("model."):
                    state_dict[f"ema_model.{key.removeprefix('model.')}"] = (
                        value.clone()
                    )

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """Lightning validation hook.

        Logs `val/loss` for scheduler control and, on the first validation batch,
        computes a full denoising reconstruction plus image/metric logging. The
        second loader contributes only a training-sample reconstruction image.
        """
        if dataloader_idx == 1:
            if self.log_reconstruction_images and batch_idx == 0:
                pred_batch = self.predict_step(batch, batch_idx, dataloader_idx)
                self._log_val_reconstruction(
                    batch, pred_batch, wandb_key="images/train_reconstruction"
                )
            return None
        input, output, target_mask = self._unpack_batch(batch)
        model_target, condition, loss_weight = self._prepare_training_inputs(
            input, output, target_mask, batch
        )
        batch_size = int(output.shape[0])
        loss = self._diffusion_loss(
            model_target, condition, loss_weight, batch, stage="val"
        )

        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            add_dataloader_idx=False,
            batch_size=batch_size,
        )

        if batch_idx < self.validation_reconstruction_batches:
            ensemble_predictions = None
            if (
                self.validation_ensemble_size > 1
                and batch_idx < self.validation_ensemble_batches
            ):
                raw_ensemble, ensemble_predictions = self.sample_ensemble(
                    batch,
                    batch_idx,
                    dataloader_idx,
                    num_samples=self.validation_ensemble_size,
                )
                raw_pred = raw_ensemble[0]
                pred_batch = ensemble_predictions[0]
                ensemble_metrics = self._ensemble_probabilistic_metrics(
                    ensemble_predictions, batch
                )
                for name, value in ensemble_metrics.items():
                    self.log(
                        f"val/{name}",
                        value,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=name == "probabilistic_refinement_score",
                        logger=True,
                        sync_dist=True,
                        add_dataloader_idx=False,
                        batch_size=batch_size,
                    )
                for member_index in range(
                    1, min(int(ensemble_predictions.shape[0]), 4)
                ):
                    if self.log_reconstruction_images:
                        self._log_val_reconstruction(
                            batch,
                            ensemble_predictions[member_index],
                            wandb_key=(f"images/val_ensemble_member_{member_index}"),
                        )
                if self.log_reconstruction_images:
                    self._log_val_reconstruction(
                        batch,
                        ensemble_predictions.mean(dim=0),
                        wandb_key="images/val_ensemble_mean",
                    )
            else:
                raw_pred, pred_batch = self._predict_batch(
                    batch, batch_idx, dataloader_idx
                )
            psnr, ssim, l1 = self._compute_reconstruction_metrics(
                pred_batch, output, target_mask
            )
            self.log(
                "val/psnr",
                psnr,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
                add_dataloader_idx=False,
                batch_size=batch_size,
            )
            self.log(
                "val/ssim",
                ssim,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
                add_dataloader_idx=False,
                batch_size=batch_size,
            )
            self.log(
                "val/l1",
                l1,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
                add_dataloader_idx=False,
                batch_size=batch_size,
            )
            self.log(
                "val/reconstruction_l1",
                l1,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
                add_dataloader_idx=False,
                batch_size=batch_size,
            )
            saturation = (raw_pred.abs() >= 1.0 - 1e-6).to(raw_pred.dtype).mean()
            self.log(
                "val/saturation_fraction",
                saturation,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
                add_dataloader_idx=False,
                batch_size=batch_size,
            )
            self.log(
                "val/raw_sample_min",
                raw_pred.min(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
                add_dataloader_idx=False,
                batch_size=batch_size,
            )
            self.log(
                "val/raw_sample_max",
                raw_pred.max(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
                add_dataloader_idx=False,
                batch_size=batch_size,
            )
            self._accumulate_storm_statistics(
                self._validation_storm_statistics,
                pred_batch,
                batch,
            )
            self._accumulate_physical_statistics(
                self._validation_physical_statistics,
                pred_batch,
                batch,
            )
            if self.log_reconstruction_images:
                self._log_val_reconstruction(
                    batch, pred_batch, wandb_key="images/val_reconstruction"
                )

        return loss

    def on_validation_epoch_start(self):
        self._validation_storm_statistics.zero_()
        self._validation_physical_statistics.zero_()

    def on_validation_epoch_end(self):
        self._log_physical_statistics("val", self._validation_physical_statistics)
        self._log_storm_statistics("val", self._validation_storm_statistics)

    def test_step(self, batch, batch_idx):
        """Evaluate held-out reconstructions on observed target pixels only."""
        input, output, target_mask = self._unpack_batch(batch)
        model_target, condition, loss_weight = self._prepare_training_inputs(
            input, output, target_mask, batch
        )
        batch_size = int(output.shape[0])
        loss = self._diffusion_loss(
            model_target, condition, loss_weight, batch, stage="test"
        )
        self.log(
            "test/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
        )

        pred_batch = self.predict_step(batch, batch_idx)
        self._accumulate_storm_statistics(
            self._test_storm_statistics,
            pred_batch,
            batch,
        )
        self._accumulate_physical_statistics(
            self._test_physical_statistics,
            pred_batch,
            batch,
        )
        psnr, ssim, l1 = self._compute_reconstruction_metrics(
            pred_batch, output, target_mask
        )
        self.log(
            "test/psnr",
            psnr,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "test/ssim",
            ssim,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "test/l1",
            l1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        return loss

    def on_test_epoch_start(self):
        self._test_storm_statistics.zero_()
        self._test_physical_statistics.zero_()

    def on_test_epoch_end(self):
        self._log_physical_statistics("test", self._test_physical_statistics)
        self._log_storm_statistics("test", self._test_storm_statistics)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Lightning predict hook that runs the full denoising chain.

        This uses `DenoisingDiffusionConditionalProcess.forward`, which starts from
        random noise and iteratively denoises to produce the final reconstruction.
        """
        _, prediction = self._predict_batch(batch, batch_idx, dataloader_idx)
        return prediction

    @torch.no_grad()
    def _predict_batch(self, batch, batch_idx, dataloader_idx=0, ensemble_index=0):
        """Sample one reproducible reconstruction for each batch item."""
        batch = self._prepare_batch_context(batch)
        input, _, _ = self._unpack_batch(batch)
        condition = self._prepare_condition(input, batch)
        process = (
            self.ema_model
            if self.ema_model is not None and self.ema_use_for_eval
            else self.model
        )
        process.eval()
        initial_noise = self._fixed_initial_noise(
            batch,
            batch_idx,
            dataloader_idx,
            condition,
            ensemble_index=ensemble_index,
        )
        unconditional_condition = (
            torch.zeros_like(condition) if self.guidance_scale != 1.0 else None
        )
        raw_prediction = process(
            condition,
            initial_noise=initial_noise,
            guidance_scale=self.guidance_scale,
            unconditional_condition=unconditional_condition,
        )
        return raw_prediction, self._sample_to_prediction(raw_prediction, batch)

    @torch.no_grad()
    def sample_ensemble(self, batch, batch_idx=0, dataloader_idx=0, num_samples=None):
        """Return stable raw and reconstructed ensembles as [K, B, C, H, W]."""
        count = int(num_samples or self.validation_ensemble_size)
        if count < 1:
            raise ValueError("num_samples must be positive")
        prepared = self._prepare_batch_context(batch)
        raw_members = []
        prediction_members = []
        for ensemble_index in range(count):
            raw, prediction = self._predict_batch(
                prepared,
                batch_idx,
                dataloader_idx,
                ensemble_index=ensemble_index,
            )
            raw_members.append(raw)
            prediction_members.append(prediction)
        return torch.stack(raw_members), torch.stack(prediction_members)

    def _prepare_training_inputs(
        self,
        condition,
        target,
        target_mask,
        batch,
    ):
        """Prepare a model-space target and condition for one loss call."""
        batch = self._prepare_batch_context(batch)
        diffusion_target, loss_weight = self._prepare_diffusion_target(
            target, target_mask, batch
        )
        return (
            self._target_to_model_space(diffusion_target, batch),
            self._prepare_condition(condition, batch),
            loss_weight,
        )

    def _prepare_batch_context(self, batch):
        """Hook for task variants that need a shared dense baseline."""
        return batch

    def _target_to_model_space(self, target, batch):
        """Map an absolute normalized target to diffusion's [-1, 1] space."""
        del batch
        return self.input_T(target)

    def _sample_to_prediction(self, sample, batch):
        """Map a raw diffusion sample to an absolute normalized prediction."""
        del batch
        return self.output_T(sample)

    def _fixed_initial_noise(
        self,
        batch,
        batch_idx,
        dataloader_idx,
        condition,
        ensemble_index=0,
    ):
        """Derive stable per-sample latents without changing global RNG state."""
        batch_size, _, height, width = condition.shape
        sample_ids = batch.get("sample_id") if isinstance(batch, dict) else None
        if isinstance(sample_ids, str):
            sample_ids = [sample_ids]
        noises = []
        for index in range(batch_size):
            if isinstance(sample_ids, (list, tuple)) and index < len(sample_ids):
                identifier = str(sample_ids[index])
            else:
                identifier = f"loader={dataloader_idx}:batch={batch_idx}:item={index}"
            latent_identifier = (
                identifier
                if ensemble_index == 0
                else f"{identifier}:ensemble={ensemble_index}"
            )
            digest = hashlib.sha256(
                f"{self.validation_seed}:{latent_identifier}".encode("utf-8")
            ).digest()
            seed = int.from_bytes(digest[:8], "little") % (2**63 - 1)
            generator = torch.Generator(device=condition.device)
            generator.manual_seed(seed)
            noises.append(
                torch.randn(
                    (1, self.model.generated_channels, height, width),
                    device=condition.device,
                    dtype=condition.dtype,
                    generator=generator,
                )
            )
        return torch.cat(noises, dim=0)

    def _prepare_diffusion_target(self, target, target_mask, batch):
        """Complete sparse SAR targets with a weakly supervised ERA5 anchor."""
        fill = str(self.sparse_target_fill or "none").strip().lower()
        if fill in {"", "none", "disabled"}:
            return target, target_mask
        if fill != "era5":
            raise ValueError("sparse_target_fill must be None or 'era5'")
        if not isinstance(batch, dict):
            raise KeyError("ERA5 sparse-target completion requires dictionary batches")
        required = {"era5_wind_speed", "era5_wind_speed_mask", "target_mask"}
        missing = sorted(required.difference(batch))
        if missing:
            raise KeyError(
                "ERA5 sparse-target completion requires: " + ", ".join(missing)
            )

        observed = self._prepare_target_mask(target_mask, target).bool()
        anchor = batch["era5_wind_speed"].to(device=target.device, dtype=target.dtype)
        anchor_valid = self._prepare_target_mask(
            batch["era5_wind_speed_mask"], target
        ).bool()
        neutral = torch.full_like(target, 0.5)
        dense_target = torch.where(observed, target, neutral)
        use_anchor = ~observed & anchor_valid
        dense_target = torch.where(use_anchor, anchor, dense_target)
        loss_weight = observed.to(target.dtype)
        if self.unobserved_loss_weight > 0:
            loss_weight = loss_weight + (
                use_anchor.to(target.dtype) * self.unobserved_loss_weight
            )
        return dense_target, loss_weight

    def _physical_reconstruction_metrics(self, prediction, batch):
        """Compute observed-pixel metrics in m/s using reversible target stats."""
        if not isinstance(batch, dict):
            return {}
        required = {
            "target_physical",
            "target_mask",
            "target_norm_offset",
            "target_norm_scale",
        }
        if not required.issubset(batch):
            return {}

        offset = batch["target_norm_offset"].to(
            device=prediction.device, dtype=prediction.dtype
        )
        scale = batch["target_norm_scale"].to(
            device=prediction.device, dtype=prediction.dtype
        )
        while offset.ndim < prediction.ndim:
            offset = offset.unsqueeze(-1)
            scale = scale.unsqueeze(-1)
        prediction_ms = prediction * scale + offset
        target_ms = batch["target_physical"].to(
            device=prediction.device, dtype=prediction.dtype
        )
        mask = self._prepare_target_mask(batch["target_mask"], prediction)
        count = mask.sum().clamp_min(1.0)
        error = prediction_ms - target_ms
        mae = (error.abs() * mask).sum() / count
        mse = (error.square() * mask).sum() / count
        metrics = {
            "reconstruction_mae_ms": mae,
            "reconstruction_rmse_ms": mse.sqrt(),
        }

        if {
            "era5_wind_speed_physical",
            "era5_wind_speed_mask",
        }.issubset(batch):
            era5_ms = batch["era5_wind_speed_physical"].to(
                device=prediction.device, dtype=prediction.dtype
            )
            era5_mask = self._prepare_target_mask(
                batch["era5_wind_speed_mask"], prediction
            )
            baseline_mask = mask * era5_mask
            baseline_count = baseline_mask.sum().clamp_min(1.0)
            era5_mae = (
                (era5_ms - target_ms).abs() * baseline_mask
            ).sum() / baseline_count
            reconstruction_baseline_mae = (
                error.abs() * baseline_mask
            ).sum() / baseline_count
            metrics["era5_mae_ms"] = era5_mae
            metrics["mae_skill_vs_era5"] = (
                1.0 - reconstruction_baseline_mae / era5_mae.clamp_min(1e-12)
            )
        return metrics

    def _ensemble_probabilistic_metrics(self, ensemble, batch):
        """Score a physical ensemble for calibration, diversity, and sharpness."""
        required = {
            "target_physical",
            "target_mask",
            "target_norm_offset",
            "target_norm_scale",
        }
        if not isinstance(batch, dict) or not required.issubset(batch):
            return {}
        offset = batch["target_norm_offset"].to(
            device=ensemble.device, dtype=ensemble.dtype
        )
        scale = batch["target_norm_scale"].to(
            device=ensemble.device, dtype=ensemble.dtype
        )
        while offset.ndim < ensemble.ndim - 1:
            offset = offset.unsqueeze(-1)
            scale = scale.unsqueeze(-1)
        ensemble_ms = ensemble * scale.unsqueeze(0) + offset.unsqueeze(0)
        target_ms = batch["target_physical"].to(
            device=ensemble.device, dtype=ensemble.dtype
        )
        mask = self._prepare_target_mask(batch["target_mask"], target_ms)
        count = mask.sum().clamp_min(1.0)
        member_count = ensemble_ms.shape[0]

        absolute_error = (ensemble_ms - target_ms.unsqueeze(0)).abs()
        observation_term = (absolute_error * mask.unsqueeze(0)).sum() / (
            member_count * count
        )
        pairwise = (ensemble_ms.unsqueeze(1) - ensemble_ms.unsqueeze(0)).abs()
        pairwise_mean = pairwise.mean(dim=(0, 1))
        diversity = (pairwise_mean * mask).sum() / count
        crps = observation_term - 0.5 * diversity

        ensemble_mean = ensemble_ms.mean(dim=0)
        mean_mae = ((ensemble_mean - target_ms).abs() * mask).sum() / count
        member_image_error = (absolute_error * mask.unsqueeze(0)).sum(
            dim=(-3, -2, -1)
        ) / mask.sum(dim=(-3, -2, -1)).clamp_min(1.0).unsqueeze(0)
        best_member_mae = member_image_error.min(dim=0).values.mean()
        spread = (ensemble_ms.std(dim=0, unbiased=False) * mask).sum() / count

        target_gradient, gradient_mask = self._gradient_magnitude(target_ms, mask)
        expanded_mask = mask.unsqueeze(0).expand(member_count, -1, -1, -1, -1)
        ensemble_gradient, _ = self._gradient_magnitude(
            ensemble_ms.reshape(-1, *ensemble_ms.shape[2:]),
            expanded_mask.reshape(-1, *mask.shape[1:]),
        )
        gradient_count = gradient_mask.sum().clamp_min(1.0)
        target_sharpness = (target_gradient * gradient_mask).sum() / gradient_count
        ensemble_gradient = ensemble_gradient.reshape(
            member_count, ensemble_ms.shape[1], *ensemble_gradient.shape[1:]
        )
        sampled_sharpness = (ensemble_gradient * gradient_mask.unsqueeze(0)).sum() / (
            member_count * gradient_count
        )
        sharpness_ratio = sampled_sharpness / target_sharpness.clamp_min(1e-6)
        spectrum_error = self._log_spectrum_error(ensemble_ms, target_ms, mask)
        # A target below one deliberately selects slightly smoother members.
        # Keeping this configurable also preserves exact target matching as the
        # default for existing experiments.
        relative_sharpness = (
            sharpness_ratio / self.probabilistic_score_target_sharpness_ratio
        )
        sharpness_penalty = relative_sharpness.clamp_min(1e-6).log().abs()
        peak_metrics = self._ensemble_robust_peak_metrics(
            ensemble_ms,
            target_ms,
            mask,
            fraction=getattr(self, "probabilistic_peak_fraction", 0.005),
        )
        refinement_score = (
            crps
            + self.probabilistic_score_sharpness_weight
            * (sharpness_penalty + 0.1 * spectrum_error)
            + getattr(self, "probabilistic_score_peak_weight", 0.0)
            * peak_metrics["ensemble_robust_peak_crps_ms"]
        )
        return {
            "ensemble_crps_ms": crps,
            "ensemble_spread_ms": spread,
            "ensemble_diversity_ms": diversity,
            "ensemble_mean_mae_ms": mean_mae,
            "ensemble_best_member_mae_ms": best_member_mae,
            "ensemble_sharpness_ratio": sharpness_ratio,
            "ensemble_log_spectrum_error": spectrum_error,
            **peak_metrics,
            "probabilistic_refinement_score": refinement_score,
        }

    @staticmethod
    def _ensemble_robust_peak_metrics(ensemble_ms, target_ms, mask, *, fraction):
        """Score the distribution of member-wise stable inner-field peaks."""
        member_count, batch_size = ensemble_ms.shape[:2]
        member_peaks = []
        target_peaks = []
        for batch_index in range(batch_size):
            valid = mask[batch_index].bool().flatten()
            count = int(valid.sum().detach())
            if count == 0:
                continue
            top_count = max(1, math.ceil(count * float(fraction)))
            target_values = target_ms[batch_index].flatten()[valid]
            target_peaks.append(
                torch.topk(target_values, top_count, sorted=False).values.mean()
            )
            sample_peaks = []
            for member_index in range(member_count):
                member_values = ensemble_ms[member_index, batch_index].flatten()[valid]
                sample_peaks.append(
                    torch.topk(member_values, top_count, sorted=False).values.mean()
                )
            member_peaks.append(torch.stack(sample_peaks))
        if not target_peaks:
            zero = ensemble_ms.sum() * 0.0
            return {
                "ensemble_robust_peak_crps_ms": zero,
                "ensemble_robust_peak_median_mae_ms": zero,
                "ensemble_robust_peak_median_bias_ms": zero,
                "ensemble_robust_peak_10_90_coverage": zero,
            }
        peaks = torch.stack(member_peaks, dim=1)
        target_peak = torch.stack(target_peaks)
        observation = (peaks - target_peak.unsqueeze(0)).abs().mean()
        pairwise = (peaks.unsqueeze(1) - peaks.unsqueeze(0)).abs().mean()
        peak_crps = observation - 0.5 * pairwise
        # `Tensor.median(dim=...)` selects the lower middle member for an
        # even ensemble and its CUDA indices kernel is not deterministic.
        median = torch.quantile(peaks, 0.5, dim=0, interpolation="linear")
        median_error = median - target_peak
        lower = torch.quantile(peaks, 0.1, dim=0)
        upper = torch.quantile(peaks, 0.9, dim=0)
        coverage = (
            ((target_peak >= lower) & (target_peak <= upper))
            .to(ensemble_ms.dtype)
            .mean()
        )
        return {
            "ensemble_robust_peak_crps_ms": peak_crps,
            "ensemble_robust_peak_median_mae_ms": median_error.abs().mean(),
            "ensemble_robust_peak_median_bias_ms": median_error.mean(),
            "ensemble_robust_peak_10_90_coverage": coverage,
        }

    @staticmethod
    def _gradient_magnitude(values, mask):
        dx = F.pad(values[..., 1:] - values[..., :-1], (0, 1, 0, 0))
        dy = F.pad(values[..., 1:, :] - values[..., :-1, :], (0, 0, 0, 1))
        valid_x = F.pad(mask[..., 1:] * mask[..., :-1], (0, 1, 0, 0))
        valid_y = F.pad(mask[..., 1:, :] * mask[..., :-1, :], (0, 0, 0, 1))
        valid = valid_x * valid_y
        return (dx.square() + dy.square() + 1e-12).sqrt(), valid

    @staticmethod
    def _log_spectrum_error(ensemble_ms, target_ms, mask):
        count = mask.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
        target_mean = (target_ms * mask).sum(dim=(-2, -1), keepdim=True) / count
        target_centered = (target_ms - target_mean) * mask
        ensemble_mask = mask.unsqueeze(0)
        ensemble_count = count.unsqueeze(0)
        ensemble_mean = (ensemble_ms * ensemble_mask).sum(
            dim=(-2, -1), keepdim=True
        ) / ensemble_count
        ensemble_centered = (ensemble_ms - ensemble_mean) * ensemble_mask
        target_spectrum = torch.fft.rfft2(target_centered, norm="ortho").abs()
        ensemble_spectrum = torch.fft.rfft2(ensemble_centered, norm="ortho").abs()
        return (
            (torch.log1p(ensemble_spectrum) - torch.log1p(target_spectrum).unsqueeze(0))
            .abs()
            .mean()
        )

    def _accumulate_storm_statistics(self, statistics, prediction, batch):
        """Accumulate fixed-shape eye/high-wind statistics without DDP branches."""
        required = {
            "target_physical",
            "target_mask",
            "target_norm_offset",
            "target_norm_scale",
        }
        if not isinstance(batch, dict) or not required.issubset(batch):
            return
        offset = batch["target_norm_offset"].to(
            device=prediction.device, dtype=prediction.dtype
        )
        scale = batch["target_norm_scale"].to(
            device=prediction.device, dtype=prediction.dtype
        )
        while offset.ndim < prediction.ndim:
            offset = offset.unsqueeze(-1)
            scale = scale.unsqueeze(-1)
        prediction_ms = prediction * scale + offset
        target_ms = batch["target_physical"].to(
            device=prediction.device, dtype=prediction.dtype
        )
        mask = self._prepare_target_mask(batch["target_mask"], prediction)
        high_wind_mask = mask * (target_ms >= 17.0).to(mask.dtype)
        high_wind = torch.stack(
            [
                ((prediction_ms - target_ms).abs() * high_wind_mask).sum(),
                high_wind_mask.sum(),
            ]
        ).unsqueeze(0)
        radial = prediction.new_zeros((len(RADIAL_METRIC_NAMES), 2))
        if {"center", "target_bounds"}.issubset(batch):
            radial = radial_wind_metric_statistics(
                prediction_ms,
                target_ms,
                mask,
                batch["center"],
                batch["target_bounds"],
            )
        additions = torch.cat([high_wind, radial], dim=0)
        statistics.add_(additions.to(statistics))

    def _accumulate_physical_statistics(self, statistics, prediction, batch):
        """Accumulate pixel-weighted physical errors for exact epoch metrics."""
        required = {
            "target_physical",
            "target_mask",
            "target_norm_offset",
            "target_norm_scale",
        }
        if not isinstance(batch, dict) or not required.issubset(batch):
            return
        offset = batch["target_norm_offset"].to(
            device=prediction.device, dtype=prediction.dtype
        )
        scale = batch["target_norm_scale"].to(
            device=prediction.device, dtype=prediction.dtype
        )
        while offset.ndim < prediction.ndim:
            offset = offset.unsqueeze(-1)
            scale = scale.unsqueeze(-1)
        prediction_ms = prediction * scale + offset
        target_ms = batch["target_physical"].to(
            device=prediction.device, dtype=prediction.dtype
        )
        mask = self._prepare_target_mask(batch["target_mask"], prediction)
        error = prediction_ms - target_ms
        additions = prediction.new_zeros(6)
        additions[0] = mask.sum()
        additions[1] = (error.abs() * mask).sum()
        additions[2] = (error.square() * mask).sum()
        if {
            "era5_wind_speed_physical",
            "era5_wind_speed_mask",
        }.issubset(batch):
            era5_ms = batch["era5_wind_speed_physical"].to(
                device=prediction.device, dtype=prediction.dtype
            )
            era5_mask = self._prepare_target_mask(
                batch["era5_wind_speed_mask"], prediction
            )
            baseline_mask = mask * era5_mask
            additions[3] = baseline_mask.sum()
            additions[4] = (error.abs() * baseline_mask).sum()
            additions[5] = ((era5_ms - target_ms).abs() * baseline_mask).sum()
        statistics.add_(additions.to(statistics))

    def _log_physical_statistics(self, prefix, statistics):
        statistics = self._distributed_sum_statistics(statistics)
        count = statistics[0]
        if count <= 0:
            return
        mae = statistics[1] / count
        metrics = {
            "reconstruction_mae_ms": mae,
            "reconstruction_rmse_ms": (statistics[2] / count).sqrt(),
        }
        baseline_count = statistics[3]
        if baseline_count > 0:
            reconstruction_baseline_mae = statistics[4] / baseline_count
            era5_mae = statistics[5] / baseline_count
            metrics["era5_mae_ms"] = era5_mae
            metrics["mae_skill_vs_era5"] = (
                1.0 - reconstruction_baseline_mae / era5_mae.clamp_min(1e-12)
            )
        for name, value in metrics.items():
            self.log(
                f"{prefix}/{name}",
                value.to(torch.float32),
                on_step=False,
                on_epoch=True,
                prog_bar=name == "reconstruction_mae_ms",
                logger=True,
                sync_dist=False,
            )

    def _log_storm_statistics(self, prefix, statistics):
        statistics = self._distributed_sum_statistics(statistics)
        means = {}
        for index, name in enumerate(STORM_METRIC_NAMES):
            count = statistics[index, 1]
            if count <= 0:
                continue
            value = (statistics[index, 0] / count).to(torch.float32)
            means[name] = value
            self.log(
                f"{prefix}/{name}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=False,
            )
        score_terms = {
            "eye_mae_ms": 0.5,
            "inner_core_mae_ms": 1.0,
            "radial_profile_mae_ms": 1.0,
            "rmw_error_km": 0.1,
            "eye_to_eyewall_contrast_error_ms": 1.0,
        }
        if score_terms.keys() <= means.keys():
            score = sum(means[name] * weight for name, weight in score_terms.items())
            self.log(
                f"{prefix}/eye_structure_score",
                score,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=False,
            )

    def _distributed_sum_statistics(self, statistics):
        trainer = getattr(self, "_trainer", None)
        if trainer is None or trainer.world_size <= 1:
            return statistics
        gathered = self.all_gather(statistics)
        return gathered.reshape(-1, *statistics.shape).sum(dim=0)

    def configure_optimizers(self):
        """Create optimizer and scheduler driven by deterministic reconstruction."""
        optimizer = torch.optim.AdamW(
            list(filter(lambda p: p.requires_grad, self.model.parameters())),
            lr=self.lr,
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.lr_scheduler_factor,
            patience=self.lr_scheduler_patience,
            cooldown=self.lr_scheduler_cooldown,
            min_lr=self.lr_scheduler_min_lr,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self.lr_scheduler_monitor,
            },
        }

    def _to_plot_image(self, tensor):
        """Convert CHW tensor to a matplotlib-friendly image array."""
        image = tensor.detach().float().cpu().clamp(0, 1)
        if image.shape[0] >= 3:
            return image[:3].permute(1, 2, 0).numpy(), None
        return image[0].numpy(), "gray"

    def _compute_reconstruction_metrics(
        self,
        pred_batch,
        target_batch,
        target_mask=None,
    ):
        """Compute batch-level reconstruction metrics on [0, 1] tensors."""
        pred = pred_batch.detach().float().clamp(0, 1)
        target = target_batch.detach().float().clamp(0, 1)

        mask = (
            self._prepare_target_mask(target_mask, pred)
            if target_mask is not None
            else None
        )
        psnr_vals = []
        ssim_vals = []

        if mask is None:
            l1 = F.l1_loss(pred, target)
            mse = F.mse_loss(pred, target)
            pred_np = pred.cpu().numpy()
            target_np = target.cpu().numpy()
            for i in range(pred_np.shape[0]):
                psnr_vals.append(
                    peak_signal_noise_ratio(target_np[i], pred_np[i], data_range=1.0)
                )
                ssim_vals.append(
                    structural_similarity(
                        target_np[i],
                        pred_np[i],
                        data_range=1.0,
                        channel_axis=0,
                    )
                )
        else:
            valid = mask.sum().clamp_min(1.0)
            l1 = ((pred - target).abs() * mask).sum() / valid
            mse = ((pred - target).pow(2) * mask).sum() / valid
            psnr_vals.append(
                10.0 * torch.log10(1.0 / mse.clamp_min(1e-12)).detach().cpu().item()
            )
            ssim_vals = self._masked_ssim_values(pred, target, mask)

        psnr = torch.tensor(psnr_vals, device=pred.device, dtype=pred.dtype).mean()
        if not torch.isfinite(psnr):
            psnr = 10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))
        ssim = torch.tensor(ssim_vals, device=pred.device, dtype=pred.dtype).mean()

        return psnr, ssim, l1

    def _unpack_batch(self, batch):
        """Support both legacy tuple batches and GeoTIFF dict batches."""
        if isinstance(batch, dict):
            return (
                batch["condition"],
                batch["target"],
                batch.get("target_mask"),
            )
        input, output = batch
        return input, output, None

    def _prepare_condition(self, condition, batch):
        """Normalize GEO bands and append a binary valid-pixel channel."""
        condition = self.input_T(condition)
        if not isinstance(batch, dict) or batch.get("condition_mask") is None:
            return condition
        condition_mask = batch["condition_mask"].to(
            device=condition.device, dtype=condition.dtype
        )
        if condition_mask.ndim == 3:
            condition_mask = condition_mask.unsqueeze(1)
        if condition_mask.shape[1] != 1:
            condition_mask = condition_mask.all(dim=1, keepdim=True)
        return torch.cat([condition, condition_mask], dim=1)

    def _prepare_target_mask(self, target_mask, reference):
        """Broadcast a target-validity mask to match a BCHW target tensor."""
        mask = target_mask.detach().to(device=reference.device, dtype=reference.dtype)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        if mask.shape[1] == 1 and reference.shape[1] != 1:
            mask = mask.expand_as(reference)
        return mask

    def _masked_ssim_values(self, pred, target, mask):
        """Compute SSIM only where the complete SSIM window is observed."""
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        mask_np = mask.detach().cpu().numpy().astype(bool)
        ssim_vals = []
        for i in range(pred_np.shape[0]):
            valid = mask_np[i]
            if not valid.any():
                continue
            # Use the same neutral fill for prediction and truth, then discard
            # windows touching that fill. This avoids replacing missing
            # predictions with ground truth, which inflated the old score.
            p = np.where(valid, pred_np[i], 0.0)
            t = np.where(valid, target_np[i], 0.0)
            _, ssim_image = structural_similarity(
                t,
                p,
                data_range=1.0,
                channel_axis=0,
                full=True,
            )
            spatial_valid = valid.all(axis=0)
            full_window_valid = binary_erosion(
                spatial_valid,
                structure=np.ones((7, 7), dtype=bool),
                border_value=0,
            )
            if not full_window_valid.any():
                continue
            if ssim_image.ndim == 3:
                score_values = ssim_image[:, full_window_valid]
            else:
                score_values = ssim_image[full_window_valid]
            ssim_vals.append(float(score_values.mean()))
        return ssim_vals or [0.0]

    def _log_val_reconstruction(
        self, batch, pred_batch, *, wandb_key="images/val_reconstruction"
    ):
        """Log up to five stretched, georeferenced reconstructions."""
        input_batch, target_batch, _ = self._unpack_batch(batch)
        log_wandb_reconstruction(
            self,
            batch,
            pred_batch,
            wandb_key=wandb_key,
            condition_batch=input_batch,
            target_batch=target_batch,
        )
