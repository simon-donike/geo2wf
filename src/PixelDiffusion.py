import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/dif_img_rec_matplotlib")

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from .DenoisingDiffusionProcess import *

class PixelDiffusionConditional(pl.LightningModule):
    """Conditional pixel-space diffusion Lightning module.

    Expects each batch to provide:
    - `condition`: conditional tensor
    - `target`: target tensor to reconstruct/generate
    - `target_mask`: optional target-validity mask
    """
    def __init__(self,
                 condition_channels=3,
                 generated_channels=3,
                 num_timesteps=1000,
                 schedule='linear',
                 model_dim=64,
                 model_dim_mults=(1,2,4,8),
                 model_channels=None,
                 model_out_dim=None,
                 lr=1e-3,
                 lr_scheduler_factor=0.5,
                 lr_scheduler_patience=25):
        super().__init__()
        self.lr = lr
        self.lr_scheduler_factor=lr_scheduler_factor
        self.lr_scheduler_patience=lr_scheduler_patience
        self._backward_steps = 0
        
        # Core conditional diffusion process used by training, validation, and prediction.
        self.model=DenoisingDiffusionConditionalProcess(generated_channels=generated_channels,
                                                        condition_channels=condition_channels,
                                                        schedule=schedule,
                                                        num_timesteps=num_timesteps,
                                                        model_dim=model_dim,
                                                        model_dim_mults=model_dim_mults,
                                                        model_channels=model_channels,
                                                        model_out_dim=model_out_dim)

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        """Lightning inference helper; returns output mapped back to [0, 1]."""
        return self.output_T(self.model(*args, **kwargs))

    def input_T(self, input):
        # Model internally expects values in [-1, 1].
        return input.clamp(0, 1).mul(2).sub(1)

    def output_T(self, input):
        # Inverse mapping from [-1, 1] back to [0, 1] for visualization/metrics.
        return input.add(1).div(2)
    
    def training_step(self, batch, batch_idx):   
        """Lightning train hook for conditional diffusion."""
        input, output, target_mask = self._unpack_batch(batch)
        loss = self.model.p_loss(
            self.input_T(output),
            self._prepare_condition(input, batch),
            mask=target_mask,
        )
        
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
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

    def on_save_checkpoint(self, checkpoint):
        """Persist the backward-pass counter across resumed training runs."""
        checkpoint["backward_steps"] = self._backward_steps

    def on_load_checkpoint(self, checkpoint):
        """Restore the counter while remaining compatible with older checkpoints."""
        self._backward_steps = int(checkpoint.get("backward_steps", 0))

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """Lightning validation hook.

        Logs `val/loss` for scheduler control and, on the first validation batch,
        computes a full denoising reconstruction plus image/metric logging. The
        second loader contributes only a training-sample reconstruction image.
        """
        if dataloader_idx == 1:
            if batch_idx == 0:
                pred_batch = self.predict_step(batch, batch_idx, dataloader_idx)
                self._log_val_reconstruction(
                    batch, pred_batch, wandb_key="images/train_reconstruction"
                )
            return None
        input, output, target_mask = self._unpack_batch(batch)
        loss = self.model.p_loss(
            self.input_T(output),
            self._prepare_condition(input, batch),
            mask=target_mask,
        )
        
        self.log(
            'val/loss', loss, on_step=False, on_epoch=True, prog_bar=True,
            logger=True, sync_dist=True, add_dataloader_idx=False
        )

        if batch_idx == 0:
            pred_batch = self.predict_step(batch, batch_idx)
            psnr, ssim, l1 = self._compute_reconstruction_metrics(
                pred_batch, output, target_mask
            )
            self.log(
                'val/psnr', psnr, on_step=False, on_epoch=True,
                prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False
            )
            self.log(
                'val/ssim', ssim, on_step=False, on_epoch=True,
                prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False
            )
            self.log(
                'val/l1', l1, on_step=False, on_epoch=True,
                prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False
            )
            self._log_val_reconstruction(
                batch, pred_batch, wandb_key="images/val_reconstruction"
            )
        
        return loss

    def test_step(self, batch, batch_idx):
        """Evaluate held-out reconstructions on observed target pixels only."""
        input, output, target_mask = self._unpack_batch(batch)
        loss = self.model.p_loss(
            self.input_T(output),
            self._prepare_condition(input, batch),
            mask=target_mask,
        )
        self.log('test/loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        pred_batch = self.predict_step(batch, batch_idx)
        psnr, ssim, l1 = self._compute_reconstruction_metrics(
            pred_batch, output, target_mask
        )
        self.log('test/psnr', psnr, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test/ssim', ssim, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test/l1', l1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Lightning predict hook that runs the full denoising chain.

        This uses `DenoisingDiffusionConditionalProcess.forward`, which starts from
        random noise and iteratively denoises to produce the final reconstruction.
        """
        del batch_idx, dataloader_idx
        input, _, _ = self._unpack_batch(batch)
        pred = self.model(self._prepare_condition(input, batch))
        return self.output_T(pred)

    def configure_optimizers(self):
        """Create optimizer and ReduceLROnPlateau scheduler monitored on `val/loss`."""
        optimizer = torch.optim.AdamW(
            list(filter(lambda p: p.requires_grad, self.model.parameters())),
            lr=self.lr,
        )
        scheduler = ReduceLROnPlateau(optimizer,
                                      mode='min',
                                      factor=self.lr_scheduler_factor,
                                      patience=self.lr_scheduler_patience)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler,
                                 "monitor": "val/loss"}}

    def _to_plot_image(self, tensor):
        """Convert CHW tensor to a matplotlib-friendly image array."""
        image = tensor.detach().float().cpu().clamp(0, 1)
        if image.shape[0] >= 3:
            return image[:3].permute(1, 2, 0).numpy(), None
        return image[0].numpy(), 'gray'

    def _compute_reconstruction_metrics(
        self,
        pred_batch,
        target_batch,
        target_mask=None,
    ):
        """Compute batch-level reconstruction metrics on [0, 1] tensors."""
        pred = pred_batch.detach().float().clamp(0, 1)
        target = target_batch.detach().float().clamp(0, 1)

        mask = self._prepare_target_mask(target_mask, pred) if target_mask is not None else None
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
        """Compute SSIM from the observed SAR pixels only."""
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        mask_np = mask.detach().cpu().numpy().astype(bool)
        ssim_vals = []
        for i in range(pred_np.shape[0]):
            valid = mask_np[i]
            if not valid.any():
                continue
            p = pred_np[i].copy()
            t = target_np[i]
            p[~valid] = t[~valid]
            _, ssim_image = structural_similarity(
                t,
                p,
                data_range=1.0,
                channel_axis=0,
                full=True,
            )
            ssim_vals.append(float(ssim_image[valid].mean()))
        return ssim_vals or [0.0]

    def _log_val_reconstruction(
        self, batch, pred_batch, *, wandb_key="images/val_reconstruction"
    ):
        """Log up to five stretched, georeferenced reconstructions with no-data masks."""
        if self.logger is None or self.trainer is None or not self.trainer.is_global_zero:
            return
        try:
            import matplotlib.pyplot as plt
            import wandb
            from utils.plotting import plot_validation_reconstruction_batch
        except ImportError:
            return

        sample_count = min(int(pred_batch.shape[0]), 5)
        samples = []
        if not isinstance(batch, dict):
            input_batch, target_batch, _ = self._unpack_batch(batch)
            for index in range(sample_count):
                samples.append({
                    "condition": input_batch[index],
                    "prediction": pred_batch[index],
                    "target": target_batch[index],
                })
        else:
            meta = batch.get("meta", {})
            for index in range(sample_count):
                label = " · ".join(value for value in (
                    self._batch_value(meta.get("storm_id"), index),
                    self._batch_value(batch.get("sample_id"), index),
                ) if value)
                samples.append({
                    "condition": batch["condition"][index],
                    "prediction": pred_batch[index],
                    "target": batch["target"][index],
                    "condition_mask": self._batch_item(batch.get("condition_mask"), index),
                    "target_mask": self._batch_item(batch.get("target_mask"), index),
                    "condition_channels": self._channel_names(
                        meta.get("condition_channels"), index
                    ),
                    "condition_bounds": self._batch_item(
                        batch.get("condition_bounds"), index
                    ),
                    "target_bounds": self._batch_item(batch.get("target_bounds"), index),
                    "center": self._finite_pair(batch.get("center"), index),
                    "sample_label": label,
                })
        fig = plot_validation_reconstruction_batch(samples)
        # Keep validation media lightweight: cap the longest rendered edge and
        # use JPEG instead of W&B's lossless PNG default.
        max_edge_pixels = 1600
        width_inches, height_inches = fig.get_size_inches()
        max_dpi = max_edge_pixels / max(width_inches, height_inches)
        fig.set_dpi(min(float(fig.dpi), max_dpi))
        try:
            self.logger.experiment.log(
                {wandb_key: wandb.Image(fig, file_type="jpg")},
                step=self.global_step,
            )
        finally:
            plt.close(fig)

    @staticmethod
    def _batch_item(value, index):
        if value is None:
            return None
        item = value[index]
        if torch.is_tensor(item):
            item = item.detach().cpu()
        return item

    @staticmethod
    def _batch_value(value, index):
        if value is None:
            return ""
        if isinstance(value, (list, tuple)):
            return str(value[index]) if index < len(value) else ""
        return str(value)

    @staticmethod
    def _channel_names(value, index):
        if not isinstance(value, (list, tuple)):
            return None
        # Default collation transposes each sample's channel list by band.
        return [
            str(item[index] if isinstance(item, (list, tuple)) else item)
            for item in value
        ]

    @staticmethod
    def _finite_pair(value, index):
        if value is None:
            return None
        pair = value[index].detach().double().cpu()
        if pair.numel() != 2 or not torch.isfinite(pair).all():
            return None
        return float(pair[0]), float(pair[1])
