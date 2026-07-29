import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm.auto import tqdm

from .forward import *
from .samplers import *
from .backbones.unet_convnext import *


def _prepare_sampler(default_sampler, sampler, forward_process, device):
    """Move a sampler to the model device and enforce train/sampler agreement."""
    selected = default_sampler if sampler is None else sampler
    selected = selected.to(device)

    required = ("timesteps", "train_timesteps", "schedule", "betas")
    missing = [name for name in required if not hasattr(selected, name)]
    if missing:
        raise TypeError(
            "Sampler is missing required sampling metadata: " + ", ".join(missing)
        )
    if int(selected.train_timesteps) != int(forward_process.num_timesteps):
        raise ValueError(
            "Sampler train_timesteps does not match the trained diffusion process: "
            f"{selected.train_timesteps} != {forward_process.num_timesteps}"
        )
    if selected.schedule != forward_process.schedule:
        raise ValueError(
            "Sampler schedule does not match the trained diffusion process: "
            f"{selected.schedule!r} != {forward_process.schedule!r}"
        )
    if (
        selected.betas.shape != forward_process.betas.shape
        or not torch.equal(selected.betas, forward_process.betas)
    ):
        raise ValueError("Sampler beta coefficients do not match the forward process")
    if selected.timesteps.ndim != 1 or selected.timesteps.numel() < 1:
        raise ValueError("Sampler timesteps must be a non-empty one-dimensional tensor")
    return selected


def _prepare_initial_noise(
    shape,
    *,
    device,
    dtype,
    initial_noise=None,
    generator=None,
):
    """Create or validate the reverse chain's initial latent."""
    expected_shape = tuple(int(value) for value in shape)
    if initial_noise is None:
        return torch.randn(
            expected_shape,
            device=device,
            dtype=dtype,
            generator=generator,
        )
    if tuple(initial_noise.shape) != expected_shape:
        raise ValueError(
            f"initial_noise has shape {tuple(initial_noise.shape)}, "
            f"expected {expected_shape}"
        )
    return initial_noise.detach().to(device=device, dtype=dtype).clone()


def _run_reverse_process(
    model,
    x_t,
    sampler,
    *,
    condition=None,
    unconditional_condition=None,
    guidance_scale=1.0,
    generator=None,
    verbose=False,
):
    """Run one sampler schedule with shared conditional/unconditional dispatch."""
    guidance_scale = float(guidance_scale)
    if guidance_scale < 0:
        raise ValueError("guidance_scale must be non-negative")
    if condition is None and unconditional_condition is not None:
        raise ValueError("unconditional_condition requires a condition")
    if unconditional_condition is not None and (
        unconditional_condition.shape != condition.shape
    ):
        raise ValueError(
            "unconditional_condition must have the same shape as condition"
        )
    timestep_values = sampler.timesteps.detach().cpu().tolist()
    iterator = (
        tqdm(timestep_values, desc="diffusion sampling", total=len(timestep_values))
        if verbose
        else timestep_values
    )
    for timestep in iterator:
        t = torch.full(
            (x_t.shape[0],), int(timestep), device=x_t.device, dtype=torch.long
        )
        model_input = x_t if condition is None else torch.cat([x_t, condition], dim=1)
        noise_pred = model(model_input, t)
        if (
            condition is not None
            and unconditional_condition is not None
            and guidance_scale != 1.0
        ):
            unconditional_input = torch.cat(
                [x_t, unconditional_condition], dim=1
            )
            unconditional_noise = model(unconditional_input, t)
            noise_pred = unconditional_noise + guidance_scale * (
                noise_pred - unconditional_noise
            )
        x_t = sampler(x_t, t, noise_pred, generator=generator)
    return x_t


class DenoisingDiffusionProcess(nn.Module):
    
    def __init__(self,
                 generated_channels=3,              
                 loss_fn=F.mse_loss,
                 schedule='linear',
                 num_timesteps=1000,
                 sampler=None,
                 model_dim=64,
                 model_dim_mults=(1,2,4,8),
                 model_channels=None,
                 model_out_dim=None
                ):
        super().__init__()
        
        # Basic Params
        self.generated_channels=generated_channels
        self.num_timesteps=num_timesteps
        self.loss_fn=loss_fn
        
        # Forward Process Used for Training
        self.forward_process=GaussianForwardProcess(num_timesteps=self.num_timesteps,
                                                    schedule=schedule)
        if model_channels is None:
            model_channels=self.generated_channels
        if model_out_dim is None:
            model_out_dim=self.generated_channels
        self.model=UnetConvNextBlock(dim=model_dim,
                                     dim_mults=model_dim_mults,
                                     channels=model_channels,
                                     out_dim=model_out_dim,
                                     with_time_emb=True)
               
        
        # defaults to a DDPM sampler if None is provided
        self.sampler = (
            DDPM_Sampler(
                num_timesteps=self.num_timesteps,
                schedule=schedule,
                clip_sample=True,
            )
            if sampler is None
            else sampler
        )
        
    @torch.no_grad()
    def forward(
        self,
        shape=(256, 256),
        batch_size=1,
        sampler=None,
        verbose=False,
        initial_noise=None,
        generator=None,
    ):
        """Run a complete unconditional reverse process.

        Samplers expose actual training timesteps, so the model time embedding
        and sampler coefficients always receive exactly the same timestep.
        """
        b, h, w = batch_size, *shape
        model_parameter = next(self.model.parameters())
        device = model_parameter.device
        sampler = _prepare_sampler(
            self.sampler, sampler, self.forward_process, device
        )
        x_t = _prepare_initial_noise(
            (b, self.generated_channels, h, w),
            device=device,
            dtype=model_parameter.dtype,
            initial_noise=initial_noise,
            generator=generator,
        )
        return _run_reverse_process(
            self.model,
            x_t,
            sampler,
            generator=generator,
            verbose=verbose,
        )
        
    def p_loss(self,output):
        """
            Assumes output is in [-1,+1] range
        """        
        
        b,c,h,w=output.shape
        device=output.device
        
        # loss for training
        
        # input is the optional condition
        t = torch.randint(0, self.forward_process.num_timesteps, (b,), device=device).long()
        output_noisy, noise=self.forward_process(output,t,return_noise=True)        

        # reverse pass
        noise_hat = self.model(output_noisy, t) 

        # apply loss
        return self.loss_fn(noise, noise_hat)
    
    
class DenoisingDiffusionConditionalProcess(nn.Module):
    
    def __init__(self,
                 generated_channels=3,
                 condition_channels=3,
                 loss_fn=F.mse_loss,
                 schedule='linear',
                 num_timesteps=1000,
                 sampler=None,
                 model_dim=64,
                 model_dim_mults=(1,2,4,8),
                 model_channels=None,
                 model_out_dim=None
                ):
        super().__init__()
        
        # Basic Params
        self.generated_channels=generated_channels
        self.condition_channels=condition_channels
        self.num_timesteps=num_timesteps
        self.loss_fn=loss_fn
        
        # Forward Process
        self.forward_process=GaussianForwardProcess(num_timesteps=self.num_timesteps,
                                                    schedule=schedule)
        
        # Neural Network Backbone
        if model_channels is None:
            model_channels=self.generated_channels + condition_channels
        if model_out_dim is None:
            model_out_dim=self.generated_channels
        self.model=UnetConvNextBlock(dim=model_dim,
                                     dim_mults=model_dim_mults,
                                     channels=model_channels,
                                     out_dim=model_out_dim,
                                     with_time_emb=True)
        
        # defaults to a DDPM sampler if None is provided
        self.sampler = (
            DDPM_Sampler(
                num_timesteps=self.num_timesteps,
                schedule=schedule,
                clip_sample=True,
            )
            if sampler is None
            else sampler
        )
        
    @torch.no_grad()
    def forward(
        self,
        condition,
        sampler=None,
        verbose=False,
        initial_noise=None,
        generator=None,
        guidance_scale=1.0,
        unconditional_condition=None,
    ):
        """Run a complete conditional reverse process.

        ``initial_noise`` makes deterministic DDIM sampling reproducible without
        touching global RNG state. For stochastic samplers, pass a same-device
        ``torch.Generator`` to control both initialization and reverse noise.
        """
        b, _, h, w = condition.shape
        model_parameter = next(self.model.parameters())
        device = model_parameter.device
        condition = condition.to(device=device, dtype=model_parameter.dtype)
        if unconditional_condition is not None:
            unconditional_condition = unconditional_condition.to(
                device=device, dtype=model_parameter.dtype
            )
        sampler = _prepare_sampler(
            self.sampler, sampler, self.forward_process, device
        )
        x_t = _prepare_initial_noise(
            (b, self.generated_channels, h, w),
            device=device,
            dtype=condition.dtype,
            initial_noise=initial_noise,
            generator=generator,
        )
        return _run_reverse_process(
            self.model,
            x_t,
            sampler,
            condition=condition,
            unconditional_condition=unconditional_condition,
            guidance_scale=guidance_scale,
            generator=generator,
            verbose=verbose,
        )
        
    def diffusion_terms(self, output, condition, t=None):
        """Return one noisy training problem and its clean-sample estimate."""
        batch_size = output.shape[0]
        device = output.device
        if t is None:
            t = torch.randint(
                0,
                self.forward_process.num_timesteps,
                (batch_size,),
                device=device,
            ).long()
        elif t.shape != (batch_size,):
            raise ValueError(f"t must have shape ({batch_size},)")
        output_noisy, noise = self.forward_process(
            output, t, return_noise=True
        )
        model_input = torch.cat([output_noisy, condition], 1).to(device)
        noise_hat = self.model(model_input, t)
        alpha_bar = self.forward_process.alphas_cumprod[t].view(
            batch_size, *((1,) * (output.ndim - 1))
        ).to(device=device, dtype=output.dtype)
        clean_prediction = (
            output_noisy
            - (1.0 - alpha_bar).clamp_min(0).sqrt() * noise_hat
        ) / alpha_bar.clamp_min(1e-20).sqrt()
        snr = (
            self.forward_process.alphas_cumprod[t]
            / (1.0 - self.forward_process.alphas_cumprod[t]).clamp_min(1e-20)
        ).to(device=device, dtype=output.dtype)
        return {
            "t": t,
            "noise": noise,
            "noise_prediction": noise_hat,
            "squared_error": (noise - noise_hat).square(),
            "clean_prediction": clean_prediction,
            "snr": snr,
        }

    @staticmethod
    def weighted_loss(values, weight=None, sample_weight=None):
        """Average BCHW losses with optional pixel and per-sample weights."""
        combined = torch.ones_like(values)
        if weight is not None:
            weight = weight.to(device=values.device, dtype=values.dtype)
            if weight.ndim == 3:
                weight = weight.unsqueeze(1)
            if weight.shape[1] == 1 and values.shape[1] != 1:
                weight = weight.expand_as(values)
            combined = combined * weight
        if sample_weight is not None:
            sample_weight = sample_weight.to(
                device=values.device, dtype=values.dtype
            )
            sample_weight = sample_weight.view(
                values.shape[0], *((1,) * (values.ndim - 1))
            )
            combined = combined * sample_weight
        return (values * combined).sum() / combined.sum().clamp_min(1.0)

    @staticmethod
    def min_snr_weight(snr, gamma):
        """Return epsilon-prediction Min-SNR weights."""
        if gamma is None:
            return torch.ones_like(snr)
        gamma = float(gamma)
        if gamma <= 0:
            raise ValueError("min_snr_gamma must be positive")
        return snr.clamp(max=gamma) / snr.clamp_min(1e-20)

    def p_loss(
        self,
        output,
        condition,
        mask=None,
        *,
        min_snr_gamma=None,
        t=None,
        return_details=False,
    ):
        """Train epsilon prediction with optional masking and Min-SNR weighting."""
        details = self.diffusion_terms(output, condition, t=t)
        timestep_weight = self.min_snr_weight(
            details["snr"], min_snr_gamma
        )
        loss = self.weighted_loss(
            details["squared_error"], mask, timestep_weight
        )
        if return_details:
            details["timestep_weight"] = timestep_weight
            details["loss"] = loss
            return details
        return loss
