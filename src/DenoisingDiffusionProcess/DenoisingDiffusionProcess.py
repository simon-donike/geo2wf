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
    generator=None,
    verbose=False,
):
    """Run one sampler schedule with shared conditional/unconditional dispatch."""
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
            generator=generator,
            verbose=verbose,
        )
        
    def p_loss(self,output,condition,mask=None):
        """
            Assumes output and input are in [-1,+1] range
        """        
        
        b,c,h,w=output.shape
        device=output.device
        
        # loss for training
        
        # input is the optional condition
        t = torch.randint(0, self.forward_process.num_timesteps, (b,), device=device).long()
        output_noisy, noise=self.forward_process(output,t,return_noise=True)        

        # reverse pass
        model_input=torch.cat([output_noisy,condition],1).to(device)
        noise_hat = self.model(model_input, t) 
            
        # apply loss
        if mask is not None:
            mask = mask.to(device=device, dtype=noise.dtype)
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            if mask.shape[1] == 1 and noise.shape[1] != 1:
                mask = mask.expand_as(noise)
            return ((noise - noise_hat).pow(2) * mask).sum() / mask.sum().clamp_min(1.0)
        return self.loss_fn(noise, noise_hat)
