import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from .DenoisingDiffusionProcess import *

class PixelDiffusion(pl.LightningModule):
    def __init__(self,
                 train_dataset,
                 valid_dataset=None,
                 num_timesteps=1000,
                 batch_size=1,
                 lr=1e-3,
                 lr_scheduler_factor=0.5,
                 lr_scheduler_patience=10):
        super().__init__()
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.lr = lr
        self.batch_size=batch_size
        self.lr_scheduler_factor=lr_scheduler_factor
        self.lr_scheduler_patience=lr_scheduler_patience
        
        self.model=DenoisingDiffusionProcess(num_timesteps=num_timesteps)

    @torch.no_grad()
    def forward(self,*args,**kwargs):
        return self.output_T(self.model(*args,**kwargs))
    
    def input_T(self, input):
        # By default, let the model accept samples in [0,1] range, and transform them automatically
        return (input.clip(0,1).mul_(2)).sub_(1)
    
    def output_T(self, input):
        # Inverse transform of model output from [-1,1] to [0,1] range
        return (input.add_(1)).div_(2)
    
    def training_step(self, batch, batch_idx):   
        images=batch
        loss = self.model.p_loss(self.input_T(images))
        
        self.log('train_loss',loss,on_step=True,on_epoch=True,prog_bar=True,logger=True)
        
        return loss
            
    def validation_step(self, batch, batch_idx):     
        images=batch
        loss = self.model.p_loss(self.input_T(images))
        
        self.log('val_loss',loss,on_step=False,on_epoch=True,prog_bar=True,logger=True)
        
        return loss
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=4)
    
    def val_dataloader(self):
        if self.valid_dataset is not None:
            return DataLoader(self.valid_dataset,
                              batch_size=self.batch_size,
                              shuffle=False,
                              num_workers=4)
        else:
            return None
    
    def configure_optimizers(self):
        optimizer=torch.optim.AdamW(list(filter(lambda p: p.requires_grad, self.model.parameters())), lr=self.lr)
        scheduler=ReduceLROnPlateau(optimizer,
                                    mode='min',
                                    factor=self.lr_scheduler_factor,
                                    patience=self.lr_scheduler_patience)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler,
                                 "monitor": "val_loss"}}
    
class PixelDiffusionConditional(PixelDiffusion):
    def __init__(self,
                 train_dataset,
                 valid_dataset=None,
                 condition_channels=3,
                 generated_channels=3,
                 num_timesteps=1000,
                 schedule='linear',
                 model_dim=64,
                 model_dim_mults=(1,2,4,8),
                 model_channels=None,
                 model_out_dim=None,
                 batch_size=1,
                 lr=1e-3,
                 lr_scheduler_factor=0.5,
                 lr_scheduler_patience=10):
        pl.LightningModule.__init__(self)
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.lr = lr
        self.batch_size=batch_size
        self.lr_scheduler_factor=lr_scheduler_factor
        self.lr_scheduler_patience=lr_scheduler_patience
        
        self.model=DenoisingDiffusionConditionalProcess(generated_channels=generated_channels,
                                                        condition_channels=condition_channels,
                                                        schedule=schedule,
                                                        num_timesteps=num_timesteps,
                                                        model_dim=model_dim,
                                                        model_dim_mults=model_dim_mults,
                                                        model_channels=model_channels,
                                                        model_out_dim=model_out_dim)
    
    def training_step(self, batch, batch_idx):   
        input,output=batch
        loss = self.model.p_loss(self.input_T(output),self.input_T(input))
        
        self.log('train_loss',loss,on_step=True,on_epoch=True,prog_bar=True,logger=True)
        
        return loss
            
    def validation_step(self, batch, batch_idx):     
        input,output=batch
        loss = self.model.p_loss(self.input_T(output),self.input_T(input))
        
        self.log('val_loss',loss,on_step=False,on_epoch=True,prog_bar=True,logger=True)

        if batch_idx == 0:
            pred_batch = self.predict_step(batch, batch_idx)
            psnr, ssim, l1 = self._compute_reconstruction_metrics(pred_batch, output)
            self.log('val_recon_psnr', psnr, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('val_recon_ssim', ssim, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('val_recon_l1', l1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self._log_val_reconstruction(input, pred_batch, output)
        
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        del batch_idx, dataloader_idx
        input,_ = batch
        pred = self.model(self.input_T(input))
        return self.output_T(pred)

    def _to_plot_image(self, tensor):
        image = tensor.detach().float().cpu().clamp(0, 1)
        if image.shape[0] >= 3:
            return image[:3].permute(1, 2, 0).numpy(), None
        return image[0].numpy(), 'gray'

    def _compute_reconstruction_metrics(self, pred_batch, target_batch):
        pred = pred_batch.detach().float().clamp(0, 1)
        target = target_batch.detach().float().clamp(0, 1)

        l1 = F.l1_loss(pred, target)

        mse = F.mse_loss(pred, target)
        psnr = 10.0 * torch.log10(1.0 / torch.clamp(mse, min=1e-12))

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        mu_x = F.avg_pool2d(pred, kernel_size=3, stride=1, padding=1)
        mu_y = F.avg_pool2d(target, kernel_size=3, stride=1, padding=1)
        sigma_x = F.avg_pool2d(pred * pred, kernel_size=3, stride=1, padding=1) - mu_x * mu_x
        sigma_y = F.avg_pool2d(target * target, kernel_size=3, stride=1, padding=1) - mu_y * mu_y
        sigma_xy = F.avg_pool2d(pred * target, kernel_size=3, stride=1, padding=1) - mu_x * mu_y
        ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
            (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
        )
        ssim = ssim_map.mean()

        return psnr, ssim, l1

    def _log_val_reconstruction(self, input_batch, pred_batch, target_batch):
        if self.logger is None or self.trainer is None or not self.trainer.is_global_zero:
            return

        try:
            import matplotlib.pyplot as plt
            import wandb
        except ImportError:
            return

        x_img, x_cmap = self._to_plot_image(input_batch[0])
        pred_img, pred_cmap = self._to_plot_image(pred_batch[0])
        y_img, y_cmap = self._to_plot_image(target_batch[0])

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(x_img, cmap=x_cmap)
        axes[0].set_title('x')
        axes[0].axis('off')
        axes[1].imshow(pred_img, cmap=pred_cmap)
        axes[1].set_title('pred')
        axes[1].axis('off')
        axes[2].imshow(y_img, cmap=y_cmap)
        axes[2].set_title('y')
        axes[2].axis('off')
        fig.tight_layout()

        self.logger.experiment.log(
            {"val/reconstruction": wandb.Image(fig)},
            step=self.global_step,
        )
        plt.close(fig)
