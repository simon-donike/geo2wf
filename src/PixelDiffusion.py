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
        
        self.log('train_loss',loss)
        
        return loss
            
    def validation_step(self, batch, batch_idx):     
        images=batch
        loss = self.model.p_loss(self.input_T(images))
        
        self.log('val_loss',loss)
        
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
        
        self.log('train_loss',loss)
        
        return loss
            
    def validation_step(self, batch, batch_idx):     
        input,output=batch
        loss = self.model.p_loss(self.input_T(output),self.input_T(input))
        
        self.log('val_loss',loss)
        
        return loss
