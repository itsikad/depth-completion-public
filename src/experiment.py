from typing import Tuple, Dict

import matplotlib.pyplot as plt

import torch
from torch import Tensor

import pytorch_lightning as pl

from omegaconf import OmegaConf

import models
import losses
from metrics.metrics import depth_comp_metrics, compute_masked_rmse
from utils.visualizations import gen_output_analysis_figure


class DepthCompletionExp(pl.LightningModule):
    """
    Class for Depth Completion experiment.
    """

    def __init__(
        self, 
        hparams: OmegaConf
    ) -> None:

        """
        Initializes an experiment.

        Arguments:
            hparams : a dictionary of hyper parameters
        """

        # Initialize
        super().__init__()
        self.save_hyperparameters(hparams)
        
        # Load model and loss
        self.model = models.builder.create(hparams.model.model_name, **hparams.model.model_params)
        self.loss = losses.builder.create(hparams.loss.loss_name, **hparams.loss.loss_params)

        # Metrics
        self.train_metrics = depth_comp_metrics.clone(prefix='train_')
        self.val_metrics = depth_comp_metrics.clone(prefix='val_')

    def forward(
        self, 
        batch
    ) -> Tensor:

        return self.model(**batch)

    def val_post_process(
        self,
        output: Dict
    ) -> Dict:

        return self.model.val_post_process(output)
        
    def training_step(
        self, 
        batch: Tuple, 
        batch_idx: int
    ) -> Tensor:

        output = self(batch)
        loss = self.loss(**output, **batch)

        # Logging
        self.train_metrics(**output, **batch)
        self.log('train_loss', loss, on_step=True, on_epoch=False)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True)

        # Log sample images
        if batch_idx % self.hparams.log_train_images_interval == 0:
            self._log_images(
                pred=output['pred'][0],
                rgb=batch['rgb'][0],
                target=batch['target'][0],
                target_mask=batch['target_mask'][0],
                batch_idx=batch_idx,
                split='train'
            )

        return loss

    def validation_step(
        self, 
        batch: Tuple, 
        batch_idx: int
    ) -> Tensor:
        
        output = self(batch)
        output = self.val_post_process(output)
        loss = self.loss(**output, **batch)

        # Logging
        self.val_metrics(**output, **batch)
        self.log('val_loss', loss)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True)

        # Log sample images
        if batch_idx % self.hparams.log_val_images_interval == 0:
            self._log_images(
                pred=output['pred'][0],
                rgb=batch['rgb'][0],
                target=batch['target'][0],
                target_mask=batch['target_mask'][0],
                batch_idx=batch_idx,
                split='val'
            )

        return loss

    def configure_optimizers(self):
        optimizer_class = getattr(torch.optim, self.hparams.optimizer.opt_name)
        optimizer = optimizer_class(self.parameters(), **self.hparams.optimizer.opt_params)

        if self.hparams.scheduler.scheduler_name is None:
            return optimizer
        else:
            scheduler_class = getattr(torch.optim.lr_scheduler, self.hparams.scheduler.scheduler_name)
            scheduler = scheduler_class(optimizer, **self.hparams.scheduler.scheduler_params)

            lr_scheduler = {'scheduler': scheduler,
                            'interval': 'epoch'}

            return {'optimizer': optimizer,
                    'lr_scheduler': lr_scheduler}
    
    def _log_images(
        self,
        pred: Tensor,
        rgb: Tensor,
        target: Tensor,
        target_mask: Tensor,
        batch_idx: int,
        split: str
    ):

        # Generate figure
        fig = gen_output_analysis_figure(pred=pred, target=target, img=rgb)

        # Set log name and save figure
        rmse = compute_masked_rmse(pred=pred, target=target, target_mask=target_mask)
        name = f'{split}_epoch_{self.current_epoch}_batch_{batch_idx}_rmse_{rmse:.4f}'
        self.logger.log_image(name, fig)
        plt.close(fig)