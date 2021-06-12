import hydra
from omegaconf import DictConfig

import pytorch_lightning as pl
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

import datasets
from experiment import DepthCompletionExp


@hydra.main(config_path='configs', config_name="config")
def main(config : DictConfig) -> None:

    # Set seed
    pl.seed_everything(config.seed_id)

    # Logger
    neptune_logger = NeptuneLogger(params=config, **config.logger)

    # Experiment
    experiment = DepthCompletionExp(hparams=config)

    # Data
    dataset = datasets.builder.create(config.dataset.dataset_name, dataset_config=config.dataset, dataloader_config=config.dataloader)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        filename='{epoch}-{val_avg_rmse:.2f}',
        monitor='val_avg_rmse',
        save_top_k=5,
        save_last=True,
        mode='min'
    )

    lr_monitor_callback = LearningRateMonitor(logging_interval='epoch')

    # Trainer    
    trainer = pl.Trainer(
        logger=neptune_logger,
        callbacks=[checkpoint_callback, lr_monitor_callback],
        **config.trainer
    )

    # Train
    trainer.fit(experiment, dataset)


if __name__ == '__main__':
    main()