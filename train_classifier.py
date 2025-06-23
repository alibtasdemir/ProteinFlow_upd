import os
import torch

import hydra
from omegaconf import DictConfig, OmegaConf

# Pytorch lightning imports
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

# from dataset.data import PdbDataModule
from dataset.classification_data import PdbDataModule
from models.classifier_wrapper_v2 import ClasfModule
from dataset.classification_data import get_dataloaders

from utils.experiments import get_pylogger, flatten_dict
import wandb

log = get_pylogger(__name__)
torch.set_float32_matmul_precision('high')


class ClassifierTrainer:
    def __init__(self, *, cfg: DictConfig):
        self._cfg = cfg
        self._data_cfg = cfg.data
        self._exp_cfg = cfg.experiment
        self._datamodule: LightningDataModule = PdbDataModule(self._data_cfg)
        # self.train_loader, self.val_loader = get_dataloaders(self._data_cfg)
        self._model: LightningModule = ClasfModule(self._cfg)
    
    def train(self):
        callbacks = []
        logger = WandbLogger(
            **self._exp_cfg.wandb,
        )
        
        # Checkpoint directory
        ckpt_dir = self._exp_cfg.checkpointer.dirpath
        os.makedirs(ckpt_dir, exist_ok=True)
        log.info(f"Checkpoints saved to {ckpt_dir}")

        # Model Checkpoints
        callbacks.append(ModelCheckpoint(**self._exp_cfg.checkpointer))
        
        # Save config
        cfg_path = os.path.join(ckpt_dir, 'config.yaml')
        with open(cfg_path, 'w') as f:
            OmegaConf.save(config=self._cfg, f=f.name)
        cfg_dict = OmegaConf.to_container(self._cfg, resolve=True)
        flat_cfg = dict(flatten_dict(cfg_dict))
        if isinstance(logger.experiment.config, wandb.sdk.wandb_config.Config):
            logger.experiment.config.update(flat_cfg)
        
        # GPU Settings
        devices = [3]
        log.info(f"Using devices: {devices}")
        trainer = Trainer(
            **self._exp_cfg.trainer,
            callbacks=callbacks,
            logger=logger,
            use_distributed_sampler=False,
            enable_progress_bar=True,
            enable_model_summary=True,
            #gpus=devices
            devices=devices,
        )
        trainer.fit(
            model=self._model,
            # train_dataloaders=self.train_loader,
            # val_dataloaders=self.val_loader,
            datamodule=self._datamodule,
            ckpt_path=self._exp_cfg.warm_start
        )

@hydra.main(version_base=None, config_path="./configs", config_name="classifier.yaml")
def main(cfg: DictConfig):
    exp = ClassifierTrainer(cfg=cfg)
    exp.train()

if __name__ == "__main__":
    main()