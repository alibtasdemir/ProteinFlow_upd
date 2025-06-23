from collections import defaultdict
import PIL
import logging
import time, os
import torch
import torchmetrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pytorch_lightning import LightningModule
from torchmetrics.classification import BinaryAccuracy
from torchmetrics import Accuracy, AUROC, AveragePrecision
from models.classifier import ProtClassifier
from utils.flows import Interpolant
import wandb

from sklearn.metrics import roc_auc_score


class ClasfModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self._print_logger = logging.getLogger(__name__)
        self._exp_cfg = cfg.experiment
        self._model_cfg = cfg.model
        self._data_cfg = cfg.data
        self._interpolant_cfg = cfg.interpolant
        
        # Set-up prediction model
        self.model = ProtClassifier(cfg.model)
        
        # Set-up interpolant
        self.interpolant = Interpolant(cfg.interpolant)
        
        self.crossent = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=2)
        
        self.val_output = defaultdict(list)
        self.save_hyperparameters()
    
    def _log_scalar(
            self,
            key,
            value,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=None,
            sync_dist=False,
            rank_zero_only=True
    ):
        if sync_dist and rank_zero_only:
            raise ValueError('Unable to sync dist when rank_zero_only=True')
        self.log(
            key,
            value,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=prog_bar,
            batch_size=batch_size,
            sync_dist=sync_dist,
            rank_zero_only=rank_zero_only
        )
    
    def model_step(self, batch):
        # Get class label
        cls = batch["class"].squeeze()
        
        # Step the batch through flow
        self.interpolant.set_device(batch['res_mask'].device)
        noisy_batch = self.interpolant.corrupt_batch(batch)
        alphas = noisy_batch["t"]
        num_batch = alphas.shape[0]
        
        # Calculate logits with classifier
        logits = self.model(noisy_batch)
        
        # Cross-entropy loss
        crent_loss = self.crossent(logits.squeeze(0), cls)
        
        probs = torch.softmax(logits, dim=-1)
        cls_pred = torch.argmax(logits, dim=-1)
        
        #print(cls_pred)
        #print(f"Shape: {cls_pred.shape}")
        #print("-"*30)
        #print(cls)
        #print(f"Shape: {cls.shape}")
        
        acc = (cls_pred == cls).cpu().numpy().mean()
        #print(acc)
        #acc = self.accuracy(cls_pred, cls.unsqueeze(0))
        
        
        if self.stage == 'val':
            self.val_output['cls'].append(cls)
            self.val_output['logits'].append(logits)
            self.val_output['alphas'].append(alphas)
        
        self._log_scalar(f"{self.stage}/accuracy", acc, on_epoch=True, batch_size=num_batch)
        self._log_scalar(f"{self.stage}/celoss", crent_loss, on_epoch=True, batch_size=num_batch)
        
        return {
            "cross_entropy": crent_loss.mean()
        }
        
    
    def on_train_start(self):
        self._epoch_start_time = time.time()

    def on_train_epoch_end(self):
        epoch_time = (time.time() - self._epoch_start_time) / 60.0
        self.log(
            'train/epoch_time_minutes',
            epoch_time,
            on_step=False,
            on_epoch=True,
            prog_bar=False
        )
        self._epoch_start_time = time.time()
    
    def training_step(self, batch):
        step_start_time = time.time()
        self.stage = 'train'
        batch_loss = self.model_step(batch)
        num_batch = batch['res_mask'].shape[0]
        
        total_losses = {
            k: torch.mean(v) for k, v in batch_loss.items()
        }
        """
        for k, v in total_losses.items():
            self._log_scalar(
                f"train/{k}", v, prog_bar=False, batch_size=num_batch)
        """
        
        # Training throughput
        self._log_scalar(
            "train/length", batch['res_mask'].shape[1], prog_bar=False, batch_size=num_batch)
        self._log_scalar(
            "train/batch_size", num_batch, prog_bar=False)
        step_time = time.time() - step_start_time
        self._log_scalar(
            "train/examples_per_second", num_batch / step_time)
        train_loss = (
                total_losses["cross_entropy"]
        )
        self._log_scalar(
            "train/loss", train_loss, batch_size=num_batch)
        return train_loss
    
    def validation_step(self, batch):
        self.stage = 'val'
        num_batch = batch['res_mask'].shape[0]
        batch_loss = self.model_step(batch)
        
        total_losses = {
            k: torch.mean(v) for k, v in batch_loss.items()
        }

        val_loss = (
            total_losses["cross_entropy"]
        )
        self._log_scalar(
            "val/loss", val_loss, prog_bar=False, batch_size=num_batch, on_epoch=True)
        return {
            'val/loss': val_loss, 
        }
    
    def configure_optimizers(self):
        return torch.optim.AdamW(
            params=self.model.parameters(),
            **self._exp_cfg.optimizer
        )
        