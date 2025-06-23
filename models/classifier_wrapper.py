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
        self.accuracy = torchmetrics.Accuracy(task='binary')
        
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
    
    def model_step(self, batch):
        cls = batch["class"].squeeze()
        self.interpolant.set_device(batch['res_mask'].device)
        noisy_batch = self.interpolant.corrupt_batch(batch)
        alphas = noisy_batch["t"]
        num_batch = alphas.shape[0]
        logits = self.model(noisy_batch)
        crent_loss = self.crossent(logits.squeeze(0), cls)
        
        probs = torch.softmax(logits, dim=-1)
        cls_pred = torch.argmax(logits, dim=-1)
        
        if self.stage == 'val':
            self.val_output['clss'].append(cls)
            self.val_output['logits'].append(logits)
            self.val_output['alphas'].append(alphas)
        
        self._log_scalar(f"{self.stage}/accuracy", cls_pred.eq(cls).float().mean(), batch_size=num_batch)
        
        return {
            "cross_entropy": crent_loss.mean()
        }
    
    def training_step(self, batch):
        step_start_time = time.time()
        self.stage = 'train'
        batch_loss = self.model_step(batch)
        total_losses = {
            k: torch.mean(v) for k, v in batch_loss.items()
        }
        num_batch = batch['res_mask'].shape[0]
        for k, v in total_losses.items():
            self._log_scalar(
                f"train/{k}", v, prog_bar=False, batch_size=num_batch)
            
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
            'dummy': 1,        
        }
    
    def on_validation_epoch_end(self):
        log = self.val_output
        log = {key: log[key] for key in log if "val" in key}
        log = self.gather_log(log, self.trainer.world_size)
        mean_log = self.get_log_mean(log)
        mean_log.update({'epoch': float(self.trainer.current_epoch), 'step': float(self.trainer.global_step)})
        # pil_auroc_aupr, pil_auroc_acc, pil_acc_aupr, aurocs, accuracies, auprs = self.scatter_plots()
        aurocs, accuracies, auprs = self.scatter_plots()
        mean_log.update({'val/max_auroc': float(aurocs.max()), 'val/max_aupr': float(auprs.max()), 'val/max_accuracy': float(accuracies.max())})
        
        if self.trainer.is_global_zero:
            self.log_dict(mean_log, batch_size=1)
            # wandb.log({'fig': [wandb.Image(pil_auroc_aupr), wandb.Image(pil_auroc_acc), wandb.Image(pil_acc_aupr)], 'step': self.trainer.global_step,'iter_step': self.iter_step})
            wandb.log(mean_log)
            pd.DataFrame(log).to_csv(os.path.join(self._exp_cfg.checkpointer.dirpath, f"val_{self.trainer.global_step}.csv"))

        self.val_output = defaultdict(list)
    
    def scatter_plots(self):
        clss = torch.stack(self.val_output["clss"])
        clss_np = clss.detach().cpu().numpy()
        
        AUROC = torchmetrics.classification.AUROC(task="binary")
        
        ACC = torchmetrics.classification.Accuracy(task="binary").to(self.device)
        AUPR = torchmetrics.classification.AveragePrecision(task="binary").to(self.device)
        
        probs = torch.softmax(torch.cat(self.val_output["logits"]), dim=-1)
        probs_np = probs.detach().cpu().numpy()
        

        aurocs = roc_auc_score(clss_np, probs_np[:, 1])
        # aurocs = AUROC(probs, clss)
        accuracies = ACC(probs[:,1], clss).detach().cpu().numpy()
        auprs = AUPR(probs[:,1], clss).detach().cpu().numpy()
        title = f"Classification Metrics"
        #pil_auroc_aupr = self.create_scatter_plot(x=aurocs, y=auprs, title=title, x_label='auROC', y_label='auPR')
        #pil_auroc_acc = self.create_scatter_plot(x=aurocs, y=accuracies, title=title, x_label='auROC', y_label='accuracy')
        #pil_acc_aupr = self.create_scatter_plot(x=accuracies, y=auprs, title=title, x_label='accuracy', y_label='auPR')
        #return pil_auroc_aupr, pil_auroc_acc, pil_acc_aupr, aurocs, accuracies, auprs
        return aurocs, accuracies, auprs
    
    def create_scatter_plot(self, x, y, title, x_label, y_label):
        """
        Creates a scatter plot with the given x and y data, title, and axis labels.

        Parameters:
        x (array-like): The data for the x-axis.
        y (array-like): The data for the y-axis.
        title (str): The title of the plot.
        x_label (str): The label for the x-axis.
        y_label (str): The label for the y-axis.
        """
        #if len(x) != len(y):
        #    raise ValueError("The length of x and y arrays must be the same.")

        sizes = np.arange(1, len(x) + 1)  # Generate size array for the markers

        plt.figure(figsize=(8, 6))  # Set the figure size
        scatter = plt.scatter(x, y, s=50, c=sizes, cmap='viridis')  # Create scatter plot with a colormap
        plt.title(title)  # Set the title
        plt.xlabel(x_label)  # Set x-axis label
        plt.ylabel(y_label)  # Set y-axis label
        plt.grid(True)  # Show grid
        for i, txt in enumerate(sizes):
            plt.annotate(txt, (x[i], y[i]), fontsize=12)  # Annotate each point with its corresponding size value

        fig = plt.gcf()
        fig.canvas.draw()
        pil_img = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        plt.close()
        return pil_img

    
    def gather_log(self, log, world_size):
        if world_size == 1:
            return log
        log_list = [None] * world_size
        torch.distributed.all_gather_object(log_list, log)
        log = {key: sum([l[key] for l in log_list], []) for key in log}
        return log

    def get_log_mean(self, log):
        out = {}
        for key in log:
            try:
                out[key] = np.nanmean(log[key])
            except:
                pass
        return out
    
    def configure_optimizers(self):
        return torch.optim.AdamW(
            params=self.model.parameters(),
            **self._exp_cfg.optimizer
        )
        