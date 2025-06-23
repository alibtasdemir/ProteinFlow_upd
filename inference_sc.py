"""DDP inference script."""
import os
import time
import numpy as np
import hydra
import torch

from pytorch_lightning.trainer import Trainer
from omegaconf import DictConfig, OmegaConf
import utils.experiments as eu
from models.proteinflow_clf_wrapperv2 import ProteinFlowModulev2

torch.set_float32_matmul_precision('high')
log = eu.get_pylogger(__name__)


class Sampler:

    def __init__(self, cfg: DictConfig):
        """Initialize sampler.

        Args:
            cfg: inference config.
        """
        self.device_id = 6
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.device_id)
        self.device = f"cuda:{self.device_id}"
        self.device = "cuda:0"
        
        ckpt_path = cfg.inference.ckpt_path
        ckpt_dir = os.path.dirname(ckpt_path)
        ckpt_cfg = OmegaConf.load(os.path.join(ckpt_dir, 'config.yaml'))

        # Set-up config.
        OmegaConf.set_struct(cfg, False)
        OmegaConf.set_struct(ckpt_cfg, False)
        cfg = OmegaConf.merge(cfg, ckpt_cfg)
        cfg.experiment.checkpointer.dirpath = './'

        self._cfg = cfg
        self._infer_cfg = cfg.inference
        self._samples_cfg = self._infer_cfg.samples
        self._rng = np.random.default_rng(self._infer_cfg.seed)

        # Set-up directories to write results to
        self._ckpt_name = '/'.join(ckpt_path.replace('.ckpt', '').split('/')[-3:])
        self._output_dir = os.path.join(
            self._infer_cfg.output_dir,
            self._ckpt_name,
            self._infer_cfg.name,
        )
        os.makedirs(self._output_dir, exist_ok=True)
        
        # ProteinMPNN directory
        self._pmpnn_dir = self._infer_cfg.pmpnn_dir
        
        log.info(f'Saving results to {self._output_dir}')
        config_path = os.path.join(self._output_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            OmegaConf.save(config=self._cfg, f=f)
        log.info(f'Saving inference config to {config_path}')

        # Read checkpoint and initialize module.
        self._flow_module = ProteinFlowModulev2.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            map_location=self.device
        )
        self._flow_module.eval()
        
        self._flow_module._infer_cfg = self._infer_cfg
        self._flow_module._samples_cfg = self._samples_cfg
        self._flow_module._output_dir = self._output_dir
        
        self._flow_module.load_classifiers(self._infer_cfg.classifier)
        self._flow_module.load_folding_model()
        
        #self._folding_model = esm.pretrained.esmfold_v1().eval()
        #self._folding_model = self._folding_model.to(self.device)

    def run_sampling(self):
        #devices = [self.device_id]
        devices = [0]
        log.info(f"Using devices: {devices}")
        eval_dataset = eu.LengthDataset(self._samples_cfg)
        dataloader = torch.utils.data.DataLoader(
            eval_dataset, batch_size=1, shuffle=False, drop_last=False)
        trainer = Trainer(
            accelerator="gpu",
            # strategy="ddp_notebook",
            devices=devices,
            inference_mode=False,
        )
        trainer.predict(self._flow_module, dataloaders=dataloader)
            


@hydra.main(version_base=None, config_path="./configs", config_name="inference")
def run(cfg: DictConfig) -> None:
    # Read model checkpoint.
    log.info(f'Starting inference with {cfg.inference.num_gpus} GPUs')
    start_time = time.time()
    sampler = Sampler(cfg)
    sampler.run_sampling()
    elapsed_time = time.time() - start_time
    log.info(f'Finished in {elapsed_time:.2f}s')


if __name__ == '__main__':
    run()
