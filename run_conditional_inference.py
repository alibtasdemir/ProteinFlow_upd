"""DDP inference script with conditional sampling."""
import os
import time
import numpy as np
import hydra
import torch
import argparse
import shutil

from pytorch_lightning.trainer import Trainer
from omegaconf import DictConfig, OmegaConf
import utils.experiments as eu
from models.proteinflow_clf_wrapperv2 import ProteinFlowModulev2

torch.set_float32_matmul_precision('high')
log = eu.get_pylogger(__name__)


class ConditionalSampler:

    def __init__(self, cfg: DictConfig):
        """Initialize sampler.

        Args:
            cfg: inference config containing input_pdb, fixed_residues, and num_samples
        """
        self.device_id = 0
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.device_id)
        self.device = f"cuda:{self.device_id}"
        
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
        self._rng = np.random.default_rng(self._infer_cfg.seed)
        self._samples_cfg = self._infer_cfg.samples

        # Set-up directories to write results to
        self._ckpt_name = '/'.join(ckpt_path.replace('.ckpt', '').split('/')[-3:])
        self._output_dir = os.path.join(
            self._infer_cfg.output_dir,
            self._ckpt_name,
            self._infer_cfg.name,
            "conditional_samples"
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
        
        # Store sampling parameters
        self._input_pdb = cfg.input_pdb
        self._fixed_residues = cfg.fixed_residues
        self._num_samples = cfg.num_samples

    def run_sampling(self):
        """Run conditional sampling with fixed residues."""
        log.info(f"Using device: {self.device}")
        log.info(f"Running conditional sampling with fixed residues: {self._fixed_residues}")
        
        try:
            samples = self._flow_module.sample_with_fixed_residues(
                pdb_path=self._input_pdb,
                fixed_residues=self._fixed_residues,
                num_samples=self._num_samples,
                output_dir=self._output_dir
            )
            log.info(f"Generated {len(samples)} samples")
            
            # Evaluate quality of samples
            for i, sample_path in enumerate(samples):
                quality_metrics = self._flow_module.evaluate_structure_quality(
                    sample_path, 
                    reference_pdb_path=self._input_pdb,
                    fixed_residues=self._fixed_residues
                )
                log.info(f"Sample {i} quality metrics: {quality_metrics}")
                # Run ProteinMPNN self-consistency (no folding)
                sc_output_dir = os.path.join(os.path.dirname(sample_path), "protein_mpnn")
                os.makedirs(sc_output_dir, exist_ok=True)
                shutil.copy(sample_path, os.path.join(sc_output_dir, os.path.basename(sample_path)))
                log.info(f"Running ProteinMPNN for sample {i}...")
                self._flow_module.run_self_consistency(
                    decoy_pdb_dir=sc_output_dir,
                    reference_pdb_path=sample_path,
                    motif_mask=None,
                    run_folding=False
                )
                log.info(f"ProteinMPNN for sample {i} complete. Results in {sc_output_dir}")
                
                
                
        except Exception as e:
            log.error(f"Error during sampling: {e}")
            log.exception("Detailed traceback:")
        
        log.info(f"All samples saved to {self._output_dir}")
        log.info("Inference complete!")


@hydra.main(version_base=None, config_path="./configs", config_name="inference")
def run(cfg: DictConfig) -> None:
    # Add default values for command-line arguments if not present
    if not hasattr(cfg, 'input_pdb'):
        raise ValueError("input_pdb must be specified in config or command line")
    if not hasattr(cfg, 'fixed_residues'):
        raise ValueError("fixed_residues must be specified in config or command line")
    if not hasattr(cfg, 'num_samples'):
        cfg.num_samples = 5
    
    # Run inference
    log.info(f'Starting conditional inference')
    start_time = time.time()
    sampler = ConditionalSampler(cfg=cfg)
    sampler.run_sampling()
    elapsed_time = time.time() - start_time
    log.info(f'Finished in {elapsed_time:.2f}s')


if __name__ == '__main__':
    run() 