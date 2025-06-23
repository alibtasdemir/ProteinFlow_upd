from typing import Any
import torch
import time
import os
import subprocess
import shutil
import random
import wandb
import numpy as np
import pandas as pd
import logging
from pytorch_lightning import LightningModule
from Bio import PDB

import esm
from biotite.sequence.io import fasta

from utils.experiments import write_prot_to_pdb, save_traj, create_full_prot
from utils import metrics
from models.proteinflow import ProteinFlow
from models.classifier import ProtClassifier
from models.classifier_wrapper import ClasfModule
from utils import all_atom
from utils import so3Utils as su
from utils import residue_constants as rc
from utils import pdbUtils as du
from utils.flows import Interpolant

from utils.modelUtils import t_stratified_loss, to_numpy
from pytorch_lightning.loggers.wandb import WandbLogger

from dataset import protein


class ProteinFlowModulev2(LightningModule):

    def __init__(self, cfg, classifier_cfg=None):
        super().__init__()
        self._print_logger = logging.getLogger(__name__)
        self._exp_cfg = cfg.experiment
        self._model_cfg = cfg.model
        self._data_cfg = cfg.data
        self._interpolant_cfg = cfg.interpolant
        # self._classf_cfg = classifier_cfg

        # Set-up vector field prediction model
        self.model = ProteinFlow(cfg.model)

        # Set-up interpolant
        self.interpolant = Interpolant(cfg.interpolant)
        
        # Classifier
        self.loaded_classifier = False
        # self.load_classifiers(self._classf_cfg)

        self._sample_write_dir = self._exp_cfg.checkpointer.dirpath
        os.makedirs(self._sample_write_dir, exist_ok=True)

        self.validation_epoch_metrics = []
        self.validation_epoch_samples = []
        self.save_hyperparameters()
        print(f"Model is initiated on GPU: {torch.cuda.current_device()}")

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

    def model_step(self, noisy_batch: Any):
        training_cfg = self._exp_cfg.training
        loss_mask = noisy_batch['res_mask']
        if training_cfg.min_plddt_mask is not None:
            plddt_mask = noisy_batch['res_plddt'] > training_cfg.min_plddt_mask
            loss_mask *= plddt_mask
        num_batch, num_res = loss_mask.shape

        # Ground truth labels
        gt_trans_1 = noisy_batch['trans_1']
        gt_rotmats_1 = noisy_batch['rotmats_1']
        rotmats_t = noisy_batch['rotmats_t']
        gt_rot_vf = su.calc_rot_vf(
            rotmats_t, gt_rotmats_1.type(torch.float32))
        gt_bb_atoms = all_atom.to_atom37(gt_trans_1, gt_rotmats_1)[:, :, :3]

        # Timestep used for normalization.
        t = noisy_batch['t']
        norm_scale = 1 - torch.min(
            t[..., None], torch.tensor(training_cfg.t_normalize_clip))
        
        # Model output predictions.
        model_output = self.model(noisy_batch)
        pred_trans_1 = model_output['pred_trans']
        pred_rotmats_1 = model_output['pred_rotmats']
        pred_rots_vf = su.calc_rot_vf(rotmats_t, pred_rotmats_1)

        # Backbone atom loss
        pred_bb_atoms = all_atom.to_atom37(pred_trans_1, pred_rotmats_1)[:, :, :3]
        gt_bb_atoms *= training_cfg.bb_atom_scale / norm_scale[..., None]
        pred_bb_atoms *= training_cfg.bb_atom_scale / norm_scale[..., None]
        loss_denom = torch.sum(loss_mask, dim=-1) * 3
        bb_atom_loss = torch.sum(
            (gt_bb_atoms - pred_bb_atoms) ** 2 * loss_mask[..., None, None],
            dim=(-1, -2, -3)
        ) / loss_denom

        # Translation VF loss
        trans_error = (gt_trans_1 - pred_trans_1) / norm_scale * training_cfg.trans_scale
        trans_loss = training_cfg.translation_loss_weight * torch.sum(
            trans_error ** 2 * loss_mask[..., None],
            dim=(-1, -2)
        ) / loss_denom

        # Rotation VF loss
        rots_vf_error = (gt_rot_vf - pred_rots_vf) / norm_scale
        rots_vf_loss = training_cfg.rotation_loss_weights * torch.sum(
            rots_vf_error ** 2 * loss_mask[..., None],
            dim=(-1, -2)
        ) / loss_denom

        # Pairwise distance loss
        gt_flat_atoms = gt_bb_atoms.reshape([num_batch, num_res * 3, 3])
        gt_pair_dists = torch.linalg.norm(
            gt_flat_atoms[:, :, None, :] - gt_flat_atoms[:, None, :, :], dim=-1)
        pred_flat_atoms = pred_bb_atoms.reshape([num_batch, num_res * 3, 3])
        pred_pair_dists = torch.linalg.norm(
            pred_flat_atoms[:, :, None, :] - pred_flat_atoms[:, None, :, :], dim=-1)

        flat_loss_mask = torch.tile(loss_mask[:, :, None], (1, 1, 3))
        flat_loss_mask = flat_loss_mask.reshape([num_batch, num_res * 3])
        flat_res_mask = torch.tile(loss_mask[:, :, None], (1, 1, 3))
        flat_res_mask = flat_res_mask.reshape([num_batch, num_res * 3])

        gt_pair_dists = gt_pair_dists * flat_loss_mask[..., None]
        pred_pair_dists = pred_pair_dists * flat_loss_mask[..., None]
        pair_dist_mask = flat_loss_mask[..., None] * flat_res_mask[:, None, :]

        dist_mat_loss = torch.sum(
            (gt_pair_dists - pred_pair_dists) ** 2 * pair_dist_mask,
            dim=(1, 2))
        dist_mat_loss /= (torch.sum(pair_dist_mask, dim=(1, 2)) - num_res)

        se3_vf_loss = trans_loss + rots_vf_loss
        auxiliary_loss = (bb_atom_loss + dist_mat_loss) * (
                t[:, 0] > training_cfg.aux_loss_t_pass
        )
        auxiliary_loss *= self._exp_cfg.training.aux_loss_weight
        se3_vf_loss += auxiliary_loss
        if torch.isnan(se3_vf_loss).any():
            raise ValueError('NaN loss encountered')
        return {
            "bb_atom_loss": bb_atom_loss,
            "trans_loss": trans_loss,
            "dist_mat_loss": dist_mat_loss,
            "auxiliary_loss": auxiliary_loss,
            "rots_vf_loss": rots_vf_loss,
            "se3_vf_loss": se3_vf_loss
        }

    def validation_step(self, batch: Any, batch_idx: int):
        res_mask = batch['res_mask']
        self.interpolant.set_device(res_mask.device)
        num_batch, num_res = res_mask.shape

        samples = self.interpolant.sample(
            num_batch,
            num_res,
            self.model,
        )[0][-1].numpy()

        batch_metrics = []
        for i in range(num_batch):

            # Write out sample to PDB file
            final_pos = samples[i]
            saved_path = write_prot_to_pdb(
                final_pos,
                os.path.join(
                    self._sample_write_dir,
                    f'sample_{i}_idx_{batch_idx}_len_{num_res}.pdb'),
                no_indexing=True
            )
            if isinstance(self.logger, WandbLogger):
                self.validation_epoch_samples.append(
                    [saved_path, self.global_step, wandb.Molecule(saved_path)]
                )

            mdtraj_metrics = metrics.calc_mdtraj_metrics(saved_path)
            ca_idx = rc.atom_order['CA']
            ca_ca_metrics = metrics.calc_ca_ca_metrics(final_pos[:, ca_idx])
            batch_metrics.append((mdtraj_metrics | ca_ca_metrics))

        batch_metrics = pd.DataFrame(batch_metrics)
        self.validation_epoch_metrics.append(batch_metrics)

    def on_validation_epoch_end(self):
        if len(self.validation_epoch_samples) > 0:
            self.logger.log_table(
                key='valid/samples',
                columns=["sample_path", "global_step", "Protein"],
                data=self.validation_epoch_samples)
            self.validation_epoch_samples.clear()
        val_epoch_metrics = pd.concat(self.validation_epoch_metrics)
        for metric_name, metric_val in val_epoch_metrics.mean().to_dict().items():
            self._log_scalar(
                f'valid/{metric_name}',
                metric_val,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=len(val_epoch_metrics),
            )
        self.validation_epoch_metrics.clear()

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

    def training_step(self, batch: Any, stage: int):
        self.stage = 'train'
        step_start_time = time.time()
        self.interpolant.set_device(batch['res_mask'].device)
        noisy_batch = self.interpolant.corrupt_batch(batch)
        if self._interpolant_cfg.self_condition and random.random() > 0.5:
            with torch.no_grad():
                model_sc = self.model(noisy_batch)
                noisy_batch['trans_sc'] = model_sc['pred_trans']
        batch_losses = self.model_step(noisy_batch)
        num_batch = batch_losses['bb_atom_loss'].shape[0]
        total_losses = {
            k: torch.mean(v) for k, v in batch_losses.items()
        }
        for k, v in total_losses.items():
            self._log_scalar(
                f"train/{k}", v, prog_bar=False, batch_size=num_batch)

        # Losses to track. Stratified across t.
        t = torch.squeeze(noisy_batch['t'])
        self._log_scalar(
            "train/t",
            np.mean(to_numpy(t)),
            prog_bar=False, batch_size=num_batch)
        for loss_name, loss_dict in batch_losses.items():
            stratified_losses = t_stratified_loss(
                t, loss_dict, loss_name=loss_name)
            for k, v in stratified_losses.items():
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
                total_losses[self._exp_cfg.training.loss]
                + total_losses['auxiliary_loss']
        )
        self._log_scalar(
            "train/loss", train_loss, batch_size=num_batch)
        return train_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            params=self.model.parameters(),
            **self._exp_cfg.optimizer
        )
        
    def load_classifiers(self, cfg, requires_grad = True):
        self._classf_cfg = cfg
        self.cls_model = ClasfModule.load_from_checkpoint(
            checkpoint_path=self._classf_cfg.ckpt_path,
            map_location=f'cuda:{torch.cuda.current_device()}'
        )
        
        self._pmpnn_dir = self._infer_cfg.pmpnn_dir
        #self.cls_model = ProtClassifier(self._classifier_cfg)
        #self.cls_model.load_state_dict(torch.load(self._classifier_cfg.ckpt_path))
        #self.cls_model.eval()
        #self.cls_model.to(self.device)
        for param in self.cls_model.parameters():
            param.requires_grad = requires_grad
    
    def load_folding_model(self):
        print(f"Current GPU of folding model is {torch.cuda.current_device()}")
        self._folding_model = esm.pretrained.esmfold_v1()
        self._folding_model = self._folding_model.eval()
        self._folding_model = self._folding_model.to(f'cuda:{torch.cuda.current_device()}')

    def run_self_consistency(
        self,
        decoy_pdb_dir: str,
        reference_pdb_path: str,
        motif_mask = None,
        run_folding=True,
        ):
        device = f'cuda:{torch.cuda.current_device()}'
        # Run ProteinMPNN
        output_path = os.path.join(decoy_pdb_dir, "parsed_pdbs.jsonl")
        process = subprocess.Popen(
            [
                "python",
                f"{self._pmpnn_dir}/helper_scripts/parse_multiple_chains.py",
                f"--input_path={decoy_pdb_dir}",
                f"--output_path={output_path}",
            ]
        )
        _ = process.wait()
        num_tries = 0
        ret = -1
        pmpnn_args = [
            "python",
            f"{self._pmpnn_dir}/protein_mpnn_run.py",
            "--out_folder",
            decoy_pdb_dir,
            "--jsonl_path",
            output_path,
            "--num_seq_per_target",
            str(self._samples_cfg.seq_per_sample),
            "--sampling_temp",
            "0.1",
            "--seed",
            str(self._infer_cfg.seed),
            "--batch_size",
            "1",
        ]
        pmpnn_args.append("--device")
        pmpnn_args.append(str(torch.cuda.current_device()))
        while ret < 0:
            try:
                process = subprocess.Popen(
                    pmpnn_args, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
                )
                ret = process.wait()
            except Exception as e:
                num_tries += 1
                self._log.info(f"Failed ProteinMPNN. Attempt {num_tries}/5 {e}")
                torch.cuda.empty_cache()
                if num_tries < 4:
                    raise e
        
        mpnn_fasta_path = os.path.join(
            decoy_pdb_dir,
            "seqs",
            os.path.basename(reference_pdb_path).replace(".pdb", ".fa")
        )
        if not run_folding:
            return mpnn_fasta_path
        # --- ESMFold and metrics part below (unchanged) ---
        mpnn_results = {
            "tm_score": [],
            "sample_path": [],
            "header": [],
            "sequence": [],
            "rmsd": [],
        }
        if motif_mask is not None:
            mpnn_results["motif_rmsd"] = []
        esmf_dir = os.path.join(decoy_pdb_dir, "esmf")
        os.makedirs(esmf_dir, exist_ok=True)
        fasta_seqs = fasta.FastaFile.read(mpnn_fasta_path)
        sample_feats = du.parse_pdb_feats("sample", reference_pdb_path)
        for i, (header, string) in enumerate(fasta_seqs.items()):
            # Run ESMFold
            esmf_sample_path = os.path.join(esmf_dir, f"sample_{i}.pdb")
            _ = self.run_folding(string, esmf_sample_path)
            esmf_feats = du.parse_pdb_feats("folded_sample", esmf_sample_path)
            sample_seq = du.aatype_to_seq(sample_feats["aatype"])
            # Calculate scTM and ESMFold outputs with reference
            _, tm_score = metrics.calc_tm_score(
                sample_feats["bb_positions"],
                esmf_feats["bb_positions"],
                sample_seq,
                sample_seq,
            )
            rmsd = metrics.calc_aligned_rmsd(
                sample_feats["bb_positions"], esmf_feats["bb_positions"]
            )
            if motif_mask is not None:
                sample_motif = sample_feats["bb_positions"][motif_mask]
                of_motif = esmf_feats["bb_positions"][motif_mask]
                motif_rmsd = metrics.calc_aligned_rmsd(sample_motif, of_motif)
                mpnn_results["motif_rmsd"].append(motif_rmsd)
            mpnn_results["rmsd"].append(rmsd)
            mpnn_results["tm_score"].append(tm_score)
            mpnn_results["sample_path"].append(esmf_sample_path)
            mpnn_results["header"].append(header)
            mpnn_results["sequence"].append(string)
        # Save results to CSV
        csv_path = os.path.join(decoy_pdb_dir, "sc_results.csv")
        mpnn_results = pd.DataFrame(mpnn_results)
        mpnn_results.to_csv(csv_path)
        
            
    def run_folding(self, sequence, save_path):
        with torch.no_grad():
            # print(sequence)
            output = self._folding_model.infer_pdb(sequence)
        
        with open(save_path, "w") as f:
            f.write(output)
        return output    
    
    def predict_step(self, batch, batch_idx):
        device = f'cuda:{torch.cuda.current_device()}'
        interpolant = Interpolant(self._infer_cfg.interpolant)
        interpolant.set_device(device)

        sample_length = batch['num_res'].item()
        diffuse_mask = torch.ones(1, sample_length)
        sample_id = batch['sample_id'].item()
        sample_dir = os.path.join(
            self._output_dir, f'length_{sample_length}', f'sample_{sample_id}')
        top_sample_csv_path = os.path.join(sample_dir, 'top_sample.csv')
        if os.path.exists(top_sample_csv_path):
            self._print_logger.info(
                f'Skipping instance {sample_id} length {sample_length}')
            return

        atom37_traj, model_traj, _ = interpolant.sample_clf(
            1, sample_length, self.model, self.cls_model
        )

        os.makedirs(sample_dir, exist_ok=True)
        bb_traj = to_numpy(torch.concat(atom37_traj, dim=0))
        traj_paths = save_traj(
            bb_traj[-1],
            bb_traj,
            np.flip(to_numpy(torch.concat(model_traj, dim=0)), axis=0),
            to_numpy(diffuse_mask)[0],
            output_dir=sample_dir,
        )
        
        # Run ProteinMPNN
        pdb_path = traj_paths["sample_path"]
        sc_output_dir = os.path.join(sample_dir, "self_consistency")
        os.makedirs(sc_output_dir, exist_ok=True)
        shutil.copy(
            pdb_path, os.path.join(sc_output_dir, os.path.basename(pdb_path))
        )
        
        # Run self consistency
        _ = self.run_self_consistency(sc_output_dir, pdb_path, motif_mask=None)
        
    def evaluate_structure_quality(self, pdb_path, reference_pdb_path=None, fixed_residues=None):
        """Evaluate the quality of a generated protein structure.
        
        Args:
            pdb_path: Path to the PDB file to evaluate
            reference_pdb_path: Optional path to a reference structure for comparison
            fixed_residues: List of residue indices that were fixed (PDB numbering)
            
        Returns:
            Dictionary containing various quality metrics:
            - Basic geometric validation (bond lengths, angles)
            - Secondary structure content
            - Ramachandran plot statistics
            - RMSD and TM-score to reference if provided
            - RMSD of fixed residues if provided
        """
        quality_metrics = {}
        
        # Calculate basic MDTraj metrics
        mdtraj_metrics = metrics.calc_mdtraj_metrics(pdb_path)
        quality_metrics.update(mdtraj_metrics)
        
        # Calculate CA-CA metrics
        sample_feats = du.parse_pdb_feats("sample", pdb_path)
        ca_idx = rc.atom_order['CA']
        
        # Debug logging
        logger = logging.getLogger(__name__)
        logger.info(f"PDB features keys: {sample_feats.keys()}")
        logger.info(f"bb_positions shape: {sample_feats['bb_positions'].shape}")
        logger.info(f"bb_positions type: {type(sample_feats['bb_positions'])}")
        
        # Ensure we have the correct shape for CA positions [N, 3]
        ca_positions = sample_feats["bb_positions"]
        if len(ca_positions.shape) == 1:
            # If we got a flattened array, reshape it
            ca_positions = ca_positions.reshape(-1, 3)
        elif len(ca_positions.shape) > 2:
            # If we got extra dimensions, take just the CA positions
            ca_positions = ca_positions[:, ca_idx]
            
        # Convert to numpy if needed
        if isinstance(ca_positions, torch.Tensor):
            ca_positions = ca_positions.detach().cpu().numpy()
            
        logger.info(f"Final ca_positions shape: {ca_positions.shape}")
            
        # Calculate CA-CA metrics
        ca_ca_metrics = metrics.calc_ca_ca_metrics(ca_positions)
        quality_metrics.update(ca_ca_metrics)
        
        # Compare to reference structure if provided
        if reference_pdb_path is not None:
            ref_feats = du.parse_pdb_feats("reference", reference_pdb_path)
            ref_seq = du.aatype_to_seq(ref_feats["aatype"])
            sample_seq = du.aatype_to_seq(sample_feats["aatype"])
            
            # Calculate TM-score
            _, tm_score = metrics.calc_tm_score(
                sample_feats["bb_positions"],
                ref_feats["bb_positions"],
                sample_seq,
                ref_seq
            )
            quality_metrics["tm_score"] = tm_score
            
            # Calculate RMSD
            rmsd = metrics.calc_aligned_rmsd(
                sample_feats["bb_positions"],
                ref_feats["bb_positions"]
            )
            quality_metrics["rmsd_to_ref"] = rmsd
            
            # Calculate RMSD for fixed residues if provided
            if fixed_residues is not None:
                # Map PDB residue numbers to indices in residue_index array
                sample_res_indices = sample_feats['residue_index']
                ref_res_indices = ref_feats['residue_index']
                sample_fixed_indices = []
                ref_fixed_indices = []
                for resnum in fixed_residues:
                    sample_matches = np.where(sample_res_indices == resnum)[0]
                    ref_matches = np.where(ref_res_indices == resnum)[0]
                    if len(sample_matches) == 0 or len(ref_matches) == 0:
                        logger.warning(f"Residue number {resnum} not found in sample or reference residue_indices.")
                        continue
                    sample_fixed_indices.append(sample_matches[0])
                    ref_fixed_indices.append(ref_matches[0])
                if sample_fixed_indices and ref_fixed_indices:
                    sample_fixed = sample_feats["bb_positions"][sample_fixed_indices]
                    ref_fixed = ref_feats["bb_positions"][ref_fixed_indices]
                    # Calculate RMSD for fixed residues
                    fixed_rmsd = metrics.calc_aligned_rmsd(sample_fixed, ref_fixed)
                    quality_metrics["fixed_residues_rmsd"] = fixed_rmsd
                    logger.info(f"RMSD for fixed residues {fixed_residues}: {fixed_rmsd:.3f} Å")
                else:
                    logger.warning(f"No valid fixed residue indices found for RMSD calculation.")
        
        return quality_metrics

    def analyze_sample_diversity(self, sample_pdbs, reference_pdb=None):
        """Analyze the diversity of a set of generated protein samples.
        
        Args:
            sample_pdbs: List of paths to PDB files of generated samples
            reference_pdb: Optional path to reference structure
            
        Returns:
            Dictionary containing diversity metrics:
            - Pairwise RMSD statistics between samples
            - RMSD to reference if provided
            - Structure clustering analysis
            - Secondary structure diversity
        """
        diversity_metrics = {}
        
        # Calculate all pairwise RMSDs between samples
        num_samples = len(sample_pdbs)
        pairwise_rmsds = []
        for i in range(num_samples):
            sample_i_feats = du.parse_pdb_feats("sample_i", sample_pdbs[i])
            for j in range(i+1, num_samples):
                sample_j_feats = du.parse_pdb_feats("sample_j", sample_pdbs[j])
                rmsd = metrics.calc_aligned_rmsd(
                    sample_i_feats["bb_positions"],
                    sample_j_feats["bb_positions"]
                )
                pairwise_rmsds.append(rmsd)
        
        diversity_metrics["mean_pairwise_rmsd"] = np.mean(pairwise_rmsds)
        diversity_metrics["std_pairwise_rmsd"] = np.std(pairwise_rmsds)
        diversity_metrics["min_pairwise_rmsd"] = np.min(pairwise_rmsds)
        diversity_metrics["max_pairwise_rmsd"] = np.max(pairwise_rmsds)
        
        # Compare all samples to reference if provided
        if reference_pdb is not None:
            ref_feats = du.parse_pdb_feats("reference", reference_pdb)
            ref_seq = du.aatype_to_seq(ref_feats["aatype"])
            ref_rmsds = []
            ref_tm_scores = []
            
            for sample_pdb in sample_pdbs:
                sample_feats = du.parse_pdb_feats("sample", sample_pdb)
                sample_seq = du.aatype_to_seq(sample_feats["aatype"])
                
                # Calculate RMSD to reference
                rmsd = metrics.calc_aligned_rmsd(
                    sample_feats["bb_positions"],
                    ref_feats["bb_positions"]
                )
                ref_rmsds.append(rmsd)
                
                # Calculate TM-score to reference
                _, tm_score = metrics.calc_tm_score(
                    sample_feats["bb_positions"],
                    ref_feats["bb_positions"],
                    sample_seq,
                    ref_seq
                )
                ref_tm_scores.append(tm_score)
            
            diversity_metrics["mean_rmsd_to_ref"] = np.mean(ref_rmsds)
            diversity_metrics["std_rmsd_to_ref"] = np.std(ref_rmsds)
            diversity_metrics["mean_tm_score_to_ref"] = np.mean(ref_tm_scores)
            diversity_metrics["std_tm_score_to_ref"] = np.std(ref_tm_scores)
        
        return diversity_metrics
        
    def prepare_conditional_inputs(self, pdb_path, fixed_residues=None, chain_id='A'):
        """Prepare inputs for conditional sampling from a PDB file.
        
        Args:
            pdb_path: Path to the PDB file containing the partial/reference structure
            fixed_residues: List of residue indices to fix (PDB numbering, e.g. 628, 629, ...)
            chain_id: Chain ID to use from the PDB file (default='A')
            
        Returns:
            Dictionary containing:
            - fixed_positions: [N, 3] tensor of fixed atom positions
            - fixed_mask: [N] boolean mask indicating which positions are fixed
            - num_res: Total number of residues
            - residue_indices: Original residue indices from PDB file
        """
        # Parse PDB features, excluding HETATM entries
        pdb_feats = du.parse_pdb_feats("reference", pdb_path, exclude_hetatm=True)
        
        # Get number of residues
        num_res = pdb_feats["aatype"].shape[0]
        device = next(self.parameters()).device
        
        # Debug logging
        logger = logging.getLogger(__name__)
        logger.info(f"PDB features keys: {pdb_feats.keys()}")
        
        # Get backbone positions - bb_positions should already be CA positions from parse_chain_feats
        if 'bb_positions' not in pdb_feats:
            raise ValueError("No backbone positions found in PDB features")
            
        fixed_positions = pdb_feats['bb_positions']  # Already [N, 3] for CA atoms
        
        # Get original residue indices from PDB
        residue_indices = pdb_feats['residue_index']
        
        # Create fixed mask
        if fixed_residues is None:
            # Fix all residues
            fixed_mask = torch.ones(num_res, dtype=torch.bool, device=device)
        else:
            # Map PDB residue numbers to indices in residue_indices array
            fixed_indices = []
            for resnum in fixed_residues:
                matches = np.where(residue_indices == resnum)[0]
                if len(matches) == 0:
                    raise ValueError(f"Residue number {resnum} not found in PDB residue_indices: {residue_indices}")
                fixed_indices.append(matches[0])
            fixed_mask = torch.zeros(num_res, dtype=torch.bool, device=device)
            fixed_mask[fixed_indices] = True
        
        # Convert positions to tensor and ensure correct shape [N, 3]
        fixed_positions = torch.tensor(fixed_positions, dtype=torch.float32, device=device)
        
        # Add debug information
        logger.info(f"fixed_positions shape: {fixed_positions.shape}")
        logger.info(f"fixed_mask shape: {fixed_mask.shape}")
        
        if len(fixed_positions.shape) != 2 or fixed_positions.shape[1] != 3:
            raise ValueError(f"Expected fixed_positions shape [N, 3], got {fixed_positions.shape}")
        
        return {
            'fixed_positions': fixed_positions,  # Shape: [N, 3]
            'fixed_mask': fixed_mask,           # Shape: [N]
            'num_res': num_res,
            'residue_indices': residue_indices
        }

    def sample_with_fixed_residues(
        self, 
        pdb_path, 
        fixed_residues=None, 
        num_samples=1,
        temperature=1.0,
        chain_id='A',
        output_dir=None,
        clf_model=None,
        guidance_scale=0.2,
        target_class=1
    ):
        """Generate protein samples while keeping specified residues fixed.
        
        Args:
            pdb_path: Path to PDB file with reference structure
            fixed_residues: List of residue indices to fix (1-indexed)
            num_samples: Number of samples to generate
            temperature: Temperature for sampling diversity
            chain_id: Chain ID to use from PDB
            output_dir: Directory to save samples (if None, uses self._sample_write_dir)
            clf_model: Optional classifier model for guidance
            guidance_scale: Scale factor for classifier guidance (default=0.2)
            target_class: Target class for classifier guidance (default=1)
        Returns:
            List of paths to generated PDB files
        """
        # Prepare inputs
        inputs = self.prepare_conditional_inputs(
            pdb_path, 
            fixed_residues=fixed_residues,
            chain_id=chain_id
        )
        
        # Initialize interpolant with current device
        device = next(self.parameters()).device
        self.interpolant.set_device(device)
        
        # Prepare inputs with correct dimensions
        # Add batch dimension to fixed_positions: [N, 3] -> [num_samples, N, 3]
        fixed_positions = inputs['fixed_positions'].unsqueeze(0).expand(num_samples, -1, -1)
        
        # Convert fixed_mask to float and add necessary dimensions
        fixed_mask = inputs['fixed_mask'].to(device)
        
        # Create additional model inputs
        batch = {
            'res_mask': torch.ones(num_samples, inputs['num_res'], device=device),
            'flow_mask': ~fixed_mask,  # Only flow non-fixed positions
            'fixed_positions': fixed_positions,
            'fixed_mask': fixed_mask,
        }
        
        # Run conditional sampling with optional classifier guidance
        atom37_traj, clean_atom37_traj, clean_traj = self.interpolant.sample_conditional(
            num_batch=num_samples,
            num_res=inputs['num_res'],
            model=self.model,
            fixed_positions=fixed_positions,  # Shape: [num_samples, N, 3]
            fixed_mask=fixed_mask,           # Shape: [N]
            temperature=temperature,
            clf_model=clf_model,
            guidance_scale=guidance_scale,
            target_class=target_class
        )
        
        # Save samples
        sample_paths = []
        save_dir = output_dir if output_dir is not None else self._sample_write_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Get original residue indices from input PDB
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure('ref', pdb_path)
        residue_indices = np.array([residue.id[1] for residue in structure[0][chain_id]])
        
        for i in range(num_samples):
            sample_path = os.path.join(
                save_dir,
                f'conditional_sample_{i}.pdb'
            )
            
            # Set b-factors to indicate fixed residues
            b_factors = torch.zeros((inputs['num_res'], 37), device=device)
            b_factors[fixed_mask] = 100.0
            
            # Convert tensors to numpy arrays before writing
            sample_coords = clean_atom37_traj[i].detach().cpu().numpy()
            b_factors = b_factors.detach().cpu().numpy()
            
            # Ensure sample_coords has shape [N, 37, 3] for a single model
            if len(sample_coords.shape) == 4:  # If shape is [1, N, 37, 3]
                sample_coords = sample_coords[0]  # Take first (and only) model
            
            # Create atom mask
            atom37_mask = np.sum(np.abs(sample_coords), axis=-1) > 1e-7
            
            # Create protein object with original residue indices
            prot = create_full_prot(
                sample_coords,
                atom37_mask,
                b_factors=b_factors,
                residue_indices=residue_indices
            )
            
            # Write protein to PDB
            pdb_str = protein.to_pdb(prot, model=1, add_end=False)
            with open(sample_path, 'w') as f:
                f.write(pdb_str)
                f.write('END\n')
            
            sample_paths.append(sample_path)
        
        return sample_paths
        