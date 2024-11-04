"""
https://github.com/ProteinDesignLab/protpardelle
License: MIT
Author: Alex Chu

Dataloader from PDB files.
"""
import logging

import hydra
import numpy as np
import torch
import math
import tree
from omegaconf import DictConfig
import pandas as pd

from openfold_data import data_transforms
import utils.openfold_rigid_utils as rigid_utils

from utils.pdbUtils import read_pkl, parse_chain_feats

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler, dist


class PdbDataModule(LightningDataModule):
    def __init__(self, data_cfg):
        super().__init__()
        self.data_cfg = data_cfg
        self.loader_cfg = data_cfg.loader
        self.dataset_cfg = data_cfg.dataset
        self.sampler_cfg = data_cfg.sampler

    def setup(self, stage: str):
        self._train_dataset = PdbDataset(
            dataset_cfg=self.dataset_cfg,
            is_training=True,
        )
        self._valid_dataset = PdbDataset(
            dataset_cfg=self.dataset_cfg,
            is_training=False
        )

    def train_dataloader(self, rank=None, num_replicas=None):
        num_workers = self.loader_cfg.num_workers
        return DataLoader(
            self._train_dataset,
            batch_sampler=LengthBatcher(
                sampler_cfg=self.sampler_cfg,
                metadata_csv=self._train_dataset.csv,
                rank=rank,
                num_replicas=num_replicas,
            ),
            num_workers=num_workers,
            prefetch_factor=None if num_workers == 0 else self.loader_cfg.prefetch_factor,
            pin_memory=False,
            persistent_workers=True if num_workers > 0 else False
        )

    def val_dataloader(self):
        return DataLoader(
            self._valid_dataset,
            #sampler=DistributedSampler(self._valid_dataset, shuffle=False),
            shuffle=False,
            num_workers=2,
            prefetch_factor=2,
            persistent_workers=True,
        )


class PdbDataset(Dataset):
    def __init__(
            self,
            *,
            dataset_cfg,
            is_training
    ):
        self._log = logging.getLogger(__name__)
        self._is_training = is_training
        self._dataset_cfg = dataset_cfg
        self._init_metadata()
        self._rng = np.random.default_rng(seed=self._dataset_cfg.seed)

    @property
    def is_training(self):
        return self._is_training

    @property
    def dataset_cfg(self):
        return self._dataset_cfg

    def _init_metadata(self):
        pdb_csv = pd.read_csv(self._dataset_cfg.csv_path)
        self.raw_csv = pdb_csv
        pdb_csv = pdb_csv[pdb_csv.modeled_seq_len <= self.dataset_cfg.max_num_res]
        pdb_csv = pdb_csv[pdb_csv.modeled_seq_len >= self.dataset_cfg.min_num_res]
        if self.dataset_cfg.subset is not None:
            pdb_csv = pdb_csv.iloc[:self.dataset_cfg.subset]
        pdb_csv = pdb_csv.sort_values('modeled_seq_len', ascending=False)

        if self.is_training:
            self.csv = pdb_csv
            self._log.info(
                f'Training: {len(self.csv)} examples.'
            )
        else:
            eval_csv = pdb_csv[pdb_csv.modeled_seq_len <= self.dataset_cfg.min_eval_length]
            all_lengths = np.sort(eval_csv.modeled_seq_len.unique())
            length_indices = (len(all_lengths) - 1) * np.linspace(
                0.0, 1.0, self.dataset_cfg.num_eval_lengths)
            length_indices = length_indices.astype(int)
            eval_lengths = all_lengths[length_indices]
            eval_csv = eval_csv[eval_csv.modeled_seq_len.isin(eval_lengths)]

            # Fix a random seed to get the same split each time.
            eval_csv = eval_csv.groupby('modeled_seq_len').sample(
                self.dataset_cfg.samples_per_eval_length, replace=True, random_state=123)
            eval_csv = eval_csv.sort_values('modeled_seq_len', ascending=False)
            self.csv = eval_csv
            self._log.info(
                f'Validation: {len(self.csv)} examples with lengths {eval_lengths}'
            )

    def _process_csv_row(self, processed_file_path):
        processed_features = read_pkl(processed_file_path)
        processed_features = parse_chain_feats(processed_features)

        modeled_idx = processed_features['modeled_idx']
        min_idx, max_idx = np.min(modeled_idx), np.max(modeled_idx)
        del processed_features['modeled_idx']
        processed_features = tree.map_structure(
            lambda x: x[min_idx:(max_idx + 1)], processed_features
        )

        chain_features = {
            'aatype': torch.tensor(processed_features['aatype']).long(),
            'all_atom_positions': torch.tensor(processed_features['atom_positions']).double(),
            'all_atom_mask': torch.tensor(processed_features['atom_mask']).double()
        }

        chain_features = data_transforms.atom37_to_frames(chain_features)
        rigids_1 = rigid_utils.Rigid.from_tensor_4x4(chain_features['rigidgroups_gt_frames'])[:, 0]
        rotmats_1 = rigids_1.get_rots().get_rot_mats()
        trans_1 = rigids_1.get_trans()
        res_idx = processed_features['residue_index']

        return {
            'aatype': chain_features['aatype'],
            'res_idx': res_idx - np.min(res_idx) + 1,
            'rotmats_1': rotmats_1,
            'trans_1': trans_1,
            'res_mask': torch.tensor(processed_features['bb_mask']).int(),
        }

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        example_idx = idx
        csv_row = self.csv.iloc[example_idx]
        class_idx = csv_row["class"]
        processed_file_path = csv_row['processed_path']
        chain_features = self._process_csv_row(processed_file_path)
        chain_features['csv_idx'] = torch.ones(1, dtype=torch.long) * idx
        chain_features["class"] = torch.ones(1, dtype=torch.long) * class_idx
        return chain_features


class LengthBatcher:

    def __init__(
            self,
            *,
            sampler_cfg,
            metadata_csv,
            seed=123,
            shuffle=True,
            num_replicas=None,
            rank=None,
    ):
        super().__init__()
        self.sample_order = None
        self._log = logging.getLogger(__name__)
        if num_replicas is None:
            self.num_replicas = 1
            # self.num_replicas = dist.get_world_size()
        else:
            self.num_replicas = num_replicas
        if rank is None:
            self.rank = 0
            #self.rank = dist.get_rank()
        else:
            self.rank = rank

        self._sampler_cfg = sampler_cfg
        self._data_csv = metadata_csv
        # Each replica needs the same number of batches. We set the number
        # of batches to arbitrarily be the number of examples per replica.
        self._num_batches = math.ceil(len(self._data_csv) / self.num_replicas)
        self._data_csv['index'] = list(range(len(self._data_csv)))
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        self.max_batch_size = self._sampler_cfg.max_batch_size
        self._log.info(f'Created dataloader rank {self.rank + 1} out of {self.num_replicas}')

    def _replica_epoch_batches(self):
        # Make sure all replicas share the same seed on each epoch.
        rng = torch.Generator()
        rng.manual_seed(self.seed + self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self._data_csv), generator=rng).tolist()
        else:
            indices = list(range(len(self._data_csv)))

        if len(self._data_csv) > self.num_replicas:
            replica_csv = self._data_csv.iloc[
                indices[self.rank::self.num_replicas]
            ]
        else:
            replica_csv = self._data_csv

        # Each batch contains multiple proteins of the same length.
        sample_order = []
        for seq_len, len_df in replica_csv.groupby('modeled_seq_len'):
            max_batch_size = min(
                self.max_batch_size,
                self._sampler_cfg.max_num_res_squared // seq_len ** 2 + 1,
            )
            num_batches = math.ceil(len(len_df) / max_batch_size)
            for i in range(num_batches):
                batch_df = len_df.iloc[i * max_batch_size:(i + 1) * max_batch_size]
                batch_indices = batch_df['index'].tolist()
                sample_order.append(batch_indices)

        # Remove any length bias.
        new_order = torch.randperm(len(sample_order), generator=rng).numpy().tolist()
        return [sample_order[i] for i in new_order]

    def _create_batches(self):
        # Make sure all replicas have the same number of batches Otherwise leads to bugs.
        # See bugs with shuffling https://github.com/Lightning-AI/lightning/issues/10947
        all_batches = []
        num_augments = -1
        while len(all_batches) < self._num_batches:
            all_batches.extend(self._replica_epoch_batches())
            num_augments += 1
            if num_augments > 1000:
                raise ValueError('Exceeded number of augmentations.')
        if len(all_batches) >= self._num_batches:
            all_batches = all_batches[:self._num_batches]
        self.sample_order = all_batches

    def __iter__(self):
        self._create_batches()
        self.epoch += 1
        return iter(self.sample_order)

    def __len__(self):
        return len(self.sample_order)


@hydra.main(version_base=None, config_path="../", config_name="train_config")
def my_app(cfg: DictConfig) -> None:
    data = PdbDataset(dataset_cfg=cfg.data.dataset, is_training=True)
    print(len(data))
    print(data[0])


if __name__ == '__main__':
    my_app()
