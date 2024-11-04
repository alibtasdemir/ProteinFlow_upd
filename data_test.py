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
import torch.utils
import torch.utils.data
import tree
from omegaconf import DictConfig
import pandas as pd

from openfold_data import data_transforms
import utils.openfold_rigid_utils as rigid_utils

from utils.pdbUtils import read_pkl, parse_chain_feats

from torch.utils.data import DataLoader, Dataset

from sklearn.model_selection import train_test_split


def get_dataloaders(cfg):
    dataset_cfg = cfg.dataset
    loader_cfg = cfg.loader
    
    num_workers = loader_cfg.num_workers
    
    pdb_csv = pd.read_csv(dataset_cfg.csv_path)
    pdb_csv = pdb_csv[pdb_csv.modeled_seq_len <= dataset_cfg.max_num_res]
    pdb_csv = pdb_csv[pdb_csv.modeled_seq_len >= dataset_cfg.min_num_res]
    
    print(pdb_csv["class"].value_counts())
    
    train_data, test_data = train_test_split(pdb_csv, test_size=0.2, shuffle=True)
    
    train = PdbDataset(
        train_data,
        dataset_cfg,
        is_training=True
    )
    test = PdbDataset(
        test_data,
        dataset_cfg,
        is_training=False
    )
    
    train_loader = DataLoader(
        train,
        batch_size=loader_cfg.batch_size,
        num_workers=num_workers,
        prefetch_factor=None if num_workers == 0 else loader_cfg.prefetch_factor,
        pin_memory=False,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
            test,
            shuffle=False,
            num_workers=2,
            prefetch_factor=2,
            persistent_workers=True
        )
    
    return train_loader, val_loader


class PdbDataset(Dataset):
    def __init__(
            self,
            dataset,
            dataset_cfg,
            is_training
    ):
        self.pdb_csv = dataset
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
        self.pdb_csv = self.pdb_csv.sort_values('modeled_seq_len', ascending=False)
        self.csv = self.pdb_csv
        self._log.info(
            f'Dataset: {len(self.csv)} examples.'
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


@hydra.main(version_base=None, config_path="./configs", config_name="classifier.yaml")
def my_app(cfg: DictConfig) -> None:
    train, test = get_dataloaders(cfg.data)
    print(next(iter(train)))
    print(test.shape)
    """
    data = PdbClfDataModule(cfg.data)
    data.setup('train')
    train_loader = data.train_dataloader()
    val_loader = data.val_dataloader()
    # data = PdbDataset(dataset_cfg=cfg.data.dataset, is_training=True)
    print(train_loader)
    print(val_loader)
    #print(data[0])
    """


if __name__ == '__main__':
    my_app()
