from dataset.data import PdbDataset
from dataset.data import PdbDataModule
import hydra
from omegaconf import DictConfig
from torchsummary import summary
from utils.flows import Interpolant
import numpy as np
from utils.modelUtils import get_time_embedding

"""
ProteinFlow model
model:
  node_embed_size: 256
  edge_embed_size: 128
  symmetric: False
  node_features:
    c_s: ${model.node_embed_size}
    c_pos_emb: 128
    c_timestep_emb: 128
    embed_diffuse_mask: False
    max_num_res: 2000
    timestep_int: 1000
  edge_features:
    single_bias_transition_n: 2
    c_s: ${model.node_embed_size}
    c_p: ${model.edge_embed_size}
    relpos_k: 64
    use_rbf: True
    num_rbf: 32
    feat_dim: 64
    num_bins: 22
    self_condition: True
  ipa:
    c_s: ${model.node_embed_size}
    c_z: ${model.edge_embed_size}
    c_hidden: 128
    no_heads: 8
    no_qk_points: 8
    no_v_points: 12
    seq_tfmr_num_heads: 4
    seq_tfmr_num_layers: 2
    num_blocks: 6
"""

import torch
from torch import nn

from models.classifier import ProtClassifier
from utils import modelUtils as u
from models import ipa_pytorch


@hydra.main(version_base=None, config_path=".", config_name="test_dataset")
def run(cfg: DictConfig) -> None:
    data = PdbDataset(dataset_cfg=cfg.data.dataset, is_training=True)
    datamodule = PdbDataModule(cfg.data)
    datamodule.setup("")
    train_loader = datamodule.train_dataloader()
    interpolant = Interpolant(cfg.interpolant)
    interpolant.set_device("cpu")
    
    print(len(data))
    # print(data[0])
    print(data[0].keys())
    for i in range(3):
        batch = next(iter(train_loader))
        print(batch["trans_1"].shape)
        noisy_batch = interpolant.corrupt_batch(batch)
        
        model = ProtClassifier(cfg.model)
        model(noisy_batch)
    
    """
    print(noisy_batch["class"].shape)
    print(noisy_batch["t"].shape)
    print(noisy_batch["trans_t"].shape)
    print(noisy_batch["rotmats_t"].shape)
    """
    #time_emb = get_time_embedding(noisy_batch["t"][:, 0], 128)
    #print(time_emb.shape)
    
    # print(summary(model, data[0].shape))
run()


"""
dirname = "class_preprocessed"
df = pd.read_csv(os.path.join(dirname, "metadata.csv"))
print(df.shape[0])

pos_df = df[df["class"] == 1]
neg_df = df[df["class"] == 0]
print(pos_df.shape[0])
print(neg_df.shape[0])

filtered_df = df[df["modeled_seq_len"] <= 256]
filtered_pos_df = filtered_df[filtered_df["class"] == 1]
filtered_neg_df = filtered_df[filtered_df["class"] == 0]
print(filtered_df.shape[0])
print(filtered_pos_df.shape[0])
print(filtered_neg_df.shape[0])
"""