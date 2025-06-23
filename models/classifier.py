import torch
from torch import nn


from utils import modelUtils as u

from models.proteinflow import NodeEmbedder, EdgeEmbedder
from models import ipa_pytorch

import torch.nn.functional as F

NM_TO_ANG_SCALE = 10.0
ANG_TO_NM_SCALE = 1 / NM_TO_ANG_SCALE

class ProtClassifier(nn.Module):
    def __init__(self, model_conf):
        super(ProtClassifier, self).__init__()
        self._model_conf = model_conf
        self._ipa_conf = model_conf.ipa
        # Convert angstrom to nm
        self.rigids_ang_to_nm = lambda x: x.apply_trans_fn(lambda x: x * ANG_TO_NM_SCALE)
        # Inverse
        self.rigids_nm_to_ang = lambda x: x.apply_trans_fn(lambda x: x * NM_TO_ANG_SCALE)
        self.node_embedder = NodeEmbedder(model_conf.node_features)
        self.edge_embedder = EdgeEmbedder(model_conf.edge_features)
        
        # Attention trunk
        # self.ipa_embedder = ipa_pytorch.InvariantPointAttention(self._ipa_conf)
        
        # Attention trunk
        self.trunk = nn.ModuleDict()
        for b in range(self._ipa_conf.num_blocks):
            self.trunk[f'ipa_{b}'] = ipa_pytorch.InvariantPointAttention(self._ipa_conf)
            self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)
            tfmr_in = self._ipa_conf.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False
            )
            self.trunk[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
                tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False
            )
            self.trunk[f'post_tfmr_{b}'] = ipa_pytorch.Linear(
                tfmr_in, self._ipa_conf.c_s, init='final'
            )
            self.trunk[f'node_transition_{b}'] = ipa_pytorch.StructureModuleTransition(
                c=self._ipa_conf.c_s
            )

            if b < self._ipa_conf.num_blocks - 1:
                # No edge update
                edge_in = self._model_conf.edge_embed_size
                self.trunk[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(
                    node_embed_size=self._ipa_conf.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self._model_conf.edge_embed_size,
                )
        # 8454144
        self.classifier_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*384, 128),
            nn.ReLU(),
            # nn.Linear(32768, 128),
            # nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),         
        )
    
    def forward(self, input_features):
        
        # Get features
        node_mask = input_features['res_mask']
        padding_amount = 256 - node_mask.shape[1]
        
        node_mask = F.pad(node_mask, pad=(0,padding_amount,0,0))
        edge_mask = node_mask[:, None] * node_mask[:, :, None]
        
        continuous_t = input_features['t']
        
        trans_t = input_features['trans_t']
        trans_t = F.pad(trans_t, pad=(0,0,0,padding_amount,0,0))
        rotmats_t = input_features['rotmats_t']
        rotmats_t = F.pad(rotmats_t, pad=(0,0,0,0,0,padding_amount,0,0))

        # Get embeddings
        init_node_embed = self.node_embedder(continuous_t, node_mask)
        if 'trans_sc' not in input_features:
            trans_sc = torch.zeros_like(trans_t)
        else:
            trans_sc = input_features['trans_sc']
            trans_sc = F.pad(trans_sc, pad=(0,0,0,padding_amount,0,0))
        init_edge_embed = self.edge_embedder(
            init_node_embed, trans_t, trans_sc, edge_mask
        )
        # print(f"init_node_embed: {init_node_embed.shape}")
        # print(f"init_edge_embed: {init_edge_embed.shape}")
        
        curr_rigids = u.create_rigid(rotmats_t, trans_t)
        
        curr_rigids = self.rigids_ang_to_nm(curr_rigids)
        init_node_embed = init_node_embed * node_mask[..., None]
        node_embed = init_node_embed * node_mask[..., None]
        edge_embed = init_edge_embed * edge_mask[..., None]
        
        # print(f"node_embed: {node_embed.shape}")
        # print(f"edge_embed: {edge_embed.shape}")
        
        for b in range(self._ipa_conf.num_blocks):
            ipa_embed = self.trunk[f'ipa_{b}'](
                node_embed,
                edge_embed,
                curr_rigids,
                node_mask
            )
            ipa_embed *= node_mask[..., None]
            node_embed = self.trunk[f'ipa_ln_{b}'](node_embed + ipa_embed)
            seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](
                node_embed, src_key_padding_mask=(1 - node_mask).to(torch.bool))
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = self.trunk[f'node_transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]

            if b < self._ipa_conf.num_blocks - 1:
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]
            
            # print(f"node_embed_{b}: {node_embed.shape}")
            # print(f"edge_embed_{b}: {edge_embed.shape}")
            # print(f"ipa_embed_{b}: {ipa_embed.shape}")
            # print(f"ipa_flatten_{b}: {nn.Flatten()(ipa_embed).shape}")
        # print()
        # print(f"ipa_embed grad_fn?: {ipa_embed.grad_fn}")
        # print(f"ipa_embed req_grad?: {ipa_embed.requires_grad}")
        # print(f"node_embed grad_fn?: {node_embed.grad_fn}")
        # print(f"node_embed req_grad?: {node_embed.requires_grad}")
        # print(f"edge_embed grad_fn?: {edge_embed.grad_fn}")
        # print(f"edge_embed req_grad?: {edge_embed.requires_grad}")
        # print()
        edge_embed_mean = torch.mean(edge_embed, dim=2)
        fused_tensor = torch.cat((ipa_embed, node_embed, edge_embed_mean), dim=-1)
        x = self.classifier_head(fused_tensor)
        # x = torch.nn.functional.softmax(self.classifier_head(ipa_embed), dim=-1)
        # print(f"Classifier output: {x.shape}")
        # print(f"Classifier output grad_fn?: {x.grad_fn}")
        # print(f"Classifier output req_grad?: {x.requires_grad}")
        return x