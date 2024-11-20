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


from utils import modelUtils as u
from models import ipa_pytorch

NM_TO_ANG_SCALE = 10.0
ANG_TO_NM_SCALE = 1 / NM_TO_ANG_SCALE


class ProteinFlow(nn.Module):
    def __init__(self, model_conf):
        super(ProteinFlow, self).__init__()
        self._model_conf = model_conf
        self._ipa_conf = model_conf.ipa
        # Convert angstrom to nm
        self.rigids_ang_to_nm = lambda x: x.apply_trans_fn(lambda x: x * ANG_TO_NM_SCALE)
        # Inverse
        self.rigids_nm_to_ang = lambda x: x.apply_trans_fn(lambda x: x * NM_TO_ANG_SCALE)
        self.node_embedder = NodeEmbedder(model_conf.node_features)
        self.edge_embedder = EdgeEmbedder(model_conf.edge_features)

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
            self.trunk[f'bb_update_{b}'] = ipa_pytorch.BackboneUpdate(
                self._ipa_conf.c_s, use_rot_updates=True
            )

            if b < self._ipa_conf.num_blocks - 1:
                # No edge update
                edge_in = self._model_conf.edge_embed_size
                self.trunk[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(
                    node_embed_size=self._ipa_conf.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self._model_conf.edge_embed_size,
                )

    def forward(self, input_features):
        # Get features
        node_mask = input_features['res_mask']
        edge_mask = node_mask[:, None] * node_mask[:, :, None]
        continuous_t = input_features['t']
        trans_t = input_features['trans_t']
        rotmats_t = input_features['rotmats_t']

        # Get embeddings
        init_node_embed = self.node_embedder(continuous_t, node_mask)
        if 'trans_sc' not in input_features:
            trans_sc = torch.zeros_like(trans_t)
        else:
            trans_sc = input_features['trans_sc']
        init_edge_embed = self.edge_embedder(
            init_node_embed, trans_t, trans_sc, edge_mask
        )

        curr_rigids = u.create_rigid(rotmats_t, trans_t)

        # Send to the trunk
        curr_rigids = self.rigids_ang_to_nm(curr_rigids)
        init_node_embed = init_node_embed * node_mask[..., None]
        node_embed = init_node_embed * node_mask[..., None]
        edge_embed = init_edge_embed * edge_mask[..., None]

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
            rigid_update = self.trunk[f'bb_update_{b}'](
                node_embed * node_mask[..., None])
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update, node_mask[..., None])

            if b < self._ipa_conf.num_blocks - 1:
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]

        curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans = curr_rigids.get_trans()
        pred_rotmats = curr_rigids.get_rots().get_rot_mats()
        return {
            'pred_trans': pred_trans,
            'pred_rotmats': pred_rotmats
        }


class NodeEmbedder(nn.Module):
    """
    node_features:
        c_s: ${model.node_embed_size}
        c_pos_emb: 128
        c_timestep_emb: 128
        embed_diffuse_mask: False
        max_num_res: 2000
        timestep_int: 1000
    """

    def __init__(self, module_cfg):
        super(NodeEmbedder, self).__init__()
        self._cfg = module_cfg
        self.c_s = self._cfg.c_s
        self.c_pos_emb = self._cfg.c_pos_emb
        self.c_timestep_emb = self._cfg.c_timestep_emb
        self.linear = nn.Linear(
            self._cfg.c_pos_emb + self._cfg.c_timestep_emb, self.c_s
        )

    def embed_t(self, timesteps, mask):
        timestep_emb = u.get_time_embedding(
            timesteps[:, 0],
            self.c_timestep_emb,
            max_positions=2056
        )
        timestep_emb = timestep_emb[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb * mask.unsqueeze(-1)

    def forward(self, timesteps, mask):
        # b: batch size
        b, num_res, device = mask.shape[0], mask.shape[1], mask.device

        pos = torch.arange(num_res, dtype=torch.float32).to(device)[None]
        pos_emb = u.get_index_embedding(pos, self.c_pos_emb, max_len=2056)
        pos_emb = pos_emb.repeat([b, 1, 1])
        pos_emb = pos_emb * mask.unsqueeze(-1)

        input_features = [pos_emb, self.embed_t(timesteps, mask)]
        return self.linear(torch.cat(input_features, dim=-1))


class EdgeEmbedder(nn.Module):
    """
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
    """

    def __init__(self, module_cfg):
        super(EdgeEmbedder, self).__init__()
        self._cfg = module_cfg

        self.c_s = self._cfg.c_s
        self.c_p = self._cfg.c_p
        self.feat_dim = self._cfg.feat_dim

        self.linear_s_p = nn.Linear(self.c_s, self.feat_dim)
        self.linear_relpos = nn.Linear(self.feat_dim, self.feat_dim)

        total_edge_feats = self.feat_dim * 3 + self._cfg.num_bins * 2
        self.edge_embedder = nn.Sequential(
            nn.Linear(total_edge_feats, self.c_p),
            nn.ReLU(),
            nn.Linear(self.c_p, self.c_p),
            nn.ReLU(),
            nn.Linear(self.c_p, self.c_p),
            nn.LayerNorm(self.c_p),
        )

    def embed_relpos(self, pos):
        rel_pos = pos[:, :, None] - pos[:, None, :]
        pos_emb = u.get_index_embedding(rel_pos, self._cfg.feat_dim, max_len=2056)
        return self.linear_relpos(pos_emb)

    def _cross_concat(self, feats_1d, num_batch, num_res):
        return torch.cat([
            torch.tile(feats_1d[:, :, None, :], (1, 1, num_res, 1)),
            torch.tile(feats_1d[:, None, :, :], (1, num_res, 1, 1)),
        ], dim=-1).float().reshape([num_batch, num_res, num_res, -1])

    def forward(self, s, t, sc_t, p_mask):
        num_batch, num_res, _ = s.shape
        p_i = self.linear_s_p(s)
        cross_node_feats = self._cross_concat(p_i, num_batch, num_res)
        pos = torch.arange(
            num_res, device=s.device).unsqueeze(0).repeat(num_batch, 1)
        relpos_feats = self.embed_relpos(pos)

        dist_feats = u.calc_distogram(
            t, min_bin=1e-3, max_bin=20.0, num_bins=self._cfg.num_bins)
        sc_feats = u.calc_distogram(
            sc_t, min_bin=1e-3, max_bin=20.0, num_bins=self._cfg.num_bins)

        all_edge_feats = torch.concat(
            [cross_node_feats, relpos_feats, dist_feats, sc_feats], dim=-1)
        edge_feats = self.edge_embedder(all_edge_feats)
        edge_feats *= p_mask.unsqueeze(-1)
        return edge_feats

