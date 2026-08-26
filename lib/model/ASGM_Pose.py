import math
import logging
from functools import partial
from collections import OrderedDict
from einops import rearrange, repeat
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import time

from math import sqrt
import os
import sys


current_directory = os.path.dirname(__file__) + '/../' + '../'
sys.path.append(current_directory)
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from lib.model.drop import DropPath
from timm.models.registry import register_model
import torch.nn.functional as F
from functools import partial
import torch.fft

from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math
import numpy as np

from lib.model.mambablocks import BiSTSSMBlock


CONNECTIONS = {10: [9], 9: [8, 10], 8: [7, 9], 14: [15, 8], 15: [16, 14], 11: [12, 8], 12: [13, 11],
               7: [0, 8], 0: [1, 7], 1: [2, 0], 2: [3, 1], 4: [5, 0], 5: [6, 4], 16: [15], 13: [12], 3: [2], 6: [5]}


class GCN(nn.Module):
    def __init__(self, dim_in, dim_out, num_nodes, neighbour_num=4, mode='spatial', use_temporal_similarity=True,
                 temporal_connection_len=1, connections=None, norm_layer=None):
        super().__init__()
        assert mode in ['spatial', 'temporal'], "Mode is undefined"

        self.relu = nn.ReLU()
        self.neighbour_num = neighbour_num
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.mode = mode
        self.use_temporal_similarity = use_temporal_similarity
        self.num_nodes = num_nodes
        self.connections = connections

        self.U = nn.Linear(self.dim_in, self.dim_out)
        self.V = nn.Linear(self.dim_in, self.dim_out)
        self.batch_norm = nn.BatchNorm1d(self.num_nodes)

        self._init_gcn()

        if mode == 'spatial':
            self.adj = self._init_spatial_adj()
        elif mode == 'temporal' and not self.use_temporal_similarity:
            self.adj = self._init_temporal_adj(temporal_connection_len)

        if norm_layer is not None:
            self.batch_norm = norm_layer(self.num_nodes)

    def _init_gcn(self):
        self.U.weight.data.normal_(0, math.sqrt(2. / self.dim_in))
        self.V.weight.data.normal_(0, math.sqrt(2. / self.dim_in))
        self.batch_norm.weight.data.fill_(1)
        self.batch_norm.bias.data.zero_()

    def _init_spatial_adj(self):
        adj = torch.zeros((self.num_nodes, self.num_nodes))
        connections = self.connections if self.connections is not None else CONNECTIONS

        for i in range(self.num_nodes):
            connected_nodes = connections[i]
            for j in connected_nodes:
                adj[i, j] = 1
        return adj

    def _init_temporal_adj(self, connection_length):
        adj = torch.zeros((self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes):
            try:
                for j in range(connection_length + 1):
                    adj[i, i + j] = 1
            except IndexError:
                pass
        return adj

    @staticmethod
    def normalize_digraph(adj):
        b, n, c = adj.shape
        node_degrees = adj.detach().sum(dim=-1)
        deg_inv_sqrt = node_degrees ** -0.5
        norm_deg_matrix = torch.eye(n)
        dev = adj.get_device()
        if dev >= 0:
            norm_deg_matrix = norm_deg_matrix.to(dev)
        norm_deg_matrix = norm_deg_matrix.view(1, n, n) * deg_inv_sqrt.view(b, n, 1)
        norm_adj = torch.bmm(torch.bmm(norm_deg_matrix, adj), norm_deg_matrix)
        return norm_adj

    def change_adj_device_to_cuda(self, adj):
        dev = self.V.weight.get_device()
        if dev >= 0 and adj.get_device() < 0:
            adj = adj.to(dev)
        return adj

    def forward(self, x):
        b, t, j, c = x.shape
        if self.mode == 'temporal':
            x = x.transpose(1, 2)
            x = x.reshape(-1, t, c)
            if self.use_temporal_similarity:
                similarity = x @ x.transpose(1, 2)
                threshold = similarity.topk(k=self.neighbour_num, dim=-1, largest=True)[0][..., -1].view(b * j, t, 1)
                adj = (similarity >= threshold).float()
            else:
                adj = self.adj
                adj = self.change_adj_device_to_cuda(adj)
                adj = adj.repeat(b * j, 1, 1)
        else:
            x = x.reshape(-1, j, c)
            adj = self.adj
            adj = self.change_adj_device_to_cuda(adj)
            adj = adj.repeat(b * t, 1, 1)

        norm_adj = self.normalize_digraph(adj)
        aggregate = norm_adj @ self.V(x)

        if self.dim_in == self.dim_out:
            x = self.relu(x + self.batch_norm(aggregate + self.U(x)))
        else:
            x = self.relu(self.batch_norm(aggregate + self.U(x)))

        x = x.reshape(-1, t, j, self.dim_out) if self.mode == 'spatial' \
            else x.reshape(-1, j, t, self.dim_out).transpose(1, 2)
        return x



def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def cluster_dpc_knn(x, cluster_num, k, token_mask=None):
    with torch.no_grad():
        B, N, C = x.shape
        dist_matrix = torch.cdist(x, x) / (C ** 0.5)

        if token_mask is not None:
            token_mask = token_mask > 0
            dist_matrix = dist_matrix * token_mask[:, None, :] + (dist_matrix.max() + 1) * (~token_mask[:, None, :])

        dist_nearest, index_nearest = torch.topk(dist_matrix, k=k, dim=-1, largest=False)
        density = (-(dist_nearest ** 2).mean(dim=-1)).exp()
        density = density + torch.rand(density.shape, device=density.device, dtype=density.dtype) * 1e-6

        if token_mask is not None:
            density = density * token_mask

        mask = density[:, None, :] > density[:, :, None]
        mask = mask.type(x.dtype)
        dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
        dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)

        score = dist * density
        _, index_down = torch.topk(score, k=cluster_num, dim=-1)

        dist_matrix = index_points(dist_matrix, index_down)
        idx_cluster = dist_matrix.argmin(dim=1)

        idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)
        idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)
        idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)

    return index_down, idx_cluster



class Cross_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., length=27):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.linear_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.linear_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.linear_v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_1, x_2, x_3):
        B, N, C = x_1.shape
        B, N_1, C = x_3.shape

        q = self.linear_q(x_1).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.linear_k(x_2).reshape(B, N_1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.linear_v(x_3).reshape(B, N_1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class AVWGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim,
                 graph_edge_mode='signed', graph_degree_mode='algebraic',
                 graph_filter_mode='node_adaptive'):
        super(AVWGCN, self).__init__()
        self.cheb_k = cheb_k
        valid_edge_modes = {'signed', 'positive'}
        valid_degree_modes = {'algebraic', 'absolute'}
        if graph_edge_mode not in valid_edge_modes:
            raise ValueError(
                f'Unsupported graph_edge_mode={graph_edge_mode!r}. '
                f'Expected one of {sorted(valid_edge_modes)}.'
            )
        if graph_degree_mode not in valid_degree_modes:
            raise ValueError(
                f'Unsupported graph_degree_mode={graph_degree_mode!r}. '
                f'Expected one of {sorted(valid_degree_modes)}.'
            )
        self.graph_edge_mode = graph_edge_mode
        self.graph_degree_mode = graph_degree_mode
        valid_filter_modes = {'node_adaptive', 'shared'}
        if graph_filter_mode not in valid_filter_modes:
            raise ValueError(
                f'Unsupported graph_filter_mode={graph_filter_mode!r}. '
                f'Expected one of {sorted(valid_filter_modes)}.'
            )
        self.graph_filter_mode = graph_filter_mode
        if graph_filter_mode == 'node_adaptive':
            self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
            self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
            nn.init.xavier_uniform_(self.weights_pool)
            nn.init.zeros_(self.bias_pool)
        else:
            self.weights_shared = nn.Parameter(torch.FloatTensor(cheb_k, dim_in, dim_out))
            self.bias_shared = nn.Parameter(torch.FloatTensor(dim_out))
            nn.init.xavier_uniform_(self.weights_shared)
            nn.init.zeros_(self.bias_shared)

    def normalize_adjacency(self, adj):
        """Apply the configured edge transform and symmetric normalization."""
        if self.graph_edge_mode == 'positive':
            adj = torch.relu(adj)


        adj = adj + torch.eye(adj.shape[0], device=adj.device)


        if self.graph_degree_mode == 'absolute':
            degree = adj.abs().sum(dim=1)
        else:
            degree = adj.sum(dim=1)


        degree = torch.clamp(degree, min=1e-8)


        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt = torch.diag(degree_inv_sqrt)

        normalized_adj = torch.mm(torch.mm(degree_inv_sqrt, adj), degree_inv_sqrt)

        return normalized_adj

    def forward(self, x, node_embeddings):
        node_num = node_embeddings.shape[0]


        if torch.isnan(node_embeddings).any() or torch.isinf(node_embeddings).any():
            print("Warning: node_embeddings contains NaN or Inf values")
            node_embeddings = torch.nan_to_num(node_embeddings, nan=0.0, posinf=1.0, neginf=-1.0)


        adj_matrix = torch.mm(node_embeddings, node_embeddings.transpose(0, 1))




        supports = self.normalize_adjacency(adj_matrix)


        if torch.isnan(supports).any() or torch.isinf(supports).any():
            print("Warning: supports contains NaN or Inf values")
            supports = torch.nan_to_num(supports, nan=0.0, posinf=1.0, neginf=0.0)
            supports = self.normalize_adjacency(supports)

        support_set = [torch.eye(node_num).to(supports.device), supports]


        for k in range(2, self.cheb_k):
            next_support = torch.matmul(2 * supports, support_set[-1]) - support_set[-2]


            if torch.isnan(next_support).any() or torch.isinf(next_support).any():
                print(f"Warning: support {k} contains NaN or Inf values")
                next_support = torch.nan_to_num(next_support, nan=0.0, posinf=1.0, neginf=-1.0)

            support_set.append(next_support)

        supports = torch.stack(support_set, dim=0)


        if self.graph_filter_mode == 'node_adaptive':
            weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)
            bias = torch.matmul(node_embeddings, self.bias_pool)
        else:
            weights = self.weights_shared
            bias = self.bias_shared


        if torch.isnan(weights).any() or torch.isinf(weights).any():
            print("Warning: weights contains NaN or Inf values")
            weights = torch.nan_to_num(weights, nan=0.0, posinf=1.0, neginf=-1.0)

        if torch.isnan(bias).any() or torch.isinf(bias).any():
            print("Warning: bias contains NaN or Inf values")
            bias = torch.nan_to_num(bias, nan=0.0, posinf=1.0, neginf=-1.0)


        x_g = torch.einsum("knm,bmc->bknc", supports, x)
        x_g = x_g.permute(0, 2, 1, 3)
        if self.graph_filter_mode == 'node_adaptive':
            x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias
        else:
            x_gconv = torch.einsum('bnki,kio->bno', x_g, weights) + bias


        if torch.isnan(x_gconv).any() or torch.isinf(x_gconv).any():
            print("Warning: x_gconv contains NaN or Inf values")
            x_gconv = torch.nan_to_num(x_gconv, nan=0.0, posinf=1.0, neginf=-1.0)

        return x_gconv


class TemporalTransformerBlock(nn.Module):
    """Lightweight temporal Transformer applied independently to each joint."""

    def __init__(self, hidden_dim, mlp_ratio=2.0, num_heads=8, drop=0.0):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f'hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads}).'
            )
        self.encoder = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=int(hidden_dim * mlp_ratio),
            dropout=drop,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )

    def forward(self, x):
        b, f, n, c = x.shape
        x = rearrange(x, 'b f n c -> (b n) f c')
        x = self.encoder(x)
        return rearrange(x, '(b n) f c -> b f n c', b=b, n=n)


class ASGM_Pose(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=256, depth=6, mlp_ratio=2.,
                 drop_rate=0., drop_path_rate=0.2, norm_layer=None, token_num=None, layer_index=None,
                 selection_method='dpc', graph_edge_mode='signed', graph_degree_mode='algebraic',
                 graph_filter_mode='node_adaptive', token_ablation_mode='selection_restoration',
                 temporal_block_type='mamba', transformer_num_heads=8):
        """HoT ASGM-Pose: Hourglass Tokenizer for Efficient ASGM-Pose
        Args:
            num_frame (int): input frame number
            num_joints (int): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            drop_rate (float): dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
            token_num (int): number of tokens after pruning
            layer_index (int): layer index for token pruning
            selection_method (str): frame selector: dpc, uniform, strided, random, or motion
            graph_edge_mode (str): signed Gram adjacency or positive-part adjacency
            graph_degree_mode (str): algebraic or absolute degree for symmetric normalization
            graph_filter_mode (str): node-adaptive or node-shared graph filters
            token_ablation_mode (str): selection-restoration path, restoration-only,
                or selection-only with fixed interpolation
            temporal_block_type (str): mamba or transformer temporal blocks
            transformer_num_heads (int): attention heads for transformer blocks
        """
        super().__init__()

        norm_layer = nn.LayerNorm
        embed_dim = embed_dim_ratio
        out_dim = 3


        self.token_num = token_num if token_num is not None else num_frame // 3
        self.layer_index = layer_index if layer_index is not None else depth // 2
        self.recover_num = num_frame
        self.center = (num_frame - 1) // 2
        valid_selection_methods = {'dpc', 'uniform', 'strided', 'random', 'motion'}
        if selection_method not in valid_selection_methods:
            raise ValueError(
                f'Unsupported selection_method={selection_method!r}. '
                f'Expected one of {sorted(valid_selection_methods)}.'
            )
        self.selection_method = selection_method

        valid_token_modes = {'selection_restoration', 'restoration_only', 'selection_only'}
        if token_ablation_mode not in valid_token_modes:
            raise ValueError(
                f'Unsupported token_ablation_mode={token_ablation_mode!r}. '
                f'Expected one of {sorted(valid_token_modes)}.'
            )
        self.token_ablation_mode = token_ablation_mode
        self.enable_selection = token_ablation_mode != 'restoration_only'
        self.enable_learned_restoration = token_ablation_mode != 'selection_only'
        if not self.enable_selection and self.token_num != self.recover_num:
            raise ValueError(
                'restoration_only requires token_num to equal num_frame so that M=T.'
            )

        valid_temporal_blocks = {'mamba', 'transformer'}
        if temporal_block_type not in valid_temporal_blocks:
            raise ValueError(
                f'Unsupported temporal_block_type={temporal_block_type!r}. '
                f'Expected one of {sorted(valid_temporal_blocks)}.'
            )
        self.temporal_block_type = temporal_block_type
        self.transformer_num_heads = int(transformer_num_heads)


        self.pool = nn.AdaptiveAvgPool1d(1)
        if self.enable_selection:
            self.pos_embed_token = nn.Parameter(torch.zeros(1, self.token_num, embed_dim))
        else:
            self.register_parameter('pos_embed_token', None)


        if self.enable_learned_restoration:
            self.x_token = nn.Parameter(torch.zeros(1, self.recover_num, embed_dim))
            self.cross_attention = Cross_Attention(embed_dim, num_heads=8, qkv_bias=True,
                                                   qk_scale=None, attn_drop=0., proj_drop=drop_rate)
        else:
            self.register_parameter('x_token', None)
            self.cross_attention = None


        self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.block_depth = depth

        self.node_embeddings = nn.Parameter(torch.randn(num_joints, embed_dim_ratio), requires_grad=True)
        self.spatial_gcn = AVWGCN(
            embed_dim_ratio,
            embed_dim_ratio,
            cheb_k=3,
            embed_dim=embed_dim_ratio,
            graph_edge_mode=graph_edge_mode,
            graph_degree_mode=graph_degree_mode,
            graph_filter_mode=graph_filter_mode,
        )
        self.spatial_gcn.adj_matrix_saved = None

        if self.temporal_block_type == 'mamba':
            self.TTEblocks = nn.ModuleList([
                BiSTSSMBlock(
                    hidden_dim=embed_dim,
                    mlp_ratio=mlp_ratio,
                    drop_path=dpr[i],
                    norm_layer=nn.LayerNorm,
                    forward_type='v2_plus_poselimbs'
                )
                for i in range(depth)])
        else:
            self.TTEblocks = nn.ModuleList([
                TemporalTransformerBlock(
                    hidden_dim=embed_dim,
                    mlp_ratio=mlp_ratio,
                    num_heads=self.transformer_num_heads,
                    drop=drop_rate,
                )
                for _ in range(depth)])

        self.Spatial_norm = nn.LayerNorm(embed_dim_ratio)
        self.Temporal_norm = nn.LayerNorm(embed_dim)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_dim),
        )

        self.num_joints = num_joints
        self.embed_dim_ratio = embed_dim_ratio

    def STE_forward(self, x):
        b, f, n, c = x.shape
        x = rearrange(x, 'b f n c -> (b f) n c')


        x = self.Spatial_patch_to_embedding(x)


        x_gcn = self.spatial_gcn(x, self.node_embeddings)

        x = x_gcn.view(b, f, n, -1)
        x += self.Spatial_pos_embed
        x = self.pos_drop(x)
        x = self.Spatial_norm(x)
        return x

    def TTE_foward(self, x):
        b, f, n, c = x.shape
        x = rearrange(x, 'b f n cw -> (b n) f cw', f=f)
        x += self.Temporal_pos_embed[:, :f, :]
        x = self.pos_drop(x)
        x = rearrange(x, '(b n) f cw -> b f n cw', n=n)
        blk = self.TTEblocks[0]
        x = blk(x)
        x = self.Temporal_norm(x)
        return x

    def select_frame_indices(self, x):
        """Return sorted frame indices for one selector ablation condition."""
        b, f, n, c = x.shape
        if self.token_num > f:
            raise ValueError(f'token_num ({self.token_num}) exceeds current length ({f}).')

        if self.selection_method == 'dpc':
            x_knn = rearrange(x, 'b f n c -> b (f c) n')
            x_knn = self.pool(x_knn)
            x_knn = rearrange(x_knn, 'b (f c) 1 -> b f c', f=f)
            index, _ = cluster_dpc_knn(x_knn, self.token_num, 2)
        elif self.selection_method == 'uniform':
            base_index = torch.linspace(0, f - 1, self.token_num, device=x.device).round().long()
            index = base_index.unsqueeze(0).expand(b, -1)
        elif self.selection_method == 'strided':
            stride = max(f // self.token_num, 1)
            base_index = torch.arange(self.token_num, device=x.device) * stride
            index = base_index.unsqueeze(0).expand(b, -1)
        elif self.selection_method == 'random':
            random_scores = torch.rand(b, f, device=x.device)
            index = random_scores.topk(self.token_num, dim=-1).indices
        else:
            descriptors = x.mean(dim=2)
            velocity = torch.norm(descriptors[:, 1:] - descriptors[:, :-1], dim=-1)
            motion_score = torch.cat((velocity[:, :1], velocity), dim=1)
            index = motion_score.topk(self.token_num, dim=-1).indices

        return torch.sort(index, dim=-1).values

    def ST_foward(self, x):
        assert len(x.shape) == 4, "shape is equal to 4"
        b, f, n, cw = x.shape
        selected_indices = None

        for i in range(1, self.block_depth):

            if self.enable_selection and i == self.layer_index:
                selected_indices = self.select_frame_indices(x)

                batch_ind = torch.arange(b, device=x.device).unsqueeze(-1)
                x = x[batch_ind, selected_indices]

                x = rearrange(x, 'b f n c -> (b n) f c')
                x += self.pos_embed_token
                x = rearrange(x, '(b n) f c -> b f n c', n=n)




            tteblock = self.TTEblocks[i]
            x = tteblock(x)
            x = self.Temporal_norm(x)

        return x, selected_indices

    def interpolate_selected_features(self, x, selected_indices):
        """Restore T frames by piecewise-linear interpolation on original indices."""
        if selected_indices is None:
            raise RuntimeError('selection_only requires frame indices from the selection stage.')

        b, m, n, c = x.shape
        if selected_indices.shape != (b, m):
            raise ValueError(
                f'Expected selected_indices shape {(b, m)}, got {tuple(selected_indices.shape)}.'
            )
        if m < 1:
            raise ValueError('selection_only requires at least one retained frame token.')
        if m > 1 and not torch.all(selected_indices[:, 1:] > selected_indices[:, :-1]):
            raise ValueError('Selected frame indices must be strictly increasing for interpolation.')
        if m == 1:
            return x.expand(-1, self.recover_num, -1, -1)

        target_indices = torch.arange(
            self.recover_num, device=x.device, dtype=selected_indices.dtype
        ).unsqueeze(0).expand(b, -1)
        right = torch.searchsorted(
            selected_indices.contiguous(), target_indices.contiguous(), right=False
        ).clamp(max=m - 1)
        left = (right - 1).clamp(min=0)

        first_mask = target_indices <= selected_indices[:, :1]
        last_mask = target_indices >= selected_indices[:, -1:]
        first_position = torch.zeros_like(left)
        last_position = torch.full_like(right, m - 1)
        left = torch.where(first_mask, first_position, left)
        right = torch.where(first_mask, first_position, right)
        left = torch.where(last_mask, last_position, left)
        right = torch.where(last_mask, last_position, right)

        batch_indices = torch.arange(b, device=x.device).unsqueeze(-1)
        x_left = x[batch_indices, left]
        x_right = x[batch_indices, right]
        left_time = selected_indices[batch_indices, left]
        right_time = selected_indices[batch_indices, right]
        denominator = (right_time - left_time).clamp_min(1).to(dtype=x.dtype)
        weight = ((target_indices - left_time).to(dtype=x.dtype) / denominator).view(
            b, self.recover_num, 1, 1
        )
        return x_left + weight * (x_right - x_left)

    def forward(self, x):
        b, f, n, c = x.shape


        x = self.STE_forward(x)
        x = self.TTE_foward(x)


        x, selected_indices = self.ST_foward(x)


        if self.enable_learned_restoration:
            x = rearrange(x, 'b f n c -> (b n) f c')
            x_token = repeat(self.x_token, '() f c -> b f c', b=b * n)
            x = x_token + self.cross_attention(x_token, x, x)
            x = rearrange(x, '(b n) f c -> b f n c', n=n)
        else:
            x = self.interpolate_selected_features(x, selected_indices)


        x = self.head(x)
        x = x.view(b, f, n, -1)
        return x
