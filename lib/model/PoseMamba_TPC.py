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

# 获取当前工作目录
current_directory = os.path.dirname(__file__) + '/../' + '../'
sys.path.append(current_directory)
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
import torch.nn.functional as F
from functools import partial
import torch.fft

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math
import numpy as np

from lib.model.mambablocks import BiSTSSMBlock

# === GCN implementation from MotionAGFormer ===
CONNECTIONS = {10: [9], 9: [8, 10], 8: [7, 9], 14: [15, 8], 15: [16, 14], 11: [12, 8], 12: [13, 11],
               7: [0, 8], 0: [1, 7], 1: [2, 0], 2: [3, 1], 4: [5, 0], 5: [6, 4], 16: [15], 13: [12], 3: [2], 6: [5]}


class GCN(nn.Module):
    def __init__(self, dim_in, dim_out, num_nodes, neighbour_num=4, mode='spatial', use_temporal_similarity=True,
                 temporal_connection_len=1, connections=None):
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


# === TPC Token Pruning Functions ===
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


def cluster_dpc_knn_center(x, cluster_num, k, center, token_mask=None):
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

        ## remove center
        score[:, center] = -math.inf

        _, index_down = torch.topk(score, k=cluster_num, dim=-1)

        dist_matrix = index_points(dist_matrix, index_down)
        idx_cluster = dist_matrix.argmin(dim=1)

        idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)
        idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)
        idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)

    return index_down, idx_cluster


class PoseMamba_TPC(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=256, depth=6, mlp_ratio=2.,
                 drop_rate=0., drop_path_rate=0.2, norm_layer=None, token_num=None, layer_index=None):
        """TPC PoseMamba: Token Pruning and Clustering for Efficient PoseMamba
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
        """
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio
        out_dim = 3

        # TPC parameters
        self.token_num = token_num if token_num is not None else num_frame // 4
        self.layer_index = layer_index if layer_index is not None else depth // 2
        self.center = (num_frame - 1) // 2

        # Token pruning components
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Original PoseMamba components
        self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.block_depth = depth

        self.STEblocks = nn.ModuleList([
            GCN(
                dim_in=embed_dim_ratio,
                dim_out=embed_dim_ratio,
                num_nodes=num_joints,
                neighbour_num=4,
                mode='spatial'
            )
            for _ in range(depth)
        ])

        self.TTEblocks = nn.ModuleList([
            BiSTSSMBlock(
                hidden_dim=embed_dim,
                mlp_ratio=mlp_ratio,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                forward_type='v2_plus_poselimbs'
            )
            for i in range(depth)])

        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm = norm_layer(embed_dim)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_dim),
        )

    def STE_forward(self, x):
        b, f, n, c = x.shape
        x = rearrange(x, 'b f n c  -> (b f) n c', )
        x = self.Spatial_patch_to_embedding(x)
        x += self.Spatial_pos_embed
        x = self.pos_drop(x)
        x = rearrange(x, '(b f) n c  -> b f n c', f=f)
        blk = self.STEblocks[0]
        x = blk(x)
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

    def ST_foward(self, x, index=None):
        assert len(x.shape) == 4, "shape is equal to 4"
        b, f, n, cw = x.shape

        for i in range(1, self.block_depth):
            # === TPC Token Pruning ===
            if i == self.layer_index:
                x_knn = rearrange(x, 'b f n c -> b (f c) n')
                x_knn = self.pool(x_knn)
                x_knn = rearrange(x_knn, 'b (f c) 1 -> b f c', f=f)

                if index == None:
                    index, idx_cluster = cluster_dpc_knn_center(x_knn, self.token_num - 1, 2, self.center)
                    index_center = self.center * torch.ones(b, 1, device=x.device, dtype=index.dtype)
                    index = torch.cat([index, index_center], dim=-1)
                    index, _ = torch.sort(index)

                batch_ind = torch.arange(b, device=x.device).unsqueeze(-1)
                x = x[batch_ind, index]

            # Original STE-TTE processing
            steblock = self.STEblocks[i]
            tteblock = self.TTEblocks[i]
            x = steblock(x)
            x = self.Spatial_norm(x)
            x = tteblock(x)
            x = self.Temporal_norm(x)

        return x, index

    def forward(self, x, index=None):
        b, f, n, c = x.shape

        # Initial STE and TTE
        x = self.STE_forward(x)
        x = self.TTE_foward(x)

        # Main processing with TPC
        x, index = self.ST_foward(x, index)

        # Output head
        x = self.head(x)
        x = x.view(b, -1, n, -1)
        return x, index


if __name__ == "__main__":
    torch.cuda.set_device(3)
    model = PoseMamba_TPC(num_frame=243, embed_dim_ratio=128, mlp_ratio=2, depth=10,
                          token_num=61, layer_index=3).cuda()
    from thop import profile, clever_format

    input_shape = (1, 243, 17, 2)
    x = torch.randn(input_shape).cuda()
    flops, params = profile(model, inputs=(x,))
    flops, params = clever_format([flops, params], "%.3f")
    print("FLOPs: %s" % (flops))
    print("params: %s" % (params))