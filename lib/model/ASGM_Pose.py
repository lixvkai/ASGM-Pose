import torch
import torch.nn as nn
from einops import rearrange, repeat

from lib.model.mambablocks import BiSTSSMBlock


def cluster_dpc_knn(x, cluster_num, k):
    with torch.no_grad():
        _, _, channels = x.shape
        distances = torch.cdist(x, x) / (channels ** 0.5)
        nearest_distances, _ = torch.topk(distances, k=k, dim=-1, largest=False)
        density = (-(nearest_distances ** 2).mean(dim=-1)).exp()
        density = density + torch.rand_like(density) * 1e-6

        higher_density = density[:, None, :] > density[:, :, None]
        max_distance = distances.flatten(1).max(dim=-1)[0][:, None, None]
        distance_to_parent = (
            distances * higher_density.to(x.dtype)
            + max_distance * (~higher_density).to(x.dtype)
        ).min(dim=-1)[0]
        score = distance_to_parent * density
        return torch.topk(score, k=cluster_num, dim=-1).indices


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
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super().__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(
            torch.empty(embed_dim, cheb_k, dim_in, dim_out)
        )
        self.bias_pool = nn.Parameter(torch.empty(embed_dim, dim_out))
        nn.init.xavier_uniform_(self.weights_pool)
        nn.init.zeros_(self.bias_pool)

    @staticmethod
    def normalize_adjacency(adjacency):
        adjacency = adjacency + torch.eye(
            adjacency.shape[0], device=adjacency.device
        )
        degree = adjacency.sum(dim=1).clamp(min=1e-8)
        degree_inv_sqrt = torch.diag(degree.pow(-0.5))
        return degree_inv_sqrt @ adjacency @ degree_inv_sqrt

    def forward(self, x, node_embeddings):
        if not torch.isfinite(node_embeddings).all():
            node_embeddings = torch.nan_to_num(
                node_embeddings, nan=0.0, posinf=1.0, neginf=-1.0
            )

        adjacency = node_embeddings @ node_embeddings.transpose(0, 1)
        supports = self.normalize_adjacency(adjacency)
        if not torch.isfinite(supports).all():
            supports = torch.nan_to_num(
                supports, nan=0.0, posinf=1.0, neginf=0.0
            )
            supports = self.normalize_adjacency(supports)

        support_set = [torch.eye(supports.shape[0], device=supports.device), supports]
        for _ in range(2, self.cheb_k):
            next_support = 2 * supports @ support_set[-1] - support_set[-2]
            if not torch.isfinite(next_support).all():
                next_support = torch.nan_to_num(
                    next_support, nan=0.0, posinf=1.0, neginf=-1.0
                )
            support_set.append(next_support)
        supports = torch.stack(support_set, dim=0)

        weights = torch.einsum(
            'nd,dkio->nkio', node_embeddings, self.weights_pool
        )
        bias = node_embeddings @ self.bias_pool
        if not torch.isfinite(weights).all():
            weights = torch.nan_to_num(
                weights, nan=0.0, posinf=1.0, neginf=-1.0
            )
        if not torch.isfinite(bias).all():
            bias = torch.nan_to_num(
                bias, nan=0.0, posinf=1.0, neginf=-1.0
            )

        graph_features = torch.einsum('knm,bmc->bknc', supports, x)
        graph_features = graph_features.permute(0, 2, 1, 3)
        output = torch.einsum('bnki,nkio->bno', graph_features, weights) + bias
        if not torch.isfinite(output).all():
            output = torch.nan_to_num(
                output, nan=0.0, posinf=1.0, neginf=-1.0
            )
        return output


class ASGM_Pose(nn.Module):
    def __init__(
        self,
        num_frame=243,
        num_joints=17,
        in_chans=2,
        embed_dim_ratio=128,
        depth=10,
        mlp_ratio=2.0,
        drop_rate=0.0,
        drop_path_rate=0.2,
        token_num=81,
        layer_index=5,
    ):
        super().__init__()

        embed_dim = embed_dim_ratio
        out_dim = 3
        self.token_num = token_num
        self.layer_index = layer_index
        self.recover_num = num_frame
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.pos_embed_token = nn.Parameter(
            torch.zeros(1, token_num, embed_dim)
        )
        self.x_token = nn.Parameter(torch.zeros(1, num_frame, embed_dim))
        self.cross_attention = Cross_Attention(
            embed_dim, num_heads=8, qkv_bias=True, proj_drop=drop_rate
        )

        self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.block_depth = depth

        self.node_embeddings = nn.Parameter(torch.randn(num_joints, embed_dim_ratio), requires_grad=True)
        self.spatial_gcn = AVWGCN(
            embed_dim_ratio, embed_dim_ratio, cheb_k=3, embed_dim=embed_dim_ratio
        )

        self.TTEblocks = nn.ModuleList([
            BiSTSSMBlock(
                hidden_dim=embed_dim,
                mlp_ratio=mlp_ratio,
                drop_path=dpr[i],
                norm_layer=nn.LayerNorm,
                forward_type='v2_plus_poselimbs',
            )
            for i in range(depth)
        ])

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
        batch_size, frame_count, _, _ = x.shape
        if self.token_num > frame_count:
            raise ValueError(
                f'token_num ({self.token_num}) exceeds current length ({frame_count}).'
            )
        descriptors = rearrange(x, 'b f n c -> b (f c) n')
        descriptors = self.pool(descriptors)
        descriptors = rearrange(
            descriptors, 'b (f c) 1 -> b f c', f=frame_count
        )
        indices = cluster_dpc_knn(descriptors, self.token_num, k=2)
        return torch.sort(indices, dim=-1).values

    def ST_foward(self, x):
        if x.ndim != 4:
            raise ValueError(f'Expected a 4D tensor, got shape {tuple(x.shape)}.')
        batch_size, _, num_joints, _ = x.shape

        for i in range(1, self.block_depth):
            if i == self.layer_index:
                indices = self.select_frame_indices(x)
                batch_indices = torch.arange(
                    batch_size, device=x.device
                ).unsqueeze(-1)
                x = x[batch_indices, indices]
                x = rearrange(x, 'b f n c -> (b n) f c')
                x = x + self.pos_embed_token
                x = rearrange(
                    x, '(b n) f c -> b f n c', n=num_joints
                )

            x = self.TTEblocks[i](x)
            x = self.Temporal_norm(x)
        return x

    def forward(self, x):
        batch_size, frame_count, num_joints, _ = x.shape
        x = self.STE_forward(x)
        x = self.TTE_foward(x)
        x = self.ST_foward(x)

        x = rearrange(x, 'b f n c -> (b n) f c')
        restoration_tokens = repeat(
            self.x_token, '() f c -> b f c', b=batch_size * num_joints
        )
        x = restoration_tokens + self.cross_attention(
            restoration_tokens, x, x
        )
        x = rearrange(
            x, '(b n) f c -> b f n c', n=num_joints
        )
        x = self.head(x)
        return x.view(batch_size, frame_count, num_joints, -1)
