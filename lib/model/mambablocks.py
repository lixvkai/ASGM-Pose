import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from .drop import DropPath


DropPath.__repr__ = lambda self: f"DropPath({self.drop_prob})"


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True


try:
    from .csm_triton import CrossScanTriton, CrossMergeTriton, CrossScanTriton1b1, getCSM
    from .csm_triton import CrossScanTritonF, CrossMergeTritonF, CrossScanTriton1b1F
    from .csms6s import CrossScan, CrossMerge, CrossScan_fs_ft, CrossScan_fs_bt, CrossScan_bs_ft, CrossScan_bs_bt, CrossMerge_bs_bt, CrossMerge_bs_ft, CrossMerge_fs_bt, CrossMerge_fs_ft, CrossScan_plus_poselimbs, CrossMerge_plus_poselimbs
    from .csms6s import CrossScan_Ab_1direction, CrossMerge_Ab_1direction, CrossScan_Ab_2direction, CrossMerge_Ab_2direction
    from .csms6s import SelectiveScanMamba, SelectiveScanCore, SelectiveScanOflex
    from .csms6s import flops_selective_scan_fn, flops_selective_scan_ref, selective_scan_flop_jit
except:
    from csm_triton import CrossScanTriton, CrossMergeTriton, CrossScanTriton1b1, getCSM
    from csm_triton import CrossScanTritonF, CrossMergeTritonF, CrossScanTriton1b1F
    from csms6s import CrossScan, CrossMerge
    from csms6s import CrossScan_Ab_1direction, CrossMerge_Ab_1direction, CrossScan_Ab_2direction, CrossMerge_Ab_2direction
    from csms6s import SelectiveScanMamba, SelectiveScanCore, SelectiveScanOflex
    from csms6s import flops_selective_scan_fn, flops_selective_scan_ref, selective_scan_flop_jit



class Linear2d(nn.Linear):
    def forward(self, x: torch.Tensor):


        return F.conv2d(x, self.weight[:, :, None, None], self.bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):

        state_dict[prefix + "weight"] = state_dict[prefix + "weight"].view(self.weight.shape)
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):

        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

        x = x.permute(0, 3, 1, 2)
        return x


class PatchMerging2D(nn.Module):
    def __init__(self, dim, out_dim=-1, norm_layer=nn.LayerNorm, channel_first=False):
        """
        初始化 PatchMerging2D，用于空间下采样。

        参数:
            dim (int): 输入通道数。
            out_dim (int): 输出通道数（默认: 如果为 -1，则为 2 * dim）。
            norm_layer (nn.Module): 归一化层（默认: nn.LayerNorm）。
            channel_first (bool): 如果为 True，输入为 (B, C, H, W)；否则为 (B, H, W, C)。
        """
        super().__init__()
        self.dim = dim
        Linear = Linear2d if channel_first else nn.Linear
        self._patch_merging_pad = self._patch_merging_pad_channel_first if channel_first else self._patch_merging_pad_channel_last

        self.reduction = Linear(4 * dim, (2 * dim) if out_dim < 0 else out_dim, bias=False)
        self.norm = norm_layer(4 * dim)

    @staticmethod
    def _patch_merging_pad_channel_last(x: torch.Tensor):

        H, W, _ = x.shape[-3:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2, :]
        x1 = x[..., 1::2, 0::2, :]
        x2 = x[..., 0::2, 1::2, :]
        x3 = x[..., 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        return x

    @staticmethod
    def _patch_merging_pad_channel_first(x: torch.Tensor):

        H, W = x.shape[-2:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2]
        x1 = x[..., 1::2, 0::2]
        x2 = x[..., 0::2, 1::2]
        x3 = x[..., 1::2, 1::2]
        x = torch.cat([x0, x1, x2, x3], 1)
        return x

    def forward(self, x):

        x = self._patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., channels_first=False):
        """
        初始化两层 MLP。

        参数:
            in_features (int): 输入维度。
            hidden_features (int, optional): 隐藏层维度（默认: in_features）。
            out_features (int, optional): 输出维度（默认: in_features）。
            act_layer (nn.Module): 激活函数（默认: nn.GELU）。
            drop (float): Dropout 率。
            channels_first (bool): 如果为 True，输入为 (B, C, H, W)；否则为 (B, H, W, C)。
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Mlp2(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class gMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., channels_first=False):
        """
        初始化门控 MLP，通过拆分激活增强表达能力。

        参数:
            in_features (int): 输入维度。
            hidden_features (int, optional): 隐藏层维度（默认: in_features）。
            out_features (int, optional): 输出维度（默认: in_features）。
            act_layer (nn.Module): 激活函数（默认: nn.GELU）。
            drop (float): Dropout 率。
            channels_first (bool): 如果为 True，输入为 (B, C, H, W)；否则为 (B, H, W, C)。
        """
        super().__init__()
        self.channel_first = channels_first
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, 2 * hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)

        x, z = x.chunk(2, dim=(1 if self.channel_first else -1))

        x = self.fc2(x * self.act(z))
        x = self.drop(x)
        return x


class SoftmaxSpatial(nn.Softmax):
    def forward(self, x: torch.Tensor):
        if self.dim == -1:
            B, C, H, W = x.shape

            return super().forward(x.view(B, C, -1)).view(B, C, H, W)
        elif self.dim == 1:
            B, H, W, C = x.shape

            return super().forward(x.view(B, -1, C)).view(B, H, W, C)
        else:
            raise NotImplementedError



class mamba_init:
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        """
        初始化 Mamba 的时间步长 (dt) 投影层。

        参数:
            dt_rank (int): 时间步长投影的秩。
            d_inner (int): 内部维度。
            dt_scale (float): 初始化标准差的缩放因子。
            dt_init (str): 初始化方法（"random" 或 "constant"）。
            dt_min (float): 时间步长最小值。
            dt_max (float): 时间步长最大值。
            dt_init_floor (float): 时间步长下限。
        返回:
            nn.Linear: 初始化后的时间步长投影层。
        """
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        """
        初始化 Mamba 的状态矩阵 A（对数形式）。

        参数:
            d_state (int): 状态维度。
            d_inner (int): 内部维度。
            copies (int): 复制份数（用于多组扫描）。
            device: 设备（默认: None）。
            merge (bool): 是否合并复制的矩阵。
        返回:
            nn.Parameter: 初始化后的 A_log 参数。
        """
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        """
        初始化 Mamba 的跳跃参数 D。

        参数:
            d_inner (int): 内部维度。
            copies (int): 复制份数。
            device: 设备（默认: None）。
            merge (bool): 是否合并复制的参数。
        返回:
            nn.Parameter: 初始化后的 D 参数。
        """
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D


class BiSTSSM_v2:
    def __initv2__(
        self,

        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,

        d_conv=3,
        conv_bias=True,

        dropout=0.0,
        bias=False,

        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        initialize="v0",

        forward_type="v2",
        channel_first=False,
        **kwargs,
    ):
        """
        初始化 BiSTSSM_v2，定义 Mamba 模型的核心组件。

        参数:
            d_model (int): 输入/输出模型维度。
            d_state (int): 状态空间维度。
            ssm_ratio (float): 内部维度与模型维度的比例。
            dt_rank (str/int): 时间步长投影的秩（"auto" 或具体值）。
            act_layer (nn.Module): 激活函数（默认: nn.SiLU）。
            d_conv (int): 深度卷积核大小（< 2 表示无卷积）。
            conv_bias (bool): 卷积是否包含偏置。
            dropout (float): Dropout 率。
            bias (bool): 线性层是否包含偏置。
            dt_min/max (float): 时间步长范围。
            dt_init (str): 时间步长初始化方法。
            dt_scale (float): 时间步长缩放因子。
            dt_init_floor (float): 时间步长下限。
            initialize (str): 初始化方法（"v0", "v1", "v2"）。
            forward_type (str): 前向传播类型（例如 "v2", "v2_plus_poselimbs"）。
            channel_first (bool): 输入是否为 (B, C, H, W)。
        """
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.channel_first = channel_first
        self.with_dconv = d_conv > 1
        Linear = Linear2d if channel_first else nn.Linear
        self.forward = self.forwardv2


        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        self.disable_force32, forward_type = checkpostfix("_no32", forward_type)
        self.oact, forward_type = checkpostfix("_oact", forward_type)
        self.disable_z, forward_type = checkpostfix("_noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("_nozact", forward_type)
        out_norm_none, forward_type = checkpostfix("_onnone", forward_type)
        out_norm_dwconv3, forward_type = checkpostfix("_ondwconv3", forward_type)
        out_norm_softmax, forward_type = checkpostfix("_onsoftmax", forward_type)
        out_norm_sigmoid, forward_type = checkpostfix("_onsigmoid", forward_type)


        if out_norm_none:
            self.out_norm = nn.Identity()
        elif out_norm_dwconv3:
            self.out_norm = nn.Sequential(
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_softmax:
            self.out_norm = SoftmaxSpatial(dim=(-1 if channel_first else 1))
        elif out_norm_sigmoid:
            self.out_norm = nn.Sigmoid()
        else:
            LayerNorm = LayerNorm2d if channel_first else nn.LayerNorm
            self.out_norm = LayerNorm(d_inner)


        FORWARD_TYPES = dict(
            v01=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanMamba),
            v02=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanMamba, CrossScan=CrossScanTriton, CrossMerge=CrossMergeTriton),
            v03=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanOflex, CrossScan=CrossScanTriton, CrossMerge=CrossMergeTriton),
            v04=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, CrossScan=CrossScanTriton, CrossMerge=CrossMergeTriton),
            v05=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, no_einsum=True, CrossScan=CrossScanTriton, CrossMerge=CrossMergeTriton),
            v051d=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, no_einsum=True, CrossScan=getCSM(1)[0], CrossMerge=getCSM(1)[1]),
            v052d=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, no_einsum=True, CrossScan=getCSM(2)[0], CrossMerge=getCSM(2)[1]),
            v052dc=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, no_einsum=True, cascade2d=True),
            v2=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanCore),
            v2_fs_ft=partial(self.forward_corev2, force_fp32=(not self.disable_force32), CrossScan=CrossScan_fs_ft, SelectiveScan=SelectiveScanCore, CrossMerge=CrossMerge_fs_ft),
            v2_fs_bt=partial(self.forward_corev2, force_fp32=(not self.disable_force32), CrossScan=CrossScan_fs_bt, SelectiveScan=SelectiveScanCore, CrossMerge=CrossMerge_fs_bt),
            v2_bs_ft=partial(self.forward_corev2, force_fp32=(not self.disable_force32), CrossScan=CrossScan_bs_ft, SelectiveScan=SelectiveScanCore, CrossMerge=CrossMerge_bs_ft),
            v2_bs_bt=partial(self.forward_corev2, force_fp32=(not self.disable_force32), CrossScan=CrossScan_bs_bt, SelectiveScan=SelectiveScanCore, CrossMerge=CrossMerge_bs_bt),
            v2_plus_poselimbs=partial(self.forward_corev2, force_fp32=(not self.disable_force32), CrossScan=CrossScan_plus_poselimbs, SelectiveScan=SelectiveScanCore, CrossMerge=CrossMerge_plus_poselimbs),
            v3=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex),
            v31d=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, CrossScan=CrossScan_Ab_1direction, CrossMerge=CrossMerge_Ab_1direction),
            v32d=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, CrossScan=CrossScan_Ab_2direction, CrossMerge=CrossMerge_Ab_2direction),
            v32dc=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, cascade2d=True),
        )
        self.forward_core = FORWARD_TYPES.get(forward_type, None)
        k_group = 4


        d_proj = d_inner if self.disable_z else (d_inner * 2)
        self.in_proj = Linear(d_model, d_proj, bias=bias)
        self.act = act_layer()














        if self.with_dconv:

            self.conv_dw = nn.Conv2d(
                in_channels=d_inner,
                out_channels=d_inner,
                groups=d_inner,
                bias=False,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

            self.conv_pw = nn.Conv2d(
                in_channels=d_inner,
                out_channels=d_inner,
                kernel_size=1,
                bias=conv_bias,
                **factory_kwargs,
            )

            nn.init.kaiming_normal_(self.conv_dw.weight, mode='fan_out', nonlinearity='relu')
            nn.init.kaiming_normal_(self.conv_pw.weight, mode='fan_out', nonlinearity='relu')
            if conv_bias:
                nn.init.zeros_(self.conv_pw.bias)




        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj


        self.out_act = nn.GELU() if self.oact else nn.Identity()
        self.out_proj = Linear(d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()


        if initialize in ["v0"]:
            self.dt_projs = [
                self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
                for _ in range(k_group)
            ]
            self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
            self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
            del self.dt_projs
            self.A_logs = self.A_log_init(d_state, d_inner, copies=k_group, merge=True)
            self.Ds = self.D_init(d_inner, copies=k_group, merge=True)
        elif initialize in ["v1"]:
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
            self.A_logs = nn.Parameter(torch.randn((k_group * d_inner, d_state)))
            self.dt_projs_weight = nn.Parameter(torch.randn((k_group, d_inner, dt_rank)))
            self.dt_projs_bias = nn.Parameter(torch.randn((k_group, d_inner)))
        elif initialize in ["v2"]:
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
            self.A_logs = nn.Parameter(torch.zeros((k_group * d_inner, d_state)))
            self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((k_group, d_inner, dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((k_group, d_inner)))

    def forward_corev2(
        self,
        x: torch.Tensor=None,
        to_dtype=True,
        force_fp32=False,
        ssoflex=True,
        SelectiveScan=SelectiveScanOflex,
        CrossScan=CrossScan,
        CrossMerge=CrossMerge,
        no_einsum=False,
        cascade2d=False,
        **kwargs,
    ):
        """
        Mamba 模型的核心前向传播逻辑。

        参数:
            x (torch.Tensor): 输入张量，形状为 (B, D, H, W)。
            to_dtype (bool): 是否将输出转换为输入的 dtype。
            force_fp32 (bool): 是否强制使用 fp32 精度。
            ssoflex (bool): 是否在 SelectiveScanOflex 中使用 fp32。
            SelectiveScan: 选择性扫描模块。
            CrossScan: 跨扫描模块。
            CrossMerge: 跨合并模块。
            no_einsum (bool): 是否用线性或 1D 卷积替代 einsum。
            cascade2d (bool): 是否使用级联 2D 时域扫描。
        返回:
            torch.Tensor: 输出张量，形状为 (B, D, H, W) 或 (B, H, W, D)。
        """
        x_proj_weight = self.x_proj_weight
        x_proj_bias = getattr(self, "x_proj_bias", None)
        dt_projs_weight = self.dt_projs_weight
        dt_projs_bias = self.dt_projs_bias
        A_logs = self.A_logs
        Ds = self.Ds
        delta_softplus = True
        out_norm = getattr(self, "out_norm", None)
        channel_first = self.channel_first
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        B, D, H, W = x.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = H * W

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, -1, -1, ssoflex)

        if cascade2d:
            def scan_rowcol(
                x: torch.Tensor,
                proj_weight: torch.Tensor,
                proj_bias: torch.Tensor,
                dt_weight: torch.Tensor,
                dt_bias: torch.Tensor,
                _As: torch.Tensor,
                _Ds: torch.Tensor,
                width=True,
            ):
                XB, XD, XH, XW = x.shape
                if width:
                    _B, _D, _L = XB * XH, XD, XW
                    xs = x.permute(0, 2, 1, 3).contiguous()
                else:
                    _B, _D, _L = XB * XW, XD, XH
                    xs = x.permute(0, 3, 1, 2).contiguous()

                xs = xs.unsqueeze(2)
                if no_einsum:
                    x_dbl = F.conv1d(xs.view(_B, -1, _L), proj_weight.view(-1, _D, 1), bias=(proj_bias.view(-1) if proj_bias is not None else None), groups=1)
                    dts, Bs, Cs = torch.split(x_dbl.view(_B, 1, -1, _L), [R, N, N], dim=2)
                    dts = F.conv1d(dts.contiguous().view(_B, -1, _L), dt_weight.view(_D, -1, 1), groups=1)
                else:
                    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, proj_weight[:1])
                    if x_proj_bias is not None:
                        x_dbl = x_dbl + x_proj_bias.view(1, 1, -1, 1)
                    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
                    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_weight[:1])

                xs = xs.view(_B, -1, _L)
                dts = dts.contiguous().view(_B, -1, _L)
                As = _As.view(-1, N).to(torch.float)
                Bs = Bs.contiguous().view(_B, 1, N, _L)
                Cs = Cs.contiguous().view(_B, 1, N, _L)
                Ds = _Ds.view(-1)
                delta_bias = dt_bias.view(-1).to(torch.float)

                if force_fp32:
                    xs = xs.to(torch.float)
                dts = dts.to(xs.dtype)
                Bs = Bs.to(xs.dtype)
                Cs = Cs.to(xs.dtype)

                ys = selective_scan(
                    xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
                ).view(_B, 1, -1, _L)
                return ys

            As = -torch.exp(A_logs.to(torch.float)).view(4, -1, N)
            y_row = scan_rowcol(
                x,
                proj_weight=x_proj_weight.view(4, -1, D)[:1].contiguous(),
                proj_bias=(x_proj_bias.view(4, -1)[:1].contiguous() if x_proj_bias is not None else None),
                dt_weight=dt_projs_weight.view(4, D, -1)[:1].contiguous(),
                dt_bias=(dt_projs_bias.view(4, -1)[:1].contiguous() if dt_projs_bias is not None else None),
                _As=As[:1].contiguous().view(-1, N),
                _Ds=Ds.view(4, -1)[:1].contiguous().view(-1),
                width=True,
            ).view(B, H, 1, -1, W).squeeze(2).permute(0, 2, 1, 3)
            y_col = scan_rowcol(
                y_row,
                proj_weight=x_proj_weight.view(4, -1, D)[2:3].contiguous().to(y_row.dtype),
                proj_bias=(x_proj_bias.view(4, -1)[2:3].contiguous().to(y_row.dtype) if x_proj_bias is not None else None),
                dt_weight=dt_projs_weight.view(4, D, -1)[2:3].contiguous().to(y_row.dtype),
                dt_bias=(dt_projs_bias.view(4, -1)[2:3].contiguous().to(y_row.dtype) if dt_projs_bias is not None else None),
                _As=As[2:3].contiguous().view(-1, N),
                _Ds=Ds.view(4, -1)[2:3].contiguous().view(-1),
                width=False,
            ).view(B, W, 1, -1, H).squeeze(2).permute(0, 2, 3, 1)
            y = y_col
        else:
            xs = CrossScan.apply(x)
            if no_einsum:
                x_dbl = F.conv1d(xs.view(B, -1, L), x_proj_weight.view(-1, D, 1), bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None), groups=K)
                dts, Bs, Cs = torch.split(x_dbl.view(B, K, -1, L), [R, N, N], dim=2)
                dts = F.conv1d(dts.contiguous().view(B, -1, L), dt_projs_weight.view(K * D, -1, 1), groups=K)
            else:
                x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
                if x_proj_bias is not None:
                    x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
                dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
                dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)


            xs = xs.view(B, -1, L)
            dts = dts.contiguous().view(B, -1, L)
            As = -torch.exp(A_logs.to(torch.float))
            Bs = Bs.contiguous().view(B, K, N, L)
            Cs = Cs.contiguous().view(B, K, N, L)
            Ds = Ds.to(torch.float)
            delta_bias = dt_projs_bias.view(-1).to(torch.float)

            if force_fp32:
                xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

            ys = selective_scan(
                xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
            ).view(B, K, -1, H, W)

            y = CrossMerge.apply(ys)

            if getattr(self, "__DEBUG__", False):
                setattr(self, "__data__", dict(
                    A_logs=A_logs, Bs=Bs, Cs=Cs, Ds=Ds,
                    us=xs, dts=dts, delta_bias=delta_bias,
                    ys=ys, y=y,
                ))

        y = y.view(B, -1, H, W)
        if not channel_first:
            y = y.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = out_norm(y)

        return (y.to(x.dtype) if to_dtype else y)

    def forwardv2(self, x: torch.Tensor, **kwargs):
        """
        BiSTSSM 的前向传播。

        参数:
            x (torch.Tensor): 输入张量，形状为 (B, D, H, W) 或 (B, H, W, D)。
        返回:
            torch.Tensor: 输出张量，形状与输入相同。
        """
        x = self.in_proj(x)
        if not self.disable_z:
            x, z = x.chunk(2, dim=(1 if self.channel_first else -1))
            if not self.disable_z_act:
                z = self.act(z)
        if not self.channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()


        if self.with_dconv:
            x = self.conv_dw(x)
            x = self.conv_pw(x)
        x = self.act(x)
        y = self.forward_core(x)
        y = self.out_act(y)
        if not self.disable_z:
            y = y * z
        out = self.dropout(self.out_proj(y))
        return out


class BiSTSSM(nn.Module, mamba_init, BiSTSSM_v2):
    def __init__(
        self,
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        d_conv=3,
        conv_bias=True,
        dropout=0.0,
        bias=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        initialize="v0",
        forward_type="v2",
        channel_first=False,
        **kwargs,
    ):
        """
        初始化 BiSTSSM 模型。

        参数:
            与 BiSTSSM_v2 相同，继承其参数配置。
        """
        super().__init__()
        kwargs.update(
            d_model=d_model, d_state=d_state, ssm_ratio=ssm_ratio, dt_rank=dt_rank,
            act_layer=act_layer, d_conv=d_conv, conv_bias=conv_bias, dropout=dropout, bias=bias,
            dt_min=dt_min, dt_max=dt_max, dt_init=dt_init, dt_scale=dt_scale, dt_init_floor=dt_init_floor,
            initialize=initialize, forward_type=forward_type, channel_first=channel_first,
        )
        self.__initv2__(**kwargs)


class BiSTSSMBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: nn.Module = nn.LayerNorm,
        channel_first=False,

        ssm_d_state: int = 16,
        ssm_ratio=2.0,
        ssm_dt_rank: Any = "auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias=True,
        ssm_drop_rate: float = 0,
        ssm_init="v0",
        forward_type="v2",

        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate: float = 0.0,
        gmlp=False,

        use_checkpoint: bool = False,
        post_norm: bool = False,
        **kwargs,
    ):
        """
        初始化 BiSTSSMBlock，Mamba 模型的基本构建块。

        参数:
            hidden_dim (int): 隐藏维度（输入/输出维度）。
            drop_path (float): DropPath 率。
            norm_layer (nn.Module): 归一化层（默认: nn.LayerNorm）。
            channel_first (bool): 输入是否为 (B, C, H, W)。
            ssm_d_state (int): SSM 状态维度。
            ssm_ratio (float): SSM 内部维度比例。
            ssm_dt_rank (str/int): SSM 时间步长秩。
            ssm_act_layer (nn.Module): SSM 激活函数。
            ssm_conv (int): SSM 卷积核大小。
            ssm_conv_bias (bool): SSM 卷积是否包含偏置。
            ssm_drop_rate (float): SSM Dropout 率。
            ssm_init (str): SSM 初始化方法。
            forward_type (str): SSM 前向传播类型。
            mlp_ratio (float): MLP 隐藏维度比例。
            mlp_act_layer (nn.Module): MLP 激活函数。
            mlp_drop_rate (float): MLP Dropout 率。
            gmlp (bool): 是否使用门控 MLP。
            use_checkpoint (bool): 是否使用检查点以节省内存。
            post_norm (bool): 是否在操作后应用归一化。
        """
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = BiSTSSM(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                dropout=ssm_drop_rate,
                initialize=ssm_init,
                forward_type=forward_type,
                channel_first=channel_first,
            )

        self.drop_path = DropPath(drop_path)

        if self.mlp_branch:
            _MLP = Mlp if not gmlp else gMlp
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = _MLP(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer, drop=mlp_drop_rate, channels_first=channel_first)

    def _forward(self, input: torch.Tensor):
        """
        内部前向传播逻辑。

        参数:
            input (torch.Tensor): 输入张量，形状为 (B, H, W, C) 或 (B, C, H, W)。
        返回:
            torch.Tensor: 输出张量，形状与输入相同。
        """
        x = input
        if self.ssm_branch:
            x = x + self.drop_path(self.op(self.norm(x)))
        if self.mlp_branch:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward(self, input: torch.Tensor):
        """
        前向传播，支持检查点以节省内存。

        参数:
            input (torch.Tensor): 输入张量。
        返回:
            torch.Tensor: 输出张量。
        """
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)