# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

import math
import torch
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import Tensor, nn
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks import DropPath
from mmcv.cnn.bricks.transformer import FFN, PatchEmbed
from mmengine.registry import MODELS
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import constant_init, trunc_normal_init
from mmaction.utils import ConfigType, OptConfigType
from mmaction.models.backbones.vit_mae import get_sinusoid_encoding


class TIALoRAAdapter(BaseModule):
    """
    TIA Adapter with LoRA correction
    在原始TIA adapter的基础上，在down_proj和up_proj上添加LoRA修正项

    Down(x) = x @ W_down + scale * (x @ A_down @ B_down)
    Up(h)   = h @ W_up    + scale * (h @ A_up   @ B_up)
    """
    def __init__(
        self,
        embed_dims: int,
        mlp_ratio: float = 0.25,
        kernel_size: int = 3,
        dilation: int = 1,
        temporal_size: int = 384,
        # LoRA参数
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        lora_scale: float = 1.0,  # LoRA修正项的缩放因子
    ) -> None:
        super().__init__()

        hidden_dims = int(embed_dims * mlp_ratio)

        # temporal depth-wise convolution (保持原始TIA结构)
        self.temporal_size = temporal_size
        self.dwconv = nn.Conv1d(
            hidden_dims,
            hidden_dims,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size // 2) * dilation,
            dilation=dilation,
            groups=hidden_dims,
        )
        self.conv = nn.Conv1d(hidden_dims, hidden_dims, 1)
        self.dwconv.weight.data.normal_(mean=0.0, std=math.sqrt(2.0 / kernel_size))
        self.dwconv.bias.data.zero_()
        self.conv.weight.data.normal_(mean=0.0, std=math.sqrt(2.0 / hidden_dims))
        self.conv.bias.data.zero_()

        # 原始adapter projection (保持原始结构)
        self.down_proj = nn.Linear(embed_dims, hidden_dims)
        self.act = nn.GELU()
        self.up_proj = nn.Linear(hidden_dims, embed_dims)
        self.gamma = nn.Parameter(torch.ones(1))
        trunc_normal_init(self.down_proj, std=0.02, bias=0)
        constant_init(self.up_proj, 0)  # the last projection layer is initialized to 0

        # LoRA层：添加修正项
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_scale = lora_scale

        # Down projection LoRA
        self.down_lora_a = nn.Linear(embed_dims, lora_r, bias=False)
        self.down_lora_b = nn.Linear(lora_r, hidden_dims, bias=False)
        self.down_lora_dropout = nn.Dropout(lora_dropout)

        # Up projection LoRA
        self.up_lora_a = nn.Linear(hidden_dims, lora_r, bias=False)
        self.up_lora_b = nn.Linear(lora_r, embed_dims, bias=False)
        self.up_lora_dropout = nn.Dropout(lora_dropout)

        # 初始化LoRA权重
        nn.init.kaiming_uniform_(self.down_lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.down_lora_b.weight)
        nn.init.kaiming_uniform_(self.up_lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_lora_b.weight)

    def forward(self, x: Tensor, h: int, w: int) -> Tensor:
        inputs = x

        # Down projection: x @ W_down + scale * (x @ A_down @ B_down)
        x_down_original = self.down_proj(x)  # 原始路径
        x_down_lora = self.down_lora_b(
            self.down_lora_dropout(self.down_lora_a(x))
        ) * (self.lora_alpha / self.lora_r)  # LoRA路径
        x = x_down_original + self.lora_scale * x_down_lora  # 组合
        x = self.act(x)

        # temporal depth-wise convolution (保持原始TIA结构)
        B, N, C = x.shape
        # 动态计算temporal_size：从N推断
        # N = batch_size * temporal_size * h * w
        # 所以每个batch的序列长度 = N // B
        # temporal_size = (N // B) // (h * w)
        spatial_size = h * w
        seq_len_per_batch = N // B if B > 0 else N

        # 直接计算temporal_size，强制使用计算值
        # 不进行复杂的验证，因为验证逻辑可能导致使用错误的默认值
        if seq_len_per_batch >= spatial_size and spatial_size > 0:
            # 直接使用计算值
            temporal_size = seq_len_per_batch // spatial_size
        else:
            # 如果seq_len_per_batch < spatial_size，这种情况不正常
            # 但仍然尝试计算，而不是使用默认值
            temporal_size = max(1, seq_len_per_batch // spatial_size) if spatial_size > 0 else self.temporal_size

        # 使用-1自动推断batch维度，保持与原始实现一致
        attn = x.reshape(-1, temporal_size, h, w, C)
        attn = attn.permute(0, 2, 3, 4, 1).flatten(0, 2)  # [b*h*w, C, temporal_size]
        attn = self.dwconv(attn)
        attn = self.conv(attn)
        attn = attn.unflatten(0, (-1, h, w)).permute(0, 4, 1, 2, 3)  # [b, temporal_size, h, w, C]
        attn = attn.reshape(B, N, C)
        x = x + attn

        # Up projection: h @ W_up + scale * (h @ A_up @ B_up)
        x_up_original = self.up_proj(x)  # 原始路径
        x_up_lora = self.up_lora_b(
            self.up_lora_dropout(self.up_lora_a(x))
        ) * (self.lora_alpha / self.lora_r)  # LoRA路径
        x = x_up_original + self.lora_scale * x_up_lora  # 组合

        return x * self.gamma + inputs


class TIALoRABlock(BaseModule):
    """Transformer block with TIA LoRA Adapter."""

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        act_cfg: Dict = dict(type="GELU"),
        norm_cfg: Dict = dict(type="LN", eps=1e-6),
        init_values: float = 0.0,
        with_cp: bool = False,
        # Adapter参数
        adapter_index: int = -1,
        adapter_kwargs: Optional[Dict] = None,
    ) -> None:
        super().__init__()
        self.with_cp = with_cp
        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.attn = Attention(
            embed_dims,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            drop_rate=drop_rate,
        )

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]
        mlp_hidden_dim = int(embed_dims * mlp_ratio)
        self.mlp = FFN(
            embed_dims=embed_dims,
            feedforward_channels=mlp_hidden_dim,
            act_cfg=act_cfg,
            ffn_drop=drop_rate,
            add_identity=False,
        )

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((embed_dims)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((embed_dims)), requires_grad=True)
        else:
            self.gamma_1 = None
            self.gamma_2 = None

        # Adapter
        self.adapter_index = adapter_index
        if adapter_kwargs is not None:
            self.adapter = TIALoRAAdapter(embed_dims=embed_dims, **adapter_kwargs)
        else:
            self.adapter = None

    def forward(self, x: Tensor, h: int, w: int) -> Tensor:
        def _inner_forward(x):
            if self.gamma_1 is None:
                x = x + self.drop_path(self.attn(self.norm1(x)))
            else:
                x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))

            if self.adapter is not None:
                x = x + self.adapter(x, h, w)

            if self.gamma_2 is None:
                x = x + self.drop_path(self.mlp(self.norm2(x)))
            else:
                x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

        return x


# 导入Attention类
from .vit_adapter import Attention


@MODELS.register_module()
class VisionTransformerTIALoRA(BaseModule):
    """Vision Transformer with TIA LoRA Adapter."""

    def __init__(
        self,
        img_size: Union[int, tuple] = 224,
        patch_size: Union[int, tuple] = 16,
        in_channels: int = 3,
        embed_dims: int = 768,
        num_frames: int = 16,
        tubelet_size: int = 2,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_cfg: Dict = dict(type="LN", eps=1e-6),
        init_cfg: Optional[Union[Dict, List[Dict]]] = None,
        with_cp: bool = False,
        return_feat_map: bool = False,
        # Adapter参数
        adapter_index: List[int] = [],
        adapter_kwargs: Optional[Dict] = None,
        total_frames: int = 384,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.num_layers = num_layers
        self.embed_dims = embed_dims
        self.adapter_index = adapter_index
        self.return_feat_map = return_feat_map

        # Patch embedding
        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            conv_type="Conv3d",
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
            padding=(0, 0, 0),
            dilation=(1, 1, 1),
        )
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size

        # Position embedding - 按照原始实现计算
        grid_size = img_size // patch_size
        num_patches = grid_size**2 * (num_frames // tubelet_size)
        self.grid_size = (grid_size, grid_size)

        # 使用sine-cosine positional embeddings
        pos_embed = get_sinusoid_encoding(num_patches, embed_dims)
        self.register_buffer("pos_embed", pos_embed)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        self.blocks = ModuleList()
        for i in range(num_layers):
            adapter_kwargs_layer = adapter_kwargs.copy() if adapter_kwargs is not None else None
            if adapter_kwargs_layer is not None:
                adapter_kwargs_layer["temporal_size"] = total_frames

            self.blocks.append(
                TIALoRABlock(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=dpr[i],
                    norm_cfg=norm_cfg,
                    with_cp=with_cp,
                    adapter_index=i if i in adapter_index else -1,
                    adapter_kwargs=adapter_kwargs_layer if i in adapter_index else None,
                )
            )

        self.norm = build_norm_layer(norm_cfg, embed_dims)[1]

        # pos_embed使用sine-cosine encoding，不需要初始化

    def _freeze_layers(self):
        """Prevent all the parameters not in the adapters and LoRA layers"""
        # freeze patch_embed
        self.patch_embed.eval()
        for m in self.patch_embed.modules():
            for param in m.parameters():
                param.requires_grad = False

        # freeze blocks except the adapter's and LoRA's parameters
        for block in self.blocks:
            for m, n in block.named_children():
                if "adapter" not in m and "lora" not in m and m != "drop_path":
                    n.eval()
                    for param in n.parameters():
                        param.requires_grad = False
                elif "adapter" in m:
                    # 冻结原始Linear层（down_proj和up_proj），只训练LoRA层
                    for name, param in n.named_parameters():
                        # 冻结原始down_proj和up_proj（不包含lora的）
                        if ("down_proj" in name or "up_proj" in name) and "lora" not in name:
                            param.requires_grad = False
                        # LoRA层保持可训练
                        elif "lora" in name:
                            param.requires_grad = True
                        # gamma和temporal conv也保持可训练
                        elif "gamma" in name or "dwconv" in name or "conv" in name:
                            param.requires_grad = True
                        # 其他参数（如act）保持可训练
                        else:
                            param.requires_grad = True

    def forward(self, x: Tensor, masks: Optional[Tensor] = None) -> Tensor:
        """Defines the computation performed at every call."""
        b, _, _, h, w = x.shape
        h //= self.patch_size
        w //= self.patch_size
        x = self.patch_embed(x)[0]

        # 动态调整pos_embed的大小以匹配实际的h, w
        if (h, w) != self.grid_size:
            pos_embed = self.pos_embed.reshape(-1, *self.grid_size, self.embed_dims)
            pos_embed = pos_embed.permute(0, 3, 1, 2)
            pos_embed = F.interpolate(pos_embed, size=(h, w), mode="bicubic", align_corners=False)
            pos_embed = pos_embed.permute(0, 2, 3, 1).flatten(1, 2)
            pos_embed = pos_embed.reshape(1, -1, self.embed_dims)
        else:
            pos_embed = self.pos_embed

        x = x + pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x, h, w)

        x = self.norm(x)

        if self.return_feat_map:
            x = x.reshape(b, -1, h, w, self.embed_dims)
            x = x.permute(0, 4, 1, 2, 3)
            return x

        return x

