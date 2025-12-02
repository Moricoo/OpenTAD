# Copyright (c) OpenMMLab. All rights reserved.
"""
LoRA替换Adapter的实现
直接用LoRA替换AdaTAD的Adapter，而不是在Adapter内部加LoRA
"""
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

# 导入原始Adapter中的Attention类
from .vit_adapter import Attention


class LoRAAdapter(BaseModule):
    """
    用LoRA直接替换Adapter
    使用LoRA的低秩分解替代adapter的down_proj/up_proj结构
    保留adapter的残差连接和缩放机制
    """
    def __init__(
        self,
        embed_dims: int,
        mlp_ratio: float = 0.25,  # 保持与原adapter相同的接口
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__()

        hidden_dims = int(embed_dims * mlp_ratio)
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout

        # LoRA层：模拟adapter的down_proj和up_proj
        # Down projection LoRA
        self.down_lora_a = nn.Linear(embed_dims, lora_r, bias=False)
        self.down_lora_b = nn.Linear(lora_r, hidden_dims, bias=False)

        # Activation
        self.act = nn.GELU()

        # Up projection LoRA
        self.up_lora_a = nn.Linear(hidden_dims, lora_r, bias=False)
        self.up_lora_b = nn.Linear(lora_r, embed_dims, bias=False)

        # Dropout and scaling
        self.lora_dropout = nn.Dropout(lora_dropout)
        self.gamma = nn.Parameter(torch.ones(1))

        # 初始化LoRA权重
        nn.init.kaiming_uniform_(self.down_lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.down_lora_b.weight)
        nn.init.kaiming_uniform_(self.up_lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_lora_b.weight)

    def forward(self, x: Tensor, h: int, w: int) -> Tensor:
        inputs = x

        # Down projection with LoRA
        x_lora_down = self.down_lora_b(
            self.lora_dropout(self.down_lora_a(x))
        ) * (self.lora_alpha / self.lora_r)
        x = x_lora_down
        x = self.act(x)

        # Up projection with LoRA
        x_lora_up = self.up_lora_b(
            self.lora_dropout(self.up_lora_a(x))
        ) * (self.lora_alpha / self.lora_r)
        x = x_lora_up

        # 残差连接和缩放
        return x * self.gamma + inputs


class LoRABlock(BaseModule):
    """The basic block in the Vision Transformer with LoRA Adapter."""

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        mlp_ratio: int = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        act_cfg: ConfigType = dict(type="GELU"),
        norm_cfg: ConfigType = dict(type="LN", eps=1e-6),
        init_cfg: OptConfigType = None,
        with_cp: bool = False,
        use_adapter: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        self.with_cp = with_cp
        self.use_adapter = use_adapter

        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.attn = Attention(
            embed_dims,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            drop_rate=drop_rate,
        )

        self.drop_path = nn.Identity()
        if drop_path_rate > 0.0:
            self.drop_path = DropPath(drop_path_rate)
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]

        mlp_hidden_dim = int(embed_dims * mlp_ratio)
        self.mlp = FFN(
            embed_dims=embed_dims,
            feedforward_channels=mlp_hidden_dim,
            act_cfg=act_cfg,
            ffn_drop=drop_rate,
            add_identity=False,
        )

        if self.use_adapter:
            self.adapter = LoRAAdapter(
                embed_dims=embed_dims,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )

    def forward(self, x: Tensor, h, w) -> Tensor:
        """Defines the computation performed at every call."""

        def _inner_forward(x):
            """Forward wrapper for utilizing checkpoint."""
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))

            if self.use_adapter:
                x = self.adapter(x, h, w)
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


@MODELS.register_module()
class VisionTransformerLoRA(BaseModule):
    """Vision Transformer with LoRA Adapter (直接替换Adapter).

    This version uses LoRA to directly replace the Adapter, rather than
    adding LoRA inside the Adapter.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dims: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: int = 4.0,
        qkv_bias: bool = True,
        qk_scale: int = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_cfg: ConfigType = dict(type="LN", eps=1e-6),
        num_frames: int = 16,
        tubelet_size: int = 2,
        use_mean_pooling: int = True,
        pretrained: Optional[str] = None,
        return_feat_map: bool = False,
        with_cp: bool = False,
        total_frames: int = 768,
        adapter_index: list = [3, 5, 7, 11],
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        init_cfg: Optional[Union[Dict, List[Dict]]] = [
            dict(type="TruncNormal", layer="Linear", std=0.02, bias=0.0),
            dict(type="Constant", layer="LayerNorm", val=1.0, bias=0.0),
        ],
        **kwargs,
    ) -> None:
        if pretrained:
            self.init_cfg = dict(type="Pretrained", checkpoint=pretrained)
        super().__init__(init_cfg=init_cfg)

        self.with_cp = with_cp

        self.embed_dims = embed_dims
        self.patch_size = patch_size

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            conv_type="Conv3d",
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
            padding=(0, 0, 0),
            dilation=(1, 1, 1),
        )

        grid_size = img_size // patch_size
        num_patches = grid_size**2 * (num_frames // tubelet_size)
        self.grid_size = (grid_size, grid_size)

        # sine-cosine positional embeddings
        pos_embed = get_sinusoid_encoding(num_patches, embed_dims)
        self.register_buffer("pos_embed", pos_embed)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = ModuleList(
            [
                LoRABlock(
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
                    init_cfg=init_cfg,
                    use_adapter=i in adapter_index,
                    lora_r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                )
                for i in range(depth)
            ]
        )

        if use_mean_pooling:
            self.norm = nn.Identity()
            self.fc_norm = build_norm_layer(norm_cfg, embed_dims)[1]
        else:
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
            self.fc_norm = None

        self.return_feat_map = return_feat_map

        # count the number of parameters
        num_vit_param = sum(p.numel() for name, p in self.named_parameters() if "adapter" not in name and "lora" not in name)
        num_adapter_param = sum(p.numel() for name, p in self.named_parameters() if "adapter" in name or "lora" in name)
        ratio = num_adapter_param / num_vit_param * 100 if num_vit_param > 0 else 0
        print("LoRA - ViT's param: {}, LoRA Adapter's params: {}, ratio: {:2.1f}%".format(num_vit_param, num_adapter_param, ratio))

    def forward(self, x: Tensor) -> Tensor:
        """Defines the computation performed at every call."""
        self._freeze_layers()

        b, _, _, h, w = x.shape
        h //= self.patch_size
        w //= self.patch_size
        x = self.patch_embed(x)[0]
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

        if self.fc_norm is not None:
            return self.fc_norm(x.mean(1))

        return x[:, 0]

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
                elif "adapter" in m or "lora" in m:
                    # Only adapter and LoRA parameters are trainable
                    for name, param in n.named_parameters():
                        if param.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                            param.requires_grad = True

