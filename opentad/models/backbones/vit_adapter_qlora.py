# Copyright (c) OpenMMLab. All rights reserved.
"""
QLoRA (Quantized LoRA) Adapter for AdaTAD
结合量化(4-bit/8-bit)和LoRA的参数高效微调方法
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

try:
    import bitsandbytes as bnb
    QLORA_AVAILABLE = True
except ImportError:
    QLORA_AVAILABLE = False
    print("Warning: bitsandbytes not installed. QLoRA features will be disabled.")


class QLoRAAdapter(BaseModule):
    """
    QLoRA Adapter: 结合量化LoRA的Adapter
    在原有Adapter基础上，使用量化权重和LoRA进行参数高效微调
    """
    def __init__(
        self,
        embed_dims: int,
        mlp_ratio: float = 0.25,
        kernel_size: int = 3,
        dilation: int = 1,
        temporal_size: int = 384,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        quantize_bits: int = 4,  # 4-bit or 8-bit quantization
    ) -> None:
        super().__init__()

        if not QLORA_AVAILABLE:
            raise ImportError("bitsandbytes is required for QLoRA. Install with: pip install bitsandbytes")

        hidden_dims = int(embed_dims * mlp_ratio)

        # 量化配置
        self.quantize_bits = quantize_bits
        self.quantize_fn = bnb.nn.Linear4bit if quantize_bits == 4 else bnb.nn.Linear8bitLt

        # LoRA配置
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout

        # temporal depth-wise convolution (保持原样，不量化)
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

        # adapter projection with quantization and LoRA
        # 使用量化Linear + LoRA
        self.down_proj = self.quantize_fn(
            embed_dims,
            hidden_dims,
            bias=False,
        )
        self.act = nn.GELU()
        self.up_proj = self.quantize_fn(
            hidden_dims,
            embed_dims,
            bias=False,
        )
        self.gamma = nn.Parameter(torch.ones(1))

        # LoRA层
        self.down_lora_a = nn.Linear(embed_dims, lora_r, bias=False)
        self.down_lora_b = nn.Linear(lora_r, hidden_dims, bias=False)
        self.up_lora_a = nn.Linear(hidden_dims, lora_r, bias=False)
        self.up_lora_b = nn.Linear(lora_r, embed_dims, bias=False)

        # 初始化LoRA权重
        nn.init.kaiming_uniform_(self.down_lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.down_lora_b.weight)
        nn.init.kaiming_uniform_(self.up_lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_lora_b.weight)

    def forward(self, x: Tensor, h: int, w: int) -> Tensor:
        inputs = x

        # down projection with quantization + LoRA
        x_quantized = self.down_proj(x)
        x_lora = self.down_lora_b(self.down_lora_a(x)) * (self.lora_alpha / self.lora_r)
        x = x_quantized + x_lora
        x = self.act(x)

        # temporal depth-wise convolution
        B, N, C = x.shape
        attn = x.reshape(-1, self.temporal_size, h, w, x.shape[-1])
        attn = attn.permute(0, 2, 3, 4, 1).flatten(0, 2)
        attn = self.dwconv(attn)
        attn = self.conv(attn)
        attn = attn.unflatten(0, (-1, h, w)).permute(0, 4, 1, 2, 3)
        attn = attn.reshape(B, N, C)
        x = x + attn

        # up projection with quantization + LoRA
        x_quantized = self.up_proj(x)
        x_lora = self.up_lora_b(self.up_lora_a(x)) * (self.lora_alpha / self.lora_r)
        x = x_quantized + x_lora

        return x * self.gamma + inputs


class PlainQLoRAAdapter(BaseModule):
    """
    Plain QLoRA Adapter: 简化版，无temporal convolution
    """
    def __init__(
        self,
        embed_dims: int,
        mlp_ratio: float = 0.25,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        quantize_bits: int = 4,
        **kwargs,
    ) -> None:
        super().__init__()

        if not QLORA_AVAILABLE:
            raise ImportError("bitsandbytes is required for QLoRA. Install with: pip install bitsandbytes")

        hidden_dims = int(embed_dims * mlp_ratio)
        self.quantize_bits = quantize_bits
        self.quantize_fn = bnb.nn.Linear4bit if quantize_bits == 4 else bnb.nn.Linear8bitLt

        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout

        # adapter projection with quantization + LoRA
        self.down_proj = self.quantize_fn(
            embed_dims,
            hidden_dims,
            bias=False,
        )
        self.act = nn.GELU()
        self.up_proj = self.quantize_fn(
            hidden_dims,
            embed_dims,
            bias=False,
        )
        self.gamma = nn.Parameter(torch.ones(1))

        # LoRA层
        self.down_lora_a = nn.Linear(embed_dims, lora_r, bias=False)
        self.down_lora_b = nn.Linear(lora_r, hidden_dims, bias=False)
        self.up_lora_a = nn.Linear(hidden_dims, lora_r, bias=False)
        self.up_lora_b = nn.Linear(lora_r, embed_dims, bias=False)

        nn.init.kaiming_uniform_(self.down_lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.down_lora_b.weight)
        nn.init.kaiming_uniform_(self.up_lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_lora_b.weight)

    def forward(self, x: Tensor, h: int, w: int) -> Tensor:
        inputs = x

        # down and up projection with quantization + LoRA
        x_quantized = self.down_proj(x)
        x_lora = self.down_lora_b(self.down_lora_a(x)) * (self.lora_alpha / self.lora_r)
        x = x_quantized + x_lora
        x = self.act(x)

        x_quantized = self.up_proj(x)
        x_lora = self.up_lora_b(self.up_lora_a(x)) * (self.lora_alpha / self.lora_r)
        x = x_quantized + x_lora

        return x * self.gamma + inputs

# 导入原始Adapter中的Attention类
from .vit_adapter import Attention


class QLoRABlock(BaseModule):
    """The basic block in the Vision Transformer with QLoRA Adapter."""

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
        adapter_mlp_ratio: float = 0.25,
        temporal_size: int = 384,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        quantize_bits: int = 4,
        use_plain_adapter: bool = False,
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
            if use_plain_adapter:
                self.adapter = PlainQLoRAAdapter(
                    embed_dims=embed_dims,
                    mlp_ratio=adapter_mlp_ratio,
                    lora_r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    quantize_bits=quantize_bits,
                )
            else:
                self.adapter = QLoRAAdapter(
                    embed_dims=embed_dims,
                    kernel_size=3,
                    dilation=1,
                    temporal_size=temporal_size,
                    mlp_ratio=adapter_mlp_ratio,
                    lora_r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    quantize_bits=quantize_bits,
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
class VisionTransformerQLoRA(BaseModule):
    """Vision Transformer with QLoRA Adapter support.

    This is a QLoRA version of VisionTransformerAdapter that uses quantized
    weights and LoRA for parameter-efficient fine-tuning.
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
        adapter_mlp_ratio: float = 0.25,
        total_frames: int = 768,
        adapter_index: list = [3, 5, 7, 11],
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        quantize_bits: int = 4,
        use_plain_adapter: bool = False,
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
                QLoRABlock(
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
                    adapter_mlp_ratio=adapter_mlp_ratio,
                    temporal_size=total_frames // tubelet_size,
                    lora_r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    quantize_bits=quantize_bits,
                    use_plain_adapter=use_plain_adapter,
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
        print("QLoRA - ViT's param: {}, QLoRA Adapter's params: {}, ratio: {:2.1f}%".format(num_vit_param, num_adapter_param, ratio))

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
                    # 注意：量化层的参数不能设置requires_grad，只设置LoRA层
                    for name, param in n.named_parameters():
                        # 跳过量化层的参数（down_proj和up_proj是量化Linear）
                        if "down_proj" in name or "up_proj" in name:
                            # 量化Linear的参数不能设置requires_grad
                            continue
                        # 只设置LoRA层和gamma等非量化参数
                        if param.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                            param.requires_grad = True
