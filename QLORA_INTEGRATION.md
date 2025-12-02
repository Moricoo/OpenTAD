# QLoRA 微调集成方案

## 概述

在AdaTAD baseline基础上集成QLoRA (Quantized LoRA) 进行参数高效微调。

## QLoRA 原理

QLoRA结合了：
1. **量化 (Quantization)**: 将模型权重量化为4-bit或8-bit，减少显存占用
2. **LoRA (Low-Rank Adaptation)**: 在量化模型上添加低秩适应层，只训练少量参数
3. **参数高效**: 相比全量微调，只训练<1%的参数

## 实现方案

### 1. QLoRA Adapter

已创建 `vit_adapter_qlora.py`，包含：
- `QLoRAAdapter`: 完整版，包含temporal convolution
- `PlainQLoRAAdapter`: 简化版，无temporal convolution

### 2. 关键特性

- **量化Linear层**: 使用 `bitsandbytes.nn.Linear4bit` 或 `Linear8bitLt`
- **LoRA层**: 在量化层上添加低秩适应
- **显存优化**: 相比原始Adapter，显存占用减少约50-75%

### 3. 配置参数

```python
lora_r: int = 16          # LoRA rank
lora_alpha: int = 32      # LoRA alpha (缩放因子)
lora_dropout: float = 0.1 # LoRA dropout
quantize_bits: int = 4    # 4-bit or 8-bit quantization
```

## 下一步

1. 创建支持QLoRA的VisionTransformerAdapter类
2. 创建QLoRA配置文件
3. 更新模型注册
4. 创建训练脚本

## 使用方法

待实现完成后，可以通过配置文件选择使用QLoRA：

```python
backbone=dict(
    type="VisionTransformerQLoRA",
    adapter_type="qlora",  # 使用QLoRA
    lora_r=16,
    lora_alpha=32,
    quantize_bits=4,
    ...
)
```

