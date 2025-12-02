# QLoRA 微调使用指南

## 概述

QLoRA (Quantized LoRA) 是一种参数高效的微调方法，结合了：
- **量化 (Quantization)**: 4-bit/8-bit权重量化，减少显存占用
- **LoRA (Low-Rank Adaptation)**: 低秩适应，只训练少量参数
- **显存优化**: 相比原始Adapter，显存占用减少约50-75%

## 安装依赖

```bash
pip install bitsandbytes peft
```

## 配置文件

QLoRA配置文件位于：
```
configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter_qlora.py
```

### 关键配置参数

```python
backbone=dict(
    type="VisionTransformerQLoRA",  # 使用QLoRA版本
    # ... 其他配置 ...
    # QLoRA配置
    lora_r=16,              # LoRA rank (推荐: 8-32)
    lora_alpha=32,          # LoRA alpha (通常设为 lora_r * 2)
    lora_dropout=0.1,       # LoRA dropout
    quantize_bits=4,        # 4-bit or 8-bit quantization
    use_plain_adapter=False, # False: 完整版(含temporal conv), True: 简化版
)
```

### 优化器配置

QLoRA需要为LoRA层设置学习率：

```python
optimizer = dict(
    type="AdamW",
    lr=1e-4,
    paramwise=True,
    backbone=dict(
        lr=0,  # backbone冻结
        custom=[
            dict(name="adapter", lr=2e-4, weight_decay=0.05),
            dict(name="lora", lr=2e-4, weight_decay=0.05),  # LoRA层学习率
        ],
    ),
)
```

## 训练命令

### 从头开始训练

```bash
cd /root/OpenTAD
source /root/miniconda3/bin/activate opentad
CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --nproc_per_node=1 \
    --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
    tools/train.py configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter_qlora.py
```

### 从检查点恢复

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --nproc_per_node=1 \
    --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
    tools/train.py configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter_qlora.py \
    --resume exps/thumos/adatad/e2e_actionformer_videomae_s_768x1_160_adapter_qlora/gpu1_id0/checkpoint/epoch_X.pth
```

## 参数调优建议

### LoRA Rank (lora_r)
- **较小值 (8-16)**: 参数更少，显存占用更小，但可能表达能力不足
- **较大值 (32-64)**: 参数更多，表达能力更强，但显存占用增加
- **推荐**: 从16开始，根据效果调整

### LoRA Alpha (lora_alpha)
- **推荐**: `lora_alpha = lora_r * 2` (例如 lora_r=16, lora_alpha=32)
- **作用**: 控制LoRA层的缩放因子

### Quantization Bits
- **4-bit**: 显存占用最小，但可能有精度损失
- **8-bit**: 显存占用稍大，但精度更高
- **推荐**: 先尝试4-bit，如果效果不好再试8-bit

### Adapter类型
- **use_plain_adapter=False**: 完整版，包含temporal convolution，表达能力更强
- **use_plain_adapter=True**: 简化版，无temporal convolution，参数更少

## 显存对比

| 配置 | 显存占用 | 参数量 |
|------|----------|--------|
| 原始Adapter | ~12GB | 100% |
| QLoRA (4-bit) | ~6-8GB | <1% |
| QLoRA (8-bit) | ~8-10GB | <1% |

## 注意事项

1. **bitsandbytes要求**: 需要CUDA支持，确保安装了正确版本的bitsandbytes
2. **量化精度**: 4-bit量化可能有轻微精度损失，但通常可以接受
3. **训练稳定性**: QLoRA训练通常更稳定，因为只训练少量参数
4. **收敛速度**: 可能比全量微调稍慢，但显存优势明显

## 故障排查

### 问题1: ImportError: bitsandbytes not found
```bash
pip install bitsandbytes
# 如果失败，可能需要安装CUDA版本的bitsandbytes
```

### 问题2: CUDA out of memory
- 减小batch_size
- 使用8-bit量化（quantize_bits=8）
- 减小lora_r

### 问题3: 训练效果不佳
- 增大lora_r (例如 16 -> 32)
- 调整lora_alpha
- 尝试use_plain_adapter=False（使用完整版Adapter）

## 性能对比

与原始Adapter baseline对比：
- **显存占用**: 减少50-75%
- **训练速度**: 可能稍慢（量化开销）
- **模型性能**: 通常可以达到接近原始Adapter的性能（mAP差异<1%）

## 参考

- QLoRA论文: [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- LoRA论文: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

