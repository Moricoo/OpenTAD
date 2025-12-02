# AdaTAD ActivityNet 预训练模型部署指南

## 🎯 概述

本指南将帮助您使用AdaTAD作者在ActivityNet-1.3数据集上训练的**官方预训练权重**，部署视频时序动作检测服务。

---

## 📦 预训练权重下载链接

### 推荐模型（按性能排序）

#### 1. VideoMAEv2-g + InternVideo2（最佳性能）⭐

- **性能**: mAP@0.5=63.59%, ave. mAP=42.90%
- **配置**: `configs/adatad/anet/e2e_anet_videomaev2_g_192x4_224_adapter_internvideo2.py`
- **权重**: [Google Drive](https://drive.google.com/file/d/1DQquCFhNNRcK8dAsOT81dsuM4UGZ6HiJ/view?usp=sharing)
- **输入**: 768 frames, 224x224
- **需要**: InternVideo2分类器

#### 2. VideoMAEv2-g + InternVideo（次优）

- **性能**: mAP@0.5=61.74%, ave. mAP=41.85%
- **配置**: `configs/adatad/anet/e2e_anet_videomaev2_g_192x4_224_adapter_internvideo.py`
- **权重**: 查看log文件中的checkpoint信息
- **输入**: 768 frames, 224x224

#### 3. VideoMAE-L（无需外部分类器，推荐用于部署）⭐⭐

- **性能**: mAP@0.5=59.00%, ave. mAP=39.15%
- **配置**: `configs/adatad/anet/e2e_anet_videomae_l_192x4_224_adapter_cls.py`
- **权重**: [Google Drive](https://drive.google.com/file/d/1VYAvDrc7O7W4hDmUjjE6y32WmVNQ4ZR_/view?usp=sharing)
- **输入**: 768 frames, 224x224
- **优势**: **直接训练200类分类头，无需外部分类器，最适合部署**

#### 4. VideoMAE-S（轻量级，适合资源受限）

- **性能**: mAP@0.5=56.23%, ave. mAP=37.81%
- **配置**: `configs/adatad/anet/e2e_anet_videomae_s_192x4_160_adapter.py`
- **权重**: [Google Drive](https://drive.google.com/file/d/1gncN-xjArNtgVoBKCwCJCH4ISA3yVqIU/view?usp=sharing)
- **输入**: 768 frames, 160x160
- **需要**: CUHK分类器

#### 5. 其他模型

| Backbone | mAP@0.5 | Config | Download |
|----------|---------|--------|----------|
| VideoMAE-B | 56.72% | `e2e_anet_videomae_b_192x4_160_adapter.py` | [Link](https://drive.google.com/file/d/1tePHMitdwUrWax1nYlbucaqI5LbvZhZo/view?usp=sharing) |
| VideoMAE-L | 57.73% | `e2e_anet_videomae_l_192x4_160_adapter.py` | [Link](https://drive.google.com/file/d/1GxwNLc1rRp6x5ug1zd1r_1DmYCZD_tw5/view?usp=sharing) |
| VideoMAE-H | 57.77% | `e2e_anet_videomae_h_192x4_160_adapter.py` | [Link](https://drive.google.com/file/d/1Hqpdq7Qclf0-1oF25tWwZLI8Ranp-uBv/view?usp=sharing) |
| VideoMAEv2-g | 58.42% | `e2e_anet_videomaev2_g_192x4_160_adapter.py` | [Link](https://drive.google.com/file/d/1lfWyWrt1gJOm7YfwCdXi7HiNomHPGvna/view?usp=sharing) |

---

## 🚀 快速部署步骤

### 步骤1: 下载预训练权重

#### 推荐：VideoMAE-L（无需外部分类器）

```bash
# 创建权重目录
mkdir -p /root/OpenTAD/pretrained/adatad

# 下载权重（使用gdown或wget）
# 方法1: 使用gdown（需要安装: pip install gdown）
gdown https://drive.google.com/uc?id=1VYAvDrc7O7W4hDmUjjE6y32WmVNQ4ZR_ \
    -O /root/OpenTAD/pretrained/adatad/adatad_anet_videomae_l_224_cls.pth

# 方法2: 手动下载
# 1. 在浏览器中打开: https://drive.google.com/file/d/1VYAvDrc7O7W4hDmUjjE6y32WmVNQ4ZR_/view?usp=sharing
# 2. 下载后上传到服务器
# scp adatad_anet_videomae_l_224_cls.pth root@<server>:/root/OpenTAD/pretrained/adatad/
```

#### 或选择其他模型

```bash
# VideoMAE-S (轻量级)
gdown https://drive.google.com/uc?id=1gncN-xjArNtgVoBKCwCJCH4ISA3yVqIU \
    -O /root/OpenTAD/pretrained/adatad/adatad_anet_videomae_s_160.pth

# VideoMAE-L (标准版，需要分类器)
gdown https://drive.google.com/uc?id=1GxwNLc1rRp6x5ug1zd1r_1DmYCZD_tw5 \
    -O /root/OpenTAD/pretrained/adatad/adatad_anet_videomae_l_160.pth
```

### 步骤2: 准备配置文件

配置文件已存在：
- **推荐（无需分类器）**: `configs/adatad/anet/e2e_anet_videomae_l_192x4_224_adapter_cls.py`
- **标准版**: `configs/adatad/anet/e2e_anet_videomae_s_192x4_160_adapter.py`

### 步骤3: 准备数据（如果需要）

如果使用标准版（需要CUHK分类器）：
```bash
# 下载分类器文件
mkdir -p data/activitynet-1.3/classifiers
# 分类器文件路径在配置中指定: data/activitynet-1.3/classifiers/cuhk_val_simp_7.json
```

### 步骤4: 运行推理

#### 使用OpenTAD测试脚本

```bash
cd /root/OpenTAD

# 单GPU推理（推荐VideoMAE-L，无需分类器）
CUDA_VISIBLE_DEVICES=0 torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    tools/test.py \
    configs/adatad/anet/e2e_anet_videomae_l_192x4_224_adapter_cls.py \
    --checkpoint /root/OpenTAD/pretrained/adatad/adatad_anet_videomae_l_224_cls.pth
```

---

## 📝 创建推理服务脚本

创建 `video_analysis_service.py`:

```python
#!/usr/bin/env python3
"""
AdaTAD ActivityNet 视频分析服务
使用官方预训练权重进行推理
"""

import os
import sys
sys.path.insert(0, "/root/OpenTAD")

import torch
import json
import argparse
from pathlib import Path
from mmengine.config import Config
from opentad.models import build_detector
from opentad.datasets import build_dataset, build_dataloader
from opentad.cores import eval_one_epoch

def main():
    parser = argparse.ArgumentParser(description="AdaTAD Video Analysis Service")

    # 推荐配置（无需分类器）
    parser.add_argument("--config", type=str,
                       default="configs/adatad/anet/e2e_anet_videomae_l_192x4_224_adapter_cls.py",
                       help="Config file path")
    parser.add_argument("--checkpoint", type=str,
                       default="/root/OpenTAD/pretrained/adatad/adatad_anet_videomae_l_224_cls.pth",
                       help="Checkpoint path")
    parser.add_argument("--video-dir", type=str, required=True,
                       help="Directory containing videos to analyze")
    parser.add_argument("--output", type=str, default="results.json",
                       help="Output JSON path")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device (cuda:0 or cpu)")

    args = parser.parse_args()

    print(f"Loading config from: {args.config}")
    cfg = Config.fromfile(args.config)

    # 修改数据路径（如果需要）
    if hasattr(cfg.dataset, 'test'):
        cfg.dataset.test.data_path = args.video_dir

    print(f"Building model...")
    model = build_detector(cfg.model)

    print(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        print(f"✅ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint, strict=False)
        print("✅ Loaded checkpoint")

    model.eval()
    model.to(args.device)
    print("✅ Model ready for inference!")

    # 构建数据集和数据加载器
    print("Building dataset...")
    test_dataset = build_dataset(cfg.dataset.test)
    test_loader = build_dataloader(
        test_dataset,
        rank=0,
        world_size=1,
        shuffle=False,
        drop_last=False,
        **cfg.solver.test,
    )

    # 推理
    print("Starting inference...")
    results = []
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            # 将数据移到GPU
            if isinstance(data, dict):
                for key in data:
                    if isinstance(data[key], torch.Tensor):
                        data[key] = data[key].to(args.device)

            # 推理
            output = model(**data, return_loss=False,
                          infer_cfg=cfg.inference,
                          post_cfg=cfg.post_processing)

            results.append(output)
            print(f"Processed batch {batch_idx + 1}")

    # 保存结果
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"✅ Inference completed! Results saved to {args.output}")

if __name__ == "__main__":
    main()
```

---

## 🔧 使用gdown下载Google Drive文件

### 安装gdown

```bash
pip install gdown
```

### 下载权重

```bash
# VideoMAE-L (推荐，无需分类器)
gdown https://drive.google.com/uc?id=1VYAvDrc7O7W4hDmUjjE6y32WmVNQ4ZR_ \
    -O /root/OpenTAD/pretrained/adatad/adatad_anet_videomae_l_224_cls.pth

# VideoMAE-S (轻量级)
gdown https://drive.google.com/uc?id=1gncN-xjArNtgVoBKCwCJCH4ISA3yVqIU \
    -O /root/OpenTAD/pretrained/adatad/adatad_anet_videomae_s_160.pth
```

---

## 📊 模型对比

| 模型 | mAP@0.5 | ave. mAP | 输入尺寸 | 需要分类器 | 推荐度 |
|------|---------|----------|----------|-----------|--------|
| **VideoMAE-L (cls)** | 59.00% | 39.15% | 224x224 | ❌ 不需要 | ⭐⭐⭐⭐⭐ |
| VideoMAE-S | 56.23% | 37.81% | 160x160 | ✅ CUHK | ⭐⭐⭐ |
| VideoMAE-L | 57.73% | 39.21% | 160x160 | ✅ CUHK | ⭐⭐⭐⭐ |
| VideoMAEv2-g | 58.42% | 39.77% | 160x160 | ✅ CUHK | ⭐⭐⭐⭐ |
| VideoMAEv2-g+InternVideo2 | 63.59% | 42.90% | 224x224 | ✅ InternVideo2 | ⭐⭐⭐⭐⭐ |

---

## 🎯 推荐部署方案

### 方案1: VideoMAE-L (cls) - 最简单（推荐）⭐

**优点**:
- ✅ 无需外部分类器
- ✅ 直接输出200类动作
- ✅ 部署最简单
- ✅ 性能良好（mAP@0.5=59.00%）

**配置**:
- Config: `configs/adatad/anet/e2e_anet_videomae_l_192x4_224_adapter_cls.py`
- Checkpoint: `adatad_anet_videomae_l_224_cls.pth`
- 下载: https://drive.google.com/file/d/1VYAvDrc7O7W4hDmUjjE6y32WmVNQ4ZR_/view?usp=sharing

### 方案2: VideoMAE-S - 轻量级

**优点**:
- ✅ 模型较小，推理速度快
- ✅ 显存占用少

**缺点**:
- ❌ 需要CUHK分类器
- ⚠️ 性能略低

---

## 📋 完整部署示例

```bash
# 1. 下载权重
mkdir -p /root/OpenTAD/pretrained/adatad
cd /root/OpenTAD/pretrained/adatad

# 使用gdown下载（推荐）
pip install gdown
gdown https://drive.google.com/uc?id=1VYAvDrc7O7W4hDmUjjE6y32WmVNQ4ZR_ \
    -O adatad_anet_videomae_l_224_cls.pth

# 2. 准备视频目录
mkdir -p /data/videos/input
# 将视频放入该目录

# 3. 运行推理
cd /root/OpenTAD
CUDA_VISIBLE_DEVICES=0 torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    tools/test.py \
    configs/adatad/anet/e2e_anet_videomae_l_192x4_224_adapter_cls.py \
    --checkpoint /root/OpenTAD/pretrained/adatad/adatad_anet_videomae_l_224_cls.pth \
    --cfg-options dataset.test.data_path=/data/videos/input
```

---

## ⚠️ 注意事项

1. **分类器文件**:
   - VideoMAE-L (cls)版本**不需要**分类器
   - 其他版本需要CUHK分类器: `data/activitynet-1.3/classifiers/cuhk_val_simp_7.json`

2. **视频格式**: 支持MP4等常见格式

3. **GPU内存**:
   - VideoMAE-S: 建议至少4GB
   - VideoMAE-L: 建议至少8GB
   - VideoMAEv2-g: 建议至少16GB

4. **输入要求**:
   - 视频会自动resize到配置的尺寸
   - 时序长度: 768 frames

---

## 📚 参考资源

- **OpenTAD GitHub**: https://github.com/sming256/OpenTAD
- **AdaTAD README**: `configs/adatad/README.md`
- **配置文件**: `configs/adatad/anet/e2e_anet_videomae_l_192x4_224_adapter_cls.py`
- **ActivityNet官网**: http://activity-net.org/

---

## 🎯 下一步

1. **下载预训练权重**（推荐VideoMAE-L cls版本）
2. **准备测试视频**
3. **运行推理测试**
4. **根据需要创建API服务**

**推荐使用VideoMAE-L (cls)版本，因为它无需外部分类器，部署最简单！**
