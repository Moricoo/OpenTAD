# EPIC-KITCHENS AdaTAD 部署指南 - Vlogger做饭视频分析

## 🎯 应用场景

**目标**: 使用AdaTAD在EPIC-KITCHENS上训练的模型，分析vlogger做饭视频中的动作定位。

**EPIC-KITCHENS特点**:
- **第一人称视角**（egocentric vision）
- **厨房场景**的日常活动
- **细粒度动作**检测
- **Verb（动词）**: 97类动作（如take, put, open, close等）
- **Noun（名词）**: 293类物体（如cup, plate, knife等）

---

## 📦 需要下载的内容

### 1. EPIC-pretrained VideoMAE权重（必需）

在训练AdaTAD之前，需要先下载EPIC预训练的VideoMAE权重：

| 模型 | 用途 | 下载链接 |
|------|------|----------|
| **VideoMAE-L (EPIC-Verb)** | 动词检测 | [Google Drive](https://drive.google.com/file/d/1h7oLiNN5LTXau4HWmmzS_ekvuNdZkp-b/view?usp=sharing) |
| **VideoMAE-L (EPIC-Noun)** | 名词检测 | [Google Drive](https://drive.google.com/file/d/1nRuzJI4ej90vFsKCBSugRVOmxrR8urwW/view?usp=sharing) |

### 2. AdaTAD训练好的模型权重（用于推理）

| 模型 | 类别数 | 性能 (ave. mAP) | 下载链接 |
|------|--------|----------------|----------|
| **Verb模型** | 97类 | 29.69% | [Google Drive](https://drive.google.com/file/d/16Hq3sHu0S97Ge2AewHT6DOaHSo0TqIlx/view?usp=sharing) |
| **Noun模型** | 293类 | 29.44% | [Google Drive](https://drive.google.com/file/d/17k3f6wirqniLTjKOsIXbfqJPA_iLb88E/view?usp=sharing) |

### 3. EPIC-KITCHENS-100数据集

- **原始视频**: 需要从EPIC-KITCHENS官网下载
- **标注文件**: 需要下载annotations
- **数据量**: 约500GB-1TB（原始视频）

---

## 🚀 完整部署步骤

### 步骤1: 下载EPIC-pretrained VideoMAE权重

```bash
# 创建目录
mkdir -p /root/OpenTAD/pretrained
cd /root/OpenTAD/pretrained

# 安装gdown（如果未安装）
pip install gdown

# 下载EPIC-Verb预训练权重
echo "下载EPIC-Verb预训练权重..."
gdown https://drive.google.com/uc?id=1h7oLiNN5LTXau4HWmmzS_ekvuNdZkp-b \
    -O vit-large-p16_videomae-epic_verb.pth

# 下载EPIC-Noun预训练权重
echo "下载EPIC-Noun预训练权重..."
gdown https://drive.google.com/uc?id=1nRuzJI4ej90vFsKCBSugRVOmxrR8urwW \
    -O vit-large-p16_videomae-epic_noun.pth

# 验证下载
ls -lh vit-large-p16_videomae-epic_*.pth
```

### 步骤2: 下载AdaTAD训练好的模型权重（用于推理）

```bash
# 创建目录
mkdir -p /root/OpenTAD/pretrained/adatad
cd /root/OpenTAD/pretrained/adatad

# 下载Verb模型权重
echo "下载AdaTAD EPIC-Verb模型..."
gdown https://drive.google.com/uc?id=16Hq3sHu0S97Ge2AewHT6DOaHSo0TqIlx \
    -O adatad_epic_verb.pth

# 下载Noun模型权重
echo "下载AdaTAD EPIC-Noun模型..."
gdown https://drive.google.com/uc?id=17k3f6wirqniLTjKOsIXbfqJPA_iLb88E \
    -O adatad_epic_noun.pth

# 验证下载
ls -lh adatad_epic_*.pth
```

### 步骤3: 准备EPIC-KITCHENS数据集

#### 3.1 下载数据集

EPIC-KITCHENS-100数据集需要从官网下载：
- **官网**: https://epic-kitchens.github.io/
- **需要注册**: 需要填写数据使用协议
- **下载链接**: 注册后会在邮件中提供

#### 3.2 数据集结构

```
data/epic_kitchens-100/
├── annotations/
│   ├── epic_kitchens_verb.json      # 动词标注
│   ├── epic_kitchens_noun.json      # 名词标注
│   ├── category_idx_verb.txt        # 动词类别映射
│   └── category_idx_noun.txt        # 名词类别映射
└── raw_data/
    └── epic_kitchens_100_30fps_512x288/  # 原始视频（30fps, 512x288）
        ├── P01/
        │   ├── P01_01.mp4
        │   ├── P01_02.mp4
        │   └── ...
        ├── P02/
        └── ...
```

#### 3.3 快速开始（如果只有自己的视频）

如果只是想测试模型，可以使用自己的vlogger做饭视频：

```bash
# 创建测试目录
mkdir -p /data/videos/epic_test

# 放置您的视频文件
# cp your_cooking_video.mp4 /data/videos/epic_test/
```

---

## 🔧 配置文件说明

### Verb模型配置

**文件**: `configs/adatad/epic/e2e_epic_videomae_l_ft_768x8_160_adapter_verb.py`

**关键参数**:
- **Backbone**: VideoMAE-L (1024维, 24层)
- **输入**: 768 frames, 160x160
- **类别数**: 97个动词
- **预训练**: `pretrained/vit-large-p16_videomae-epic_verb.pth`

### Noun模型配置

**文件**: `configs/adatad/epic/e2e_epic_videomae_l_ft_768x8_160_adapter_noun.py`

**关键参数**:
- **Backbone**: VideoMAE-L (1024维, 24层)
- **输入**: 768 frames, 160x160
- **类别数**: 293个名词
- **预训练**: `pretrained/vit-large-p16_videomae-epic_noun.pth`

---

## 📝 推理使用

### 方法1: 使用OpenTAD测试脚本（推荐）

#### Verb检测（动作检测）

```bash
cd /root/OpenTAD

CUDA_VISIBLE_DEVICES=0 torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    tools/test.py \
    configs/adatad/epic/e2e_epic_videomae_l_ft_768x8_160_adapter_verb.py \
    --checkpoint /root/OpenTAD/pretrained/adatad/adatad_epic_verb.pth \
    --cfg-options dataset.test.data_path=/data/videos/epic_test
```

#### Noun检测（物体检测）

```bash
CUDA_VISIBLE_DEVICES=0 torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    tools/test.py \
    configs/adatad/epic/e2e_epic_videomae_l_ft_768x8_160_adapter_noun.py \
    --checkpoint /root/OpenTAD/pretrained/adatad/adatad_epic_noun.pth \
    --cfg-options dataset.test.data_path=/data/videos/epic_test
```

### 方法2: 创建推理服务脚本

创建 `epic_inference_service.py`:

```python
#!/usr/bin/env python3
"""
EPIC-KITCHENS AdaTAD 推理服务
用于vlogger做饭视频的动作和物体检测
"""

import os
import sys
sys.path.insert(0, "/root/OpenTAD")

import torch
import json
import argparse
from mmengine.config import Config
from opentad.models import build_detector

def load_model(config_path, checkpoint_path, device="cuda:0"):
    """加载模型"""
    print(f"Loading config: {config_path}")
    cfg = Config.fromfile(config_path)

    print("Building model...")
    model = build_detector(cfg.model)

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        print(f"✅ Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint, strict=False)
        print("✅ Loaded checkpoint")

    model.eval()
    model.to(device)
    return model, cfg

def main():
    parser = argparse.ArgumentParser(description="EPIC-KITCHENS AdaTAD Inference")
    parser.add_argument("--mode", type=str, choices=["verb", "noun", "both"],
                       default="both", help="Detection mode")
    parser.add_argument("--video-dir", type=str, required=True,
                       help="Directory containing videos")
    parser.add_argument("--output", type=str, default="epic_results.json",
                       help="Output JSON path")
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()

    results = {}

    # Verb检测（动作）
    if args.mode in ["verb", "both"]:
        print("\n=== Verb Detection (动作检测) ===")
        verb_config = "configs/adatad/epic/e2e_epic_videomae_l_ft_768x8_160_adapter_verb.py"
        verb_checkpoint = "/root/OpenTAD/pretrained/adatad/adatad_epic_verb.pth"

        verb_model, verb_cfg = load_model(verb_config, verb_checkpoint, args.device)
        # 实现推理逻辑...
        results["verb"] = {}  # 占位符

    # Noun检测（物体）
    if args.mode in ["noun", "both"]:
        print("\n=== Noun Detection (物体检测) ===")
        noun_config = "configs/adatad/epic/e2e_epic_videomae_l_ft_768x8_160_adapter_noun.py"
        noun_checkpoint = "/root/OpenTAD/pretrained/adatad/adatad_epic_noun.pth"

        noun_model, noun_cfg = load_model(noun_config, noun_checkpoint, args.device)
        # 实现推理逻辑...
        results["noun"] = {}  # 占位符

    # 保存结果
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n✅ Results saved to {args.output}")

if __name__ == "__main__":
    main()
```

---

## 📊 EPIC-KITCHENS类别说明

### Verb（动词）- 97类

常见动作包括：
- **take** - 拿起
- **put** - 放下
- **open** - 打开
- **close** - 关闭
- **cut** - 切
- **pour** - 倒
- **mix** - 混合
- **wash** - 清洗
- 等等...

### Noun（名词）- 293类

常见物体包括：
- **cup** - 杯子
- **plate** - 盘子
- **knife** - 刀
- **bowl** - 碗
- **pan** - 平底锅
- **bottle** - 瓶子
- 等等...

---

## 🎯 针对Vlogger做饭视频的建议

### 推荐使用模式

**同时使用Verb和Noun检测**:
1. **Verb检测**: 识别动作（如"拿起杯子"、"切菜"等）
2. **Noun检测**: 识别物体（如"杯子"、"刀"等）
3. **组合分析**: 将动作和物体组合，得到完整的语义（如"拿起杯子"）

### 应用场景

- **动作定位**: 检测视频中每个动作的时间段
- **物体识别**: 识别视频中出现的物体
- **动作-物体关联**: 分析"谁对什么做了什么"
- **视频摘要**: 自动生成视频的关键动作片段

---

## 📋 完整下载脚本

创建 `download_epic_adatad.sh`:

```bash
#!/bin/bash
# EPIC-KITCHENS AdaTAD 完整下载脚本

set -e

echo "=== EPIC-KITCHENS AdaTAD 下载脚本 ==="
echo ""

# 检查gdown
if ! command -v gdown &> /dev/null; then
    echo "安装gdown..."
    pip install gdown
fi

# 创建目录
mkdir -p /root/OpenTAD/pretrained
mkdir -p /root/OpenTAD/pretrained/adatad
cd /root/OpenTAD/pretrained

echo "📥 步骤1: 下载EPIC-pretrained VideoMAE权重"
echo ""

# EPIC-Verb预训练权重
echo "下载EPIC-Verb预训练权重..."
gdown https://drive.google.com/uc?id=1h7oLiNN5LTXau4HWmmzS_ekvuNdZkp-b \
    -O vit-large-p16_videomae-epic_verb.pth

# EPIC-Noun预训练权重
echo "下载EPIC-Noun预训练权重..."
gdown https://drive.google.com/uc?id=1nRuzJI4ej90vFsKCBSugRVOmxrR8urwW \
    -O vit-large-p16_videomae-epic_noun.pth

echo ""
echo "📥 步骤2: 下载AdaTAD训练好的模型权重"
echo ""

cd /root/OpenTAD/pretrained/adatad

# Verb模型
echo "下载AdaTAD EPIC-Verb模型..."
gdown https://drive.google.com/uc?id=16Hq3sHu0S97Ge2AewHT6DOaHSo0TqIlx \
    -O adatad_epic_verb.pth

# Noun模型
echo "下载AdaTAD EPIC-Noun模型..."
gdown https://drive.google.com/uc?id=17k3f6wirqniLTjKOsIXbfqJPA_iLb88E \
    -O adatad_epic_noun.pth

echo ""
echo "✅ 下载完成！"
echo ""
echo "📊 文件清单："
echo "预训练权重："
ls -lh /root/OpenTAD/pretrained/vit-large-p16_videomae-epic_*.pth
echo ""
echo "AdaTAD模型："
ls -lh /root/OpenTAD/pretrained/adatad/adatad_epic_*.pth
```

---

## 🚀 快速开始（使用预训练权重推理）

### 1. 下载所有权重

```bash
cd /root/OpenTAD
chmod +x download_epic_adatad.sh
./download_epic_adatad.sh
```

### 2. 准备测试视频

```bash
mkdir -p /data/videos/epic_test
# 将您的vlogger做饭视频放入该目录
```

### 3. 运行推理

```bash
# Verb检测（动作）
CUDA_VISIBLE_DEVICES=0 torchrun \
    --nnodes=1 --nproc_per_node=1 \
    --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
    tools/test.py \
    configs/adatad/epic/e2e_epic_videomae_l_ft_768x8_160_adapter_verb.py \
    --checkpoint /root/OpenTAD/pretrained/adatad/adatad_epic_verb.pth
```

---

## ⚠️ 注意事项

1. **数据集下载**: EPIC-KITCHENS-100需要从官网注册下载
2. **GPU内存**: VideoMAE-L模型较大，建议至少8GB显存
3. **视频格式**: 支持MP4等常见格式
4. **第一人称视角**: 模型针对第一人称视角训练，vlogger视频通常也是第一人称，匹配度好

---

## 📚 参考资源

- **EPIC-KITCHENS官网**: https://epic-kitchens.github.io/
- **OpenTAD GitHub**: https://github.com/sming256/OpenTAD
- **AdaTAD README**: `configs/adatad/README.md`
- **数据准备指南**: `tools/prepare_data/epic/README.md`

---

## 🎯 下一步

1. **下载预训练权重和模型**
2. **准备EPIC-KITCHENS数据集**（或使用自己的视频）
3. **运行推理测试**
4. **根据结果调整参数**

**对于vlogger做饭视频，建议同时使用Verb和Noun检测，以获得完整的动作-物体语义！**

