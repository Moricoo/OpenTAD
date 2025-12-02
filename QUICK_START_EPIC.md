# EPIC-KITCHENS AdaTAD 快速开始指南

## 🎯 目标
使用AdaTAD在EPIC-KITCHENS上训练的模型，分析vlogger做饭视频中的动作和物体定位。

## 📦 快速下载清单

### 必需文件（约6.4GB）

1. **EPIC-pretrained VideoMAE权重** (2个文件, ~2.4GB)
   - Verb: https://drive.google.com/file/d/1h7oLiNN5LTXau4HWmmzS_ekvuNdZkp-b/view?usp=sharing
   - Noun: https://drive.google.com/file/d/1nRuzJI4ej90vFsKCBSugRVOmxrR8urwW/view?usp=sharing

2. **AdaTAD模型权重** (2个文件, ~4GB)
   - Verb: https://drive.google.com/file/d/16Hq3sHu0S97Ge2AewHT6DOaHSo0TqIlx/view?usp=sharing
   - Noun: https://drive.google.com/file/d/17k3f6wirqniLTjKOsIXbfqJPA_iLb88E/view?usp=sharing

3. **EPIC-KITCHENS标注文件**
   ```bash
   cd /root/OpenTAD/tools/prepare_data/epic
   bash download_annotation.sh
   ```

## 🚀 三步快速开始

### 步骤1: 下载文件

```bash
# 方法A: 手动下载（推荐）
# 1. 在浏览器中打开Google Drive链接
# 2. 下载4个.pth文件
# 3. 上传到服务器对应目录

# 方法B: 使用gdown（如果网络可用）
cd /root/OpenTAD
./download_epic_adatad.sh
```

### 步骤2: 检查准备状态

```bash
cd /root/OpenTAD
./setup_epic_adatad.sh
```

### 步骤3: 运行推理

```bash
# Verb检测（动作）
CUDA_VISIBLE_DEVICES=0 torchrun \
    --nnodes=1 --nproc_per_node=1 \
    --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
    tools/test.py \
    configs/adatad/epic/e2e_epic_videomae_l_ft_768x8_160_adapter_verb.py \
    --checkpoint pretrained/adatad/adatad_epic_verb.pth
```

## 📋 文件保存位置

- 预训练权重: `/root/OpenTAD/pretrained/vit-large-p16_videomae-epic_*.pth`
- AdaTAD模型: `/root/OpenTAD/pretrained/adatad/adatad_epic_*.pth`
- 标注文件: `/root/OpenTAD/data/epic_kitchens-100/annotations/`

## 💡 针对Vlogger做饭视频

- **Verb检测**: 识别动作（take, put, cut, pour等）
- **Noun检测**: 识别物体（cup, plate, knife等）
- **组合使用**: 同时运行两个模型，获得完整的动作-物体语义

详细说明请查看: `EPIC_KITCHENS_COMPLETE_GUIDE.md`
