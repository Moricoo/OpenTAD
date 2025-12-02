# EPIC-KITCHENS AdaTAD 完整部署指南 - Vlogger做饭视频分析

## 🎯 应用场景

**目标**: 使用AdaTAD在EPIC-KITCHENS上训练的模型，分析vlogger做饭视频中的动作和物体定位。

**EPIC-KITCHENS特点**:
- ✅ **第一人称视角**（egocentric vision）- 完美匹配vlogger视频
- ✅ **厨房场景**的日常活动
- ✅ **细粒度动作**检测（97个动词 + 293个名词）
- ✅ **Verb（动词）**: take, put, open, close, cut, pour, mix, wash等
- ✅ **Noun（名词）**: cup, plate, knife, bowl, pan, bottle等

---

## 📦 需要下载的内容清单

### 1. EPIC-pretrained VideoMAE权重（必需）

这些是EPIC-KITCHENS上预训练的VideoMAE权重，AdaTAD需要这些作为backbone：

| 文件 | 用途 | Google Drive链接 | 大小（估算） |
|------|------|----------------|-------------|
| `vit-large-p16_videomae-epic_verb.pth` | Verb检测backbone | [Link](https://drive.google.com/file/d/1h7oLiNN5LTXau4HWmmzS_ekvuNdZkp-b/view?usp=sharing) | ~1.2GB |
| `vit-large-p16_videomae-epic_noun.pth` | Noun检测backbone | [Link](https://drive.google.com/file/d/1nRuzJI4ej90vFsKCBSugRVOmxrR8urwW/view?usp=sharing) | ~1.2GB |

**保存位置**: `/root/OpenTAD/pretrained/`

### 2. AdaTAD训练好的模型权重（用于推理）

这些是AdaTAD在EPIC-KITCHENS上训练好的完整模型：

| 文件 | 类别数 | 性能 | Google Drive链接 | 大小（估算） |
|------|--------|------|----------------|-------------|
| `adatad_epic_verb.pth` | 97类动词 | ave. mAP=29.69% | [Link](https://drive.google.com/file/d/16Hq3sHu0S97Ge2AewHT6DOaHSo0TqIlx/view?usp=sharing) | ~2GB |
| `adatad_epic_noun.pth` | 293类名词 | ave. mAP=29.44% | [Link](https://drive.google.com/file/d/17k3f6wirqniLTjKOsIXbfqJPA_iLb88E/view?usp=sharing) | ~2GB |

**保存位置**: `/root/OpenTAD/pretrained/adatad/`

### 3. EPIC-KITCHENS-100数据集

#### 3.1 标注文件（必需）

```bash
cd /root/OpenTAD/tools/prepare_data/epic
bash download_annotation.sh
```

或手动下载：
- 从 [EPIC-KITCHENS-100 Annotations](https://github.com/epic-kitchens/epic-kitchens-100-annotations) 下载
- 保存到: `data/epic_kitchens-100/annotations/`

#### 3.2 原始视频（可选，用于训练）

- **官网**: https://github.com/epic-kitchens/epic-kitchens-download-scripts
- **需要注册**: 填写数据使用协议
- **大小**: 约500GB-1TB
- **保存位置**: `data/epic_kitchens-100/raw_data/epic_kitchens_100_30fps_512x288/`

**注意**: 如果只是推理，可以使用自己的vlogger视频，不需要下载完整数据集。

---

## 🚀 完整部署步骤

### 步骤1: 下载EPIC-pretrained VideoMAE权重

#### 方法A: 使用gdown（如果网络可用）

```bash
# 安装gdown
pip install gdown

# 创建目录
mkdir -p /root/OpenTAD/pretrained
cd /root/OpenTAD/pretrained

# 下载Verb预训练权重
gdown https://drive.google.com/uc?id=1h7oLiNN5LTXau4HWmmzS_ekvuNdZkp-b \
    -O vit-large-p16_videomae-epic_verb.pth

# 下载Noun预训练权重
gdown https://drive.google.com/uc?id=1nRuzJI4ej90vFsKCBSugRVOmxrR8urwW \
    -O vit-large-p16_videomae-epic_noun.pth
```

#### 方法B: 手动下载（推荐）

1. 在浏览器中打开Google Drive链接
2. 下载文件到本地
3. 使用scp上传到服务器：

```bash
# 在本地电脑执行
scp vit-large-p16_videomae-epic_verb.pth root@<服务器IP>:/root/OpenTAD/pretrained/
scp vit-large-p16_videomae-epic_noun.pth root@<服务器IP>:/root/OpenTAD/pretrained/
```

#### 方法C: 使用百度网盘（如果提供）

如果权重已上传到百度网盘，使用之前的bypy下载方法。

### 步骤2: 下载AdaTAD训练好的模型权重

```bash
# 创建目录
mkdir -p /root/OpenTAD/pretrained/adatad
cd /root/OpenTAD/pretrained/adatad

# 方法A: 使用gdown
gdown https://drive.google.com/uc?id=16Hq3sHu0S97Ge2AewHT6DOaHSo0TqIlx \
    -O adatad_epic_verb.pth

gdown https://drive.google.com/uc?id=17k3f6wirqniLTjKOsIXbfqJPA_iLb88E \
    -O adatad_epic_noun.pth

# 方法B: 手动下载后上传
# scp adatad_epic_verb.pth root@<服务器IP>:/root/OpenTAD/pretrained/adatad/
# scp adatad_epic_noun.pth root@<服务器IP>:/root/OpenTAD/pretrained/adatad/
```

### 步骤3: 下载EPIC-KITCHENS标注文件

```bash
cd /root/OpenTAD/tools/prepare_data/epic

# 下载标注文件
bash download_annotation.sh

# 验证
ls -lh ../../../data/epic_kitchens-100/annotations/
```

### 步骤4: 准备视频数据

#### 选项A: 使用EPIC-KITCHENS数据集（完整训练/测试）

```bash
# 从官网下载原始视频
# 保存到: data/epic_kitchens-100/raw_data/epic_kitchens_100_30fps_512x288/
```

#### 选项B: 使用自己的vlogger视频（推荐用于快速测试）

```bash
# 创建测试目录
mkdir -p /data/videos/epic_test

# 放置您的vlogger做饭视频
# cp your_cooking_video.mp4 /data/videos/epic_test/
```

---

## 🔧 配置文件说明

### Verb模型配置

**文件**: `configs/adatad/epic/e2e_epic_videomae_l_ft_768x8_160_adapter_verb.py`

**关键参数**:
- **Backbone**: VideoMAE-L (1024维, 24层)
- **输入**: 768 frames × 8 = 6144 frames, 160x160
- **类别数**: 97个动词
- **预训练**: `pretrained/vit-large-p16_videomae-epic_verb.pth`
- **性能**: mAP@0.5=24.69%, ave. mAP=29.69%

### Noun模型配置

**文件**: `configs/adatad/epic/e2e_epic_videomae_l_ft_768x8_160_adapter_noun.py`

**关键参数**:
- **Backbone**: VideoMAE-L (1024维, 24层)
- **输入**: 768 frames × 8 = 6144 frames, 160x160
- **类别数**: 293个名词
- **预训练**: `pretrained/vit-large-p16_videomae-epic_noun.pth`
- **性能**: mAP@0.5=22.67%, ave. mAP=29.44%

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

### 方法2: 同时检测Verb和Noun

创建组合推理脚本（见下方）

---

## 🎯 针对Vlogger做饭视频的建议

### 推荐使用模式

**同时使用Verb和Noun检测**:

1. **Verb检测**: 识别动作（如"拿起"、"切"、"倒"等）
2. **Noun检测**: 识别物体（如"杯子"、"刀"、"平底锅"等）
3. **组合分析**: 将动作和物体组合，得到完整的语义（如"拿起杯子"）

### 应用场景

- **动作定位**: 检测视频中每个动作的时间段
- **物体识别**: 识别视频中出现的物体
- **动作-物体关联**: 分析"谁对什么做了什么"
- **视频摘要**: 自动生成视频的关键动作片段
- **时间线分析**: 生成做饭步骤的时间线

---

## 📋 完整下载清单

### 必需文件

1. ✅ `pretrained/vit-large-p16_videomae-epic_verb.pth` (~1.2GB)
2. ✅ `pretrained/vit-large-p16_videomae-epic_noun.pth` (~1.2GB)
3. ✅ `pretrained/adatad/adatad_epic_verb.pth` (~2GB)
4. ✅ `pretrained/adatad/adatad_epic_noun.pth` (~2GB)
5. ✅ `data/epic_kitchens-100/annotations/` (标注文件)

### 可选文件

6. ⚠️ `data/epic_kitchens-100/raw_data/` (原始视频，500GB-1TB，仅训练需要)

**总计必需**: 约6.4GB（不包括原始视频）

---

## 🔍 验证下载

```bash
# 检查预训练权重
ls -lh /root/OpenTAD/pretrained/vit-large-p16_videomae-epic_*.pth

# 检查AdaTAD模型
ls -lh /root/OpenTAD/pretrained/adatad/adatad_epic_*.pth

# 检查标注文件
ls -lh /root/OpenTAD/data/epic_kitchens-100/annotations/
```

---

## 🚀 快速开始（使用预训练权重推理）

### 1. 下载所有权重（手动或使用脚本）

```bash
# 如果网络可用，使用脚本
cd /root/OpenTAD
./download_epic_adatad.sh

# 如果网络不可用，手动下载后上传
```

### 2. 下载标注文件

```bash
cd /root/OpenTAD/tools/prepare_data/epic
bash download_annotation.sh
```

### 3. 准备测试视频

```bash
mkdir -p /data/videos/epic_test
# 将您的vlogger做饭视频放入该目录
```

### 4. 运行推理

```bash
# Verb检测
cd /root/OpenTAD
CUDA_VISIBLE_DEVICES=0 torchrun \
    --nnodes=1 --nproc_per_node=1 \
    --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
    tools/test.py \
    configs/adatad/epic/e2e_epic_videomae_l_ft_768x8_160_adapter_verb.py \
    --checkpoint /root/OpenTAD/pretrained/adatad/adatad_epic_verb.pth
```

---

## ⚠️ 注意事项

1. **GPU内存**: VideoMAE-L模型较大，建议至少8GB显存
2. **视频格式**: 支持MP4等常见格式
3. **第一人称视角**: 模型针对第一人称视角训练，vlogger视频通常也是第一人称，匹配度好
4. **输入要求**:
   - 时序长度: 768 frames × 8 = 6144 frames
   - 空间尺寸: 160x160
5. **类别限制**:
   - Verb: 只能检测97个预定义的动词
   - Noun: 只能检测293个预定义的名词

---

## 📚 参考资源

- **EPIC-KITCHENS官网**: https://epic-kitchens.github.io/
- **EPIC-KITCHENS下载**: https://github.com/epic-kitchens/epic-kitchens-download-scripts
- **OpenTAD GitHub**: https://github.com/sming256/OpenTAD
- **AdaTAD README**: `configs/adatad/README.md`
- **数据准备指南**: `tools/prepare_data/epic/README.md`

---

## 🎯 下一步

1. **下载预训练权重和模型**（手动或使用脚本）
2. **下载标注文件**
3. **准备测试视频**（可以使用自己的vlogger视频）
4. **运行推理测试**
5. **根据结果调整参数**

**对于vlogger做饭视频，建议同时使用Verb和Noun检测，以获得完整的动作-物体语义！**

