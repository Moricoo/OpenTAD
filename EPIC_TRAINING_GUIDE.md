# EPIC-KITCHENS AdaTAD 训练复现指南

## ✅ 确认：训练所需内容

### 必需文件

1. **EPIC-pretrained VideoMAE权重**（必需）
   - 这是VideoMAE在EPIC-KITCHENS上预训练的backbone权重
   - 作为AdaTAD的backbone初始化
   - **Verb模型**: `pretrained/vit-large-p16_videomae-epic_verb.pth`
   - **Noun模型**: `pretrained/vit-large-p16_videomae-epic_noun.pth`

2. **EPIC-KITCHENS-100数据集**（必需）
   - **原始视频**: `data/epic_kitchens-100/raw_data/epic_kitchens_100_30fps_512x288/`
   - **标注文件**: `data/epic_kitchens-100/annotations/`

### 总结

**是的，您理解正确！**

下载EPIC-pretrained VideoMAE权重后，配合EPIC-KITCHENS数据集，就可以开始训练AdaTAD了。

---

## 📦 完整下载清单

### 1. EPIC-pretrained VideoMAE权重

| 文件 | 用途 | 下载链接 | 保存位置 |
|------|------|----------|----------|
| `vit-large-p16_videomae-epic_verb.pth` | Verb检测backbone | [Link](https://drive.google.com/file/d/1h7oLiNN5LTXau4HWmmzS_ekvuNdZkp-b/view?usp=sharing) | `pretrained/` |
| `vit-large-p16_videomae-epic_noun.pth` | Noun检测backbone | [Link](https://drive.google.com/file/d/1nRuzJI4ej90vFsKCBSugRVOmxrR8urwW/view?usp=sharing) | `pretrained/` |

### 2. EPIC-KITCHENS-100数据集

#### 2.1 标注文件

```bash
cd /root/OpenTAD/tools/prepare_data/epic
bash download_annotation.sh
```

或从 [EPIC-KITCHENS-100 Annotations](https://github.com/epic-kitchens/epic-kitchens-100-annotations) 下载

#### 2.2 原始视频

- **官网**: https://github.com/epic-kitchens/epic-kitchens-download-scripts
- **需要注册**: 填写数据使用协议
- **大小**: 约500GB-1TB
- **保存位置**: `data/epic_kitchens-100/raw_data/epic_kitchens_100_30fps_512x288/`

---

## 🚀 训练步骤

### 步骤1: 下载EPIC-pretrained VideoMAE权重

```bash
# 创建目录
mkdir -p /root/OpenTAD/pretrained
cd /root/OpenTAD/pretrained

# 方法1: 使用gdown（如果网络可用）
pip install gdown
gdown https://drive.google.com/uc?id=1h7oLiNN5LTXau4HWmmzS_ekvuNdZkp-b \
    -O vit-large-p16_videomae-epic_verb.pth
gdown https://drive.google.com/uc?id=1nRuzJI4ej90vFsKCBSugRVOmxrR8urwW \
    -O vit-large-p16_videomae-epic_noun.pth

# 方法2: 手动下载后上传
# scp vit-large-p16_videomae-epic_verb.pth root@<服务器IP>:/root/OpenTAD/pretrained/
# scp vit-large-p16_videomae-epic_noun.pth root@<服务器IP>:/root/OpenTAD/pretrained/
```

### 步骤2: 准备EPIC-KITCHENS数据集

#### 2.1 下载标注文件

```bash
cd /root/OpenTAD/tools/prepare_data/epic
bash download_annotation.sh
```

#### 2.2 下载原始视频

从EPIC-KITCHENS官网下载原始视频，保存到：
```
data/epic_kitchens-100/raw_data/epic_kitchens_100_30fps_512x288/
```

### 步骤3: 开始训练

#### Verb模型训练

```bash
cd /root/OpenTAD

# 使用2个GPU训练（推荐）
torchrun \
    --nnodes=1 \
    --nproc_per_node=2 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    tools/train.py \
    configs/adatad/epic/e2e_epic_videomae_l_ft_768x8_160_adapter_verb.py
```

#### Noun模型训练

```bash
torchrun \
    --nnodes=1 \
    --nproc_per_node=2 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    tools/train.py \
    configs/adatad/epic/e2e_epic_videomae_l_ft_768x8_160_adapter_noun.py
```

#### 单GPU训练（如果只有1个GPU）

```bash
CUDA_VISIBLE_DEVICES=0 torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    tools/train.py \
    configs/adatad/epic/e2e_epic_videomae_l_ft_768x8_160_adapter_verb.py
```

---

## 📊 训练配置说明

### Verb模型配置

**文件**: `configs/adatad/epic/e2e_epic_videomae_l_ft_768x8_160_adapter_verb.py`

**关键参数**:
- **Backbone预训练**: `pretrained/vit-large-p16_videomae-epic_verb.pth`
- **输入**: 768 frames × 8 = 6144 frames, 160x160
- **类别数**: 97个动词
- **训练epochs**: 250
- **Warmup epochs**: 5
- **Batch size**: 2 (per GPU)
- **学习率**:
  - Backbone: 0 (冻结)
  - Adapter: 8e-5
- **GPU数量**: 推荐2个GPU

### Noun模型配置

**文件**: `configs/adatad/epic/e2e_epic_videomae_l_ft_768x8_160_adapter_noun.py`

**关键参数**:
- **Backbone预训练**: `pretrained/vit-large-p16_videomae-epic_noun.pth`
- **输入**: 768 frames × 8 = 6144 frames, 160x160
- **类别数**: 293个名词
- **训练epochs**: 250
- **其他参数**: 与Verb模型相同

---

## 🔍 训练流程说明

### 训练过程

1. **加载EPIC-pretrained VideoMAE权重**
   - 配置文件中的 `pretrain="pretrained/vit-large-p16_videomae-epic_verb.pth"` 会自动加载
   - 这个权重初始化VideoMAE-L backbone

2. **训练AdaTAD**
   - Backbone (VideoMAE): 冻结（lr=0）
   - Adapter: 可训练（lr=8e-5）
   - 端到端训练adapter和projection head

3. **保存checkpoint**
   - 每2个epoch保存一次
   - 保存在: `exps/epic_kitchens/adatad/e2e_actionformer_videomae_l_ft_768x8_160_verb_adapter/checkpoint/`

---

## ⚠️ 重要注意事项

### 1. 数据集路径

确保数据集路径正确：
- **原始视频**: `data/epic_kitchens-100/raw_data/epic_kitchens_100_30fps_512x288/`
- **标注文件**: `data/epic_kitchens-100/annotations/epic_kitchens_verb.json` (或noun.json)

### 2. 预训练权重路径

确保预训练权重在正确位置：
- Verb: `pretrained/vit-large-p16_videomae-epic_verb.pth`
- Noun: `pretrained/vit-large-p16_videomae-epic_noun.pth`

### 3. GPU内存

- **推荐**: 2个GPU，每个至少8GB显存
- **单GPU**: 可能需要减小batch_size

### 4. 训练时间

- **预计时间**: 根据数据集大小和GPU数量，可能需要几天到几周
- **Checkpoint**: 每2个epoch保存，可以随时恢复训练

---

## 📋 训练前检查清单

- [ ] EPIC-pretrained VideoMAE权重已下载
  - [ ] `pretrained/vit-large-p16_videomae-epic_verb.pth`
  - [ ] `pretrained/vit-large-p16_videomae-epic_noun.pth`
- [ ] EPIC-KITCHENS数据集已准备
  - [ ] 标注文件已下载
  - [ ] 原始视频已下载（或准备下载）
- [ ] 配置文件路径正确
- [ ] GPU可用且显存足够

---

## 🎯 快速验证

### 检查预训练权重

```bash
ls -lh /root/OpenTAD/pretrained/vit-large-p16_videomae-epic_*.pth
```

### 检查数据集

```bash
# 检查标注文件
ls -lh /root/OpenTAD/data/epic_kitchens-100/annotations/

# 检查视频文件（如果已下载）
ls -lh /root/OpenTAD/data/epic_kitchens-100/raw_data/epic_kitchens_100_30fps_512x288/ | head -10
```

### 测试训练（小规模测试）

可以先运行几个epoch测试：

```bash
# 修改配置文件中的end_epoch为较小的值（如5）进行测试
# 或使用--cfg-options覆盖
torchrun \
    --nnodes=1 --nproc_per_node=2 \
    --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
    tools/train.py \
    configs/adatad/epic/e2e_epic_videomae_l_ft_768x8_160_adapter_verb.py \
    --cfg-options workflow.end_epoch=5
```

---

## 📚 参考

- **训练命令**: 见 `configs/adatad/README.md` 第107-111行
- **配置文件**:
  - Verb: `configs/adatad/epic/e2e_epic_videomae_l_ft_768x8_160_adapter_verb.py`
  - Noun: `configs/adatad/epic/e2e_epic_videomae_l_ft_768x8_160_adapter_noun.py`
- **数据准备**: `tools/prepare_data/epic/README.md`

---

## ✅ 总结

**是的，您的理解完全正确！**

1. ✅ 下载EPIC-pretrained VideoMAE权重
2. ✅ 准备EPIC-KITCHENS数据集
3. ✅ 运行训练命令

就可以开始训练AdaTAD，复现在EPIC-KITCHENS上的效果了！

**注意**: 如果只是想推理（不训练），可以直接下载已经训练好的AdaTAD权重，不需要训练过程。

