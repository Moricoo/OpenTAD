# 最小的端到端模型训练指南

## 模型介绍

**AdaTAD with VideoMAE-S (Adapter 模式)** 是最小的端到端模型，具有以下特点：

- **Backbone**: VideoMAE-S (Small) - 最小的 VideoMAE 模型
- **训练模式**: Adapter - 只训练 adapter 参数，冻结 backbone
- **内存占用**: 最小（每个 GPU 约 10-12 GB）
- **训练速度**: 相对较快
- **性能**: 在 THUMOS-14 上达到 ~69.03% 平均 mAP

## 为什么选择这个模型？

1. **内存友好**: Adapter 模式只训练少量参数，显存占用最小
2. **训练快速**: 相比 full fine-tuning，训练时间更短
3. **性能良好**: 虽然是最小模型，但性能仍然不错
4. **易于调试**: 适合快速实验和验证

## 准备工作

### 1. 下载原始视频数据

端到端训练需要原始视频文件，而不是预提取的特征。

```bash
# 参考文档
cat tools/prepare_data/thumos/README.md

# 需要下载 THUMOS-14 视频文件
# 将视频放在: data/thumos-14/videos/
```

### 2. 下载预训练模型

需要下载 VideoMAE-S 预训练模型。**注意**: 配置文件中使用的文件名是 `vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth`（带 `_my` 后缀）。

```bash
# 方法 1: 使用自动下载脚本（推荐）
cd /root/OpenTAD
bash scripts/utils/download_videomae_s.sh

# 方法 2: 使用 gdown 从 Google Drive 下载
pip install gdown
mkdir -p pretrained
cd pretrained
gdown "https://drive.google.com/uc?id=1BH5BZmdImaZesUfqtW23eBGC341Gui1D" \
    -O vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-860a3cd3.pth
# 创建符号链接（配置文件需要带 _my 后缀的文件名）
ln -sf vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-860a3cd3.pth \
    vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth

# 方法 3: 手动从 Google Drive 下载
# URL: https://drive.google.com/file/d/1BH5BZmdImaZesUfqtW23eBGC341Gui1D/view?usp=sharing
# 下载后：
# 1. 保存到: pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-860a3cd3.pth
# 2. 创建符号链接:
#    cd pretrained
#    ln -sf vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-860a3cd3.pth \
#      vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth
```

**重要提示**: 
- mmaction2 的下载链接可能已失效（404 错误），推荐使用 Google Drive 链接
- 配置文件需要文件名带 `_my` 后缀，下载后需要创建符号链接或重命名

### 3. 检查数据准备

```bash
# 检查视频目录
ls -lh data/thumos-14/videos/ | head -10

# 检查预训练模型
ls -lh pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-860a3cd3.pth
```

## 训练

### 使用脚本（推荐）

```bash
cd /root/OpenTAD
bash scripts/training/run_e2e_minimal.sh
```

### 手动运行

```bash
cd /root/OpenTAD
source /root/miniconda3/bin/activate opentad

torchrun \
    --nnodes=1 \
    --nproc_per_node=2 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    tools/train.py \
    configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py
```

### 如果只有 1 个 GPU

如果只有 1 个 GPU，可以修改配置或使用单 GPU 训练：

```bash
torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    tools/train.py \
    configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py \
    --cfg-options solver.train.batch_size=1
```

## 测试

### 使用脚本

```bash
bash scripts/testing/test_e2e_minimal.sh
# 或指定检查点
bash scripts/testing/test_e2e_minimal.sh exps/thumos/e2e_thumos_videomae_s_768x1_160_adapter/gpu2_id0/checkpoint/epoch_10.pth
```

### 手动运行

```bash
torchrun \
    --nnodes=1 \
    --nproc_per_node=2 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    tools/test.py \
    configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py \
    --checkpoint exps/thumos/e2e_thumos_videomae_s_768x1_160_adapter/gpu2_id0/checkpoint/best.pth
```

## 预期性能

根据论文，VideoMAE-S (Adapter) 在 THUMOS-14 上的性能：

| mAP@0.3 | mAP@0.4 | mAP@0.5 | mAP@0.6 | mAP@0.7 | 平均 mAP |
| :-----: | :-----: | :-----: | :-----: | :-----: | :------: |
|  83.90  |  79.01  |  72.38  |  61.57  |  48.27  |  69.03   |

## 配置说明

### 关键参数

- **window_size**: 768 帧
- **img_size**: 160x160
- **batch_size**: 根据 GPU 数量自动调整
- **训练模式**: Adapter（只训练 adapter，不训练 backbone）

### 与其他模式对比

| 模式 | 训练参数 | 内存占用 | 训练速度 | 性能 |
| :--: | :------: | :------: | :------: | :--: |
| **Adapter** | 最少 | 最小 | 最快 | 良好 |
| Frozen | 中等 | 中等 | 中等 | 较好 |
| Full Fine-tune | 最多 | 最大 | 最慢 | 最好 |

## 资源需求

- **GPU**: 2 个 GPU（推荐）或 1 个 GPU（需要减小 batch size）
- **显存**: 每个 GPU 约 10-12 GB
- **训练时间**: 约 10-15 小时（取决于硬件）
- **磁盘空间**: 需要存储原始视频和训练输出

## 常见问题

### 1. 显存不足

如果显存不足，可以：
- 减小 batch size: `--cfg-options solver.train.batch_size=1`
- 使用 1 个 GPU 训练
- 减小图像尺寸（需要修改配置）

### 2. 训练速度慢

- 确保使用 GPU 训练
- 检查数据加载是否正常
- 考虑使用更多 GPU（如果可用）

### 3. 找不到视频文件

- 检查视频路径: `data/thumos-14/videos/`
- 确保视频文件格式正确（通常是 mp4）
- 参考数据准备文档

## 下一步

训练完成后，可以尝试：

1. **测试模型性能**: 在测试集上评估
2. **尝试更大的模型**: VideoMAE-B, VideoMAE-L 等
3. **尝试其他模式**: Frozen 或 Full Fine-tune
4. **调整超参数**: 学习率、训练轮数等

## 参考

- **论文**: [End-to-End Temporal Action Detection with 1B Parameters Across 1000 Frames](https://arxiv.org/abs/2311.17241)
- **配置目录**: `configs/adatad/thumos/`
- **README**: `configs/adatad/README.md`

