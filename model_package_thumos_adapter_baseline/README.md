# THUMOS-14 时序动作检测模型

## 模型信息

- **模型类型**: VisionTransformerAdapter (AdaTAD)
- **数据集**: THUMOS-14
- **输入尺寸**: 160x160
- **窗口大小**: 768 frames
- **训练epoch**: 60
- **最新checkpoint**: epoch_59

## 文件结构

```
model_package/
├── checkpoint/
│   ├── latest.pth          # 最新模型权重（epoch_59）
│   ├── epoch_57.pth        # 历史checkpoint
│   └── epoch_59.pth        # 历史checkpoint
├── config/
│   └── e2e_thumos_videomae_s_768x1_160_adapter.py  # 训练配置文件
├── pretrained/
│   └── vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth  # 预训练backbone
├── inference_example.py    # 推理示例脚本
└── README.md              # 本文件
```

## 下载方法

### 方法1: 直接下载压缩包（推荐）

服务器上已创建压缩包：`thumos_adapter_baseline.tar.gz` (约1.6GB)

```bash
# 在本地机器执行
scp root@<服务器IP>:/root/OpenTAD/thumos_adapter_baseline.tar.gz ./

# 解压
tar -xzf thumos_adapter_baseline.tar.gz
```

### 方法2: 使用rsync（支持断点续传）

```bash
# 在本地机器执行
rsync -avz --progress root@<服务器IP>:/root/OpenTAD/model_package_thumos_adapter_baseline ./
```

### 方法3: 使用scp直接下载目录

```bash
# 在本地机器执行
scp -r root@<服务器IP>:/root/OpenTAD/model_package_thumos_adapter_baseline ./
```

## 使用方法

### 1. 环境要求

```bash
# 安装OpenTAD框架（参考原始项目README）
# 需要的主要依赖：
# - PyTorch >= 1.8.0
# - mmengine
# - mmcv
# - decord (视频解码)
# - 其他依赖见 requirements.txt
pip install -r requirements.txt
```

### 2. 加载模型进行推理

#### 方法A: 使用OpenTAD测试脚本（推荐）

```bash
# 单GPU推理
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
    model_package_thumos_adapter_baseline/config/e2e_thumos_videomae_s_768x1_160_adapter.py \
    model_package_thumos_adapter_baseline/checkpoint/latest.pth

# 多GPU推理
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 tools/test.py \
    model_package_thumos_adapter_baseline/config/e2e_thumos_videomae_s_768x1_160_adapter.py \
    model_package_thumos_adapter_baseline/checkpoint/latest.pth
```

#### 方法B: 在Python代码中加载

```python
import torch
from mmengine import Config
from opentad.models import build_detector

# 加载配置
cfg = Config.fromfile('model_package_thumos_adapter_baseline/config/e2e_thumos_videomae_s_768x1_160_adapter.py')

# 构建模型
model = build_detector(cfg.model)

# 加载checkpoint
checkpoint_path = 'model_package_thumos_adapter_baseline/checkpoint/latest.pth'
checkpoint = torch.load(checkpoint_path, map_location='cpu')

if 'state_dict' in checkpoint:
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    print(f"✅ 已加载模型 (epoch {checkpoint.get('epoch', 'unknown')})")
else:
    model.load_state_dict(checkpoint, strict=False)
    print("✅ 已加载模型权重")

model.eval()
model.to('cuda:0')

# 进行推理...
```

### 3. 模型输入格式

- **输入形状**: `(B, C, T, H, W)` = `(Batch, Channels, Time, Height, Width)`
- **输入尺寸**: 160x160
- **时间窗口**: 768 frames
- **数据格式**: NCTHW
- **归一化**: mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]

## 模型参数

- **Backbone**: VideoMAE-S (ViT-Small)
- **Adapter**: TIA (Temporal Interaction Adapter)
- **Adapter位置**: 所有12个transformer blocks
- **学习率**: Backbone=0, Adapter=2e-4
- **优化器**: AdamW
- **训练策略**: EMA, AMP, Activation Checkpointing

## 注意事项

1. 确保预训练模型文件存在，路径在配置文件中指定
2. 推理时需要与训练时相同的数据预处理流程
3. 模型输入格式: NCTHW (Batch, Channels, Time, Height, Width)
4. 窗口大小: 768 frames，使用sliding window进行长视频推理

## 模型结构说明

### Backbone
- **类型**: VideoMAE-S (Vision Transformer Small)
- **输入尺寸**: 224x224 (训练时resize到160x160)
- **Patch大小**: 16x16
- **嵌入维度**: 384
- **深度**: 12层
- **注意力头数**: 6

### Adapter
- **类型**: TIA (Temporal Interaction Adapter)
- **位置**: 所有12个transformer blocks
- **结构**: Down-projection → Temporal Conv → Up-projection
- **可训练参数**: 仅adapter参数（backbone冻结）

### 输出
- **特征维度**: 384
- **时序长度**: 768 (可插值到任意长度)

## 性能指标

（请根据实际评估结果填写）

- mAP@0.5:
- mAP@0.75:
- Average mAP:

## 常见问题

### Q1: 如何修改输入视频尺寸？
A: 修改配置文件中的 `dataset.test.pipeline` 中的 `Resize` 和 `CenterCrop` 参数。

### Q2: 如何处理长视频？
A: 模型使用sliding window方法，配置文件中已设置 `window_size=768`。长视频会自动切分成多个窗口进行推理。

### Q3: 预训练模型路径错误？
A: 确保 `pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth` 文件存在，或修改配置文件中的路径。

### Q4: 内存不足？
A: 减小 `solver.test.batch_size` 和 `num_workers`，或使用更小的模型。

## 文件清单

- ✅ `checkpoint/latest.pth` - 最新模型权重 (595MB)
- ✅ `checkpoint/epoch_57.pth` - 历史checkpoint
- ✅ `checkpoint/epoch_59.pth` - 历史checkpoint
- ✅ `config/e2e_thumos_videomae_s_768x1_160_adapter.py` - 完整配置文件
- ✅ `pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth` - 预训练backbone (87MB)
- ✅ `inference_example.py` - 推理示例代码
- ✅ `README.md` - 本文件

## 联系信息

如有问题，请参考OpenTAD项目文档或提交issue。
