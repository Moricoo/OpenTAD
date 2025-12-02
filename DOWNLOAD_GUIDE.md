# 模型下载指南

## 📦 已打包的模型文件

训练好的模型已打包完成，包含以下内容：

### 文件位置

1. **压缩包**（推荐下载）:
   - 路径: `/root/OpenTAD/thumos_adapter_baseline.tar.gz`
   - 大小: 1.6GB
   - 包含: 所有模型文件、配置、预训练权重

2. **目录**（可选）:
   - 路径: `/root/OpenTAD/model_package_thumos_adapter_baseline/`
   - 大小: 1.9GB

### 包含内容

```
model_package_thumos_adapter_baseline/
├── checkpoint/
│   ├── latest.pth          # 最新模型权重（epoch_59, 595MB）
│   ├── epoch_57.pth        # 历史checkpoint
│   └── epoch_59.pth        # 历史checkpoint
├── config/
│   └── e2e_thumos_videomae_s_768x1_160_adapter.py  # 完整配置文件
├── pretrained/
│   └── vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth  # 预训练backbone (87MB)
├── inference_example.py    # 推理示例代码
└── README.md              # 详细使用说明
```

## 📥 下载方法

### 方法1: 下载压缩包（推荐 ⭐）

**优点**: 单文件，传输快，支持断点续传

```bash
# 在本地机器执行
# 替换 <服务器IP> 为实际服务器IP地址
scp root@<服务器IP>:/root/OpenTAD/thumos_adapter_baseline.tar.gz ./

# 解压
tar -xzf thumos_adapter_baseline.tar.gz

# 查看内容
cd model_package_thumos_adapter_baseline
ls -lh
```

### 方法2: 使用rsync（支持断点续传）

**优点**: 支持断点续传，显示进度

```bash
# 在本地机器执行
rsync -avz --progress root@<服务器IP>:/root/OpenTAD/model_package_thumos_adapter_baseline ./
```

### 方法3: 直接下载目录

```bash
# 在本地机器执行
scp -r root@<服务器IP>:/root/OpenTAD/model_package_thumos_adapter_baseline ./
```

## 🚀 快速开始

### 1. 下载后解压（如果下载的是压缩包）

```bash
tar -xzf thumos_adapter_baseline.tar.gz
cd model_package_thumos_adapter_baseline
```

### 2. 检查文件完整性

```bash
# 检查关键文件是否存在
ls -lh checkpoint/latest.pth      # 应该约595MB
ls -lh config/*.py                # 配置文件
ls -lh pretrained/*.pth           # 预训练模型，约87MB
```

### 3. 加载模型（Python示例）

```python
import torch
from mmengine import Config
from opentad.models import build_detector

# 加载配置
cfg = Config.fromfile('model_package_thumos_adapter_baseline/config/e2e_thumos_videomae_s_768x1_160_adapter.py')

# 构建模型
model = build_detector(cfg.model)

# 加载checkpoint
checkpoint = torch.load('model_package_thumos_adapter_baseline/checkpoint/latest.pth', map_location='cpu')
if 'state_dict' in checkpoint:
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    print(f"✅ 已加载模型 (epoch {checkpoint.get('epoch', 'unknown')})")
else:
    model.load_state_dict(checkpoint, strict=False)

model.eval()
model.to('cuda:0')
print("✅ 模型已准备就绪！")
```

### 4. 使用OpenTAD测试脚本推理

```bash
# 确保在OpenTAD项目根目录
cd /path/to/OpenTAD

# 单GPU推理
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
    model_package_thumos_adapter_baseline/config/e2e_thumos_videomae_s_768x1_160_adapter.py \
    model_package_thumos_adapter_baseline/checkpoint/latest.pth
```

## 📋 模型信息

- **模型类型**: VisionTransformerAdapter (AdaTAD)
- **数据集**: THUMOS-14
- **训练epoch**: 60 (最新checkpoint: epoch_59)
- **输入尺寸**: 160x160
- **窗口大小**: 768 frames
- **Backbone**: VideoMAE-S (ViT-Small, 384维)
- **Adapter**: TIA (Temporal Interaction Adapter)

## ⚠️ 注意事项

1. **预训练模型路径**: 确保预训练模型文件在正确位置，或修改配置文件中的路径
2. **环境依赖**: 需要安装OpenTAD框架及其依赖（PyTorch, mmengine, mmcv等）
3. **GPU要求**: 推理建议使用GPU，至少需要4GB显存
4. **输入格式**: 模型输入格式为NCTHW，需要与训练时相同的数据预处理

## 📚 更多信息

详细使用说明请查看 `model_package_thumos_adapter_baseline/README.md`

## 🔗 相关文件

- 配置文件: `configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py`
- 训练日志: `exps/thumos/adatad/e2e_actionformer_videomae_s_768x1_160_adapter/gpu1_id0/log.json`

