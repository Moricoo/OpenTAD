#!/bin/bash
# 模型打包脚本 - 用于导出训练好的模型到本地使用

MODEL_NAME="thumos_adapter_baseline"
PACKAGE_DIR="model_package_${MODEL_NAME}"
CHECKPOINT_DIR="exps/thumos/adatad/e2e_actionformer_videomae_s_768x1_160_adapter/gpu1_id0"
CONFIG_FILE="configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py"

echo "=== 开始打包模型 ==="
echo ""

# 创建打包目录
mkdir -p ${PACKAGE_DIR}
mkdir -p ${PACKAGE_DIR}/checkpoint
mkdir -p ${PACKAGE_DIR}/config
mkdir -p ${PACKAGE_DIR}/pretrained

# 1. 复制最新的checkpoint（只复制最后几个epoch）
echo "📦 复制checkpoint文件..."
LATEST_CHECKPOINT=$(ls -t ${CHECKPOINT_DIR}/checkpoint/epoch_*.pth | head -1)
cp ${LATEST_CHECKPOINT} ${PACKAGE_DIR}/checkpoint/latest.pth
echo "  ✅ 已复制: $(basename ${LATEST_CHECKPOINT}) -> latest.pth"

# 可选：复制最后3个checkpoint
for epoch in 57 59; do
    if [ -f "${CHECKPOINT_DIR}/checkpoint/epoch_${epoch}.pth" ]; then
        cp ${CHECKPOINT_DIR}/checkpoint/epoch_${epoch}.pth ${PACKAGE_DIR}/checkpoint/
        echo "  ✅ 已复制: epoch_${epoch}.pth"
    fi
done

# 2. 复制配置文件
echo ""
echo "📦 复制配置文件..."
cp ${CONFIG_FILE} ${PACKAGE_DIR}/config/
echo "  ✅ 已复制: $(basename ${CONFIG_FILE})"

# 3. 复制预训练模型（如果需要）
echo ""
echo "📦 检查预训练模型..."
if [ -f "pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth" ]; then
    cp pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth ${PACKAGE_DIR}/pretrained/
    echo "  ✅ 已复制预训练模型"
else
    echo "  ⚠️  预训练模型不存在，请手动下载"
fi

# 4. 创建推理脚本
echo ""
echo "📦 创建推理脚本..."
cat > ${PACKAGE_DIR}/inference_example.py << 'EOF'
"""
时序动作检测模型推理示例
使用方法：
    python inference_example.py --video <video_path> --checkpoint checkpoint/latest.pth --config config/e2e_thumos_videomae_s_768x1_160_adapter.py
"""

import argparse
import torch
from mmengine import Config
from opentad.models import build_model
from opentad.datasets import build_dataset, build_dataloader

def main():
    parser = argparse.ArgumentParser(description='时序动作检测推理')
    parser.add_argument('--video', type=str, required=True, help='输入视频路径')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型checkpoint路径')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--device', type=str, default='cuda:0', help='推理设备')
    args = parser.parse_args()

    # 加载配置
    cfg = Config.fromfile(args.config)

    # 构建模型
    model = build_model(cfg.model)

    # 加载checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model.eval()
    model.to(args.device)

    print(f"✅ 模型已加载到 {args.device}")
    print(f"📹 开始处理视频: {args.video}")

    # TODO: 实现视频推理逻辑
    # 这里需要根据OpenTAD的实际推理接口来实现

    print("推理完成！")

if __name__ == '__main__':
    main()
EOF
echo "  ✅ 已创建: inference_example.py"

# 5. 创建README
echo ""
echo "📦 创建README文档..."
cat > ${PACKAGE_DIR}/README.md << 'EOF'
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

## 使用方法

### 1. 环境要求

```bash
# 安装OpenTAD框架（参考原始项目README）
pip install -r requirements.txt
```

### 2. 加载模型

```python
import torch
from mmengine import Config
from opentad.models import build_model

# 加载配置
cfg = Config.fromfile('config/e2e_thumos_videomae_s_768x1_160_adapter.py')

# 构建模型
model = build_model(cfg.model)

# 加载checkpoint
checkpoint = torch.load('checkpoint/latest.pth', map_location='cpu')
if 'state_dict' in checkpoint:
    model.load_state_dict(checkpoint['state_dict'], strict=False)
else:
    model.load_state_dict(checkpoint, strict=False)

model.eval()
model.to('cuda:0')
```

### 3. 推理

使用提供的 `inference_example.py` 脚本，或参考OpenTAD项目的测试脚本：

```bash
python tools/test.py config/e2e_thumos_videomae_s_768x1_160_adapter.py \
    checkpoint/latest.pth \
    --video <your_video_path>
```

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

## 性能指标

（请根据实际评估结果填写）

- mAP@0.5:
- mAP@0.75:
- Average mAP:

## 联系信息

如有问题，请参考OpenTAD项目文档或提交issue。
EOF
echo "  ✅ 已创建: README.md"

# 6. 计算文件大小
echo ""
echo "=== 打包完成 ==="
echo ""
echo "📊 文件大小统计:"
du -sh ${PACKAGE_DIR}
du -sh ${PACKAGE_DIR}/*

echo ""
echo "✅ 模型已打包到: ${PACKAGE_DIR}/"
echo ""
echo "📥 下载方法："
echo "  方法1 (scp):"
echo "    scp -r root@<server_ip>:${PACKAGE_DIR} ./"
echo ""
echo "  方法2 (rsync):"
echo "    rsync -avz root@<server_ip>:${PACKAGE_DIR} ./"
echo ""
echo "  方法3 (压缩后下载):"
echo "    tar -czf ${MODEL_NAME}.tar.gz ${PACKAGE_DIR}"
echo "    scp root@<server_ip>:${MODEL_NAME}.tar.gz ./"

