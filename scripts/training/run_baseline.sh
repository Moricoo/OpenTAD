#!/bin/bash
# OpenTAD Baseline 训练脚本
# 使用 ActionFormer 在 THUMOS-14 数据集上训练

set -e

# 激活 conda 环境
source /root/miniconda3/bin/activate opentad

# 进入 OpenTAD 目录
cd /root/OpenTAD

# 确保 exps 符号链接正确
EXPS_LINK="/root/OpenTAD/exps"
EXPS_TARGET="/data/opentad-exps/exps"
if [ ! -L "$EXPS_LINK" ] || [ "$(readlink "$EXPS_LINK")" != "$EXPS_TARGET" ]; then
    echo "修复 exps 符号链接..."
    rm -rf "$EXPS_LINK"
    mkdir -p "$EXPS_TARGET"
    ln -sf "$EXPS_TARGET" "$EXPS_LINK"
    echo "✓ 符号链接已修复"
fi

# 检查数据是否存在
if [ ! -d "data/thumos-14/features/i3d_actionformer_stride4_thumos" ]; then
    echo "错误: 数据目录不存在！"
    echo "请先准备 THUMOS-14 数据集："
    echo "1. 下载注释文件: bash tools/prepare_data/thumos/download_annotation.sh"
    echo "2. 下载特征文件: 参考 tools/prepare_data/thumos/README.md"
    exit 1
fi

# 运行训练
# 使用 1 个 GPU 训练（推荐用于 feature-based TAD）
echo "开始训练 ActionFormer baseline..."
torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    tools/train.py \
    configs/actionformer/thumos_i3d.py

echo "训练完成！"

