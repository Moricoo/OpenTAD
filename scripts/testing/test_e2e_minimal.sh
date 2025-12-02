#!/bin/bash
# 测试最小的端到端模型

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

# 获取检查点路径（如果提供）
CHECKPOINT="${1:-exps/thumos/e2e_thumos_videomae_s_768x1_160_adapter/gpu2_id0/checkpoint/best.pth}"

# 检查检查点是否存在
REAL_CHECKPOINT=$(readlink -f "$CHECKPOINT" 2>/dev/null || echo "$CHECKPOINT")
if [ ! -f "$REAL_CHECKPOINT" ]; then
    echo "错误: 检查点文件不存在: $CHECKPOINT"
    echo "请先训练模型或指定正确的检查点路径"
    exit 1
fi

echo "=========================================="
echo "测试最小的端到端模型"
echo "=========================================="
echo "模型: AdaTAD with VideoMAE-S (Adapter)"
echo "检查点: $CHECKPOINT"
echo "实际路径: $REAL_CHECKPOINT"
echo ""

# 运行测试
echo "开始测试..."
torchrun \
    --nnodes=1 \
    --nproc_per_node=2 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    tools/test.py \
    configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py \
    --checkpoint "$CHECKPOINT"

echo ""
echo "=========================================="
echo "测试完成！"
echo "=========================================="


