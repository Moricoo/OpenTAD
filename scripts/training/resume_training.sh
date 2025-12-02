#!/bin/bash
# 恢复训练脚本（如果需要）

WORK_DIR="/root/OpenTAD/exps/thumos/actionformer_i3d/gpu1_id0"
CHECKPOINT_DIR="$WORK_DIR/checkpoint"

echo "=========================================="
echo "检查训练状态"
echo "=========================================="

# 确保工作目录存在（通过符号链接解析）
REAL_WORK_DIR=$(readlink -f "$WORK_DIR" 2>/dev/null || echo "$WORK_DIR")
if [ ! -d "$REAL_WORK_DIR" ]; then
    mkdir -p "$REAL_WORK_DIR"
    echo "已创建工作目录: $REAL_WORK_DIR"
fi

# 检查最新检查点
LATEST_CHECKPOINT=$(ls -t "$CHECKPOINT_DIR"/*.pth 2>/dev/null | head -1)
if [ -n "$LATEST_CHECKPOINT" ]; then
    EPOCH=$(basename "$LATEST_CHECKPOINT" | grep -oP 'epoch_\K[0-9]+')
    echo "最新检查点: epoch_$EPOCH"
    echo "文件: $LATEST_CHECKPOINT"
    
    # 检查是否是符号链接
    if [ -L "$LATEST_CHECKPOINT" ]; then
        REAL_PATH=$(readlink -f "$LATEST_CHECKPOINT")
        echo "实际位置: $REAL_PATH"
    fi
else
    echo "未找到检查点文件"
fi

echo ""
echo "如果要恢复训练，使用："
echo "torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \\"
echo "  tools/train.py configs/actionformer/thumos_i3d.py \\"
echo "  --resume $LATEST_CHECKPOINT"

