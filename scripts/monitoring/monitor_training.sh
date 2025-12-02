#!/bin/bash
# 训练监控脚本

WORK_DIR="/root/OpenTAD/exps/thumos/actionformer_i3d/gpu1_id0"
# 通过符号链接解析实际路径
REAL_WORK_DIR=$(readlink -f "$WORK_DIR" 2>/dev/null || echo "$WORK_DIR")
LOG_FILE="$REAL_WORK_DIR/log.json"

echo "=========================================="
echo "OpenTAD 训练监控"
echo "=========================================="
echo ""

# 检查训练进程
if pgrep -f "train.py" > /dev/null; then
    echo "✓ 训练进程: 运行中"
else
    echo "✗ 训练进程: 已停止"
fi

# GPU 状态
echo ""
echo "GPU 状态:"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | awk -F', ' '{printf "  使用率: %s%%, 显存: %s/%s MB, 温度: %s°C\n", $1, $2, $3, $4}'

# 训练进度
echo ""
echo "训练进度:"
if [ -f "$LOG_FILE" ]; then
    # 获取最新的训练记录
    LATEST_LOG=$(tail -1 "$LOG_FILE" 2>/dev/null)
    if [ -n "$LATEST_LOG" ]; then
        # 提取关键信息（简单解析）
        EPOCH=$(echo "$LATEST_LOG" | grep -oP '\[Train\]: \[\K[0-9]+' | head -1)
        ITER=$(echo "$LATEST_LOG" | grep -oP '\[Train\]: \[[0-9]+\]\[[0-9]+/[0-9]+\]' | grep -oP '\[[0-9]+\]/\[0-9]+\]' | grep -oP '[0-9]+' | head -1)
        TOTAL_ITER=$(echo "$LATEST_LOG" | grep -oP '\[Train\]: \[[0-9]+\]\[[0-9]+/[0-9]+\]' | grep -oP '/[0-9]+\]' | grep -oP '[0-9]+' | head -1)
        LOSS=$(echo "$LATEST_LOG" | grep -oP 'Loss=\K[0-9.]+' | head -1)
        
        if [ -n "$EPOCH" ] && [ -n "$ITER" ]; then
            echo "  当前 Epoch: $EPOCH"
            echo "  当前迭代: $ITER/$TOTAL_ITER"
            if [ -n "$LOSS" ]; then
                echo "  当前 Loss: $LOSS"
            fi
        fi
    fi
    
    # 显示最后几行日志
    echo ""
    echo "最新训练日志:"
    tail -5 "$LOG_FILE" 2>/dev/null | sed 's/^/  /'
else
    echo "  (日志文件尚未生成)"
fi

# 检查点文件
echo ""
echo "检查点文件:"
CHECKPOINT_DIR="$REAL_WORK_DIR/checkpoint"
if [ -d "$CHECKPOINT_DIR" ]; then
    NUM_CHECKPOINTS=$(ls -1 "$CHECKPOINT_DIR"/*.pth 2>/dev/null | wc -l)
    echo "  已保存检查点: $NUM_CHECKPOINTS 个"
    LATEST_CHECKPOINT=$(ls -t "$CHECKPOINT_DIR"/*.pth 2>/dev/null | head -1)
    if [ -n "$LATEST_CHECKPOINT" ]; then
        SIZE=$(du -h "$LATEST_CHECKPOINT" | cut -f1)
        MTIME=$(stat -c %y "$LATEST_CHECKPOINT" | cut -d'.' -f1)
        echo "  最新检查点: $(basename $LATEST_CHECKPOINT) ($SIZE, $MTIME)"
    fi
else
    echo "  (检查点目录尚未创建)"
fi

# 磁盘空间
echo ""
echo "磁盘空间:"
df -h / | tail -1 | awk '{printf "  根分区: %s/%s (%s 使用, %s 可用)\n", $3, $2, $5, $4}'
df -h /data | tail -1 | awk '{printf "  数据盘: %s/%s (%s 使用, %s 可用)\n", $3, $2, $5, $4}'

echo ""
echo "=========================================="
echo "提示: 使用 'tail -f $LOG_FILE' 实时查看训练日志"
echo "=========================================="

