#!/bin/bash
# AdaTAD 训练监控脚本
# 位置: scripts/monitoring/monitor_adatad_training.sh

set -e

cd /root/OpenTAD

WORK_DIR="exps/thumos/adatad/e2e_actionformer_videomae_s_768x1_160_adapter/gpu1_id0"
LOG_FILE="/tmp/adatad_training.log"

echo "=========================================="
echo "AdaTAD 训练监控"
echo "=========================================="
echo ""

# 检查训练进程
echo "[1] 训练进程状态"
echo "----------------------------------------"
if pgrep -f "train.py.*adatad.*thumos" > /dev/null; then
    echo "✓ 训练进程正在运行"
    ps aux | grep "train.py" | grep -v grep | head -2
else
    echo "✗ 训练进程未运行"
fi
echo ""

# GPU 状态
echo "[2] GPU 状态"
echo "----------------------------------------"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader | \
    awk -F', ' '{printf "GPU %s: %s\n  使用率: %s | 显存: %s/%s | 温度: %s°C\n", $1, $2, $3, $4, $5, $6}'
echo ""

# 训练日志
echo "[3] 最新训练日志"
echo "----------------------------------------"
if [ -f "$LOG_FILE" ]; then
    tail -20 "$LOG_FILE" | grep -E "Epoch|loss|mAP|INFO.*Train|INFO.*Val" | tail -10
else
    echo "日志文件不存在: $LOG_FILE"
fi
echo ""

# 训练进度
echo "[4] 训练进度"
echo "----------------------------------------"
if [ -f "$WORK_DIR/log.json" ]; then
    echo "训练日志文件: $WORK_DIR/log.json"
    echo ""
    echo "最新训练记录:"
    tail -30 "$WORK_DIR/log.json" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if isinstance(data, list):
        for item in data[-5:]:
            if 'mode' in item:
                mode = item['mode']
                epoch = item.get('epoch', 'N/A')
                iter = item.get('iter', 'N/A')
                loss = item.get('loss', {})
                if isinstance(loss, dict):
                    loss_str = ', '.join([f'{k}: {v:.4f}' for k, v in loss.items()])
                else:
                    loss_str = f'{loss:.4f}'
                print(f'  [{mode}] Epoch {epoch}, Iter {iter}, Loss: {loss_str}')
except:
    print('  解析日志中...')
" 2>/dev/null || echo "  日志格式解析中..."
else
    echo "训练日志尚未生成"
fi
echo ""

# 检查点
echo "[5] 检查点状态"
echo "----------------------------------------"
if [ -d "$WORK_DIR/checkpoint" ]; then
    CHECKPOINTS=$(ls -1t "$WORK_DIR/checkpoint"/*.pth 2>/dev/null | head -5)
    if [ -n "$CHECKPOINTS" ]; then
        echo "最新检查点:"
        ls -lht "$WORK_DIR/checkpoint"/*.pth 2>/dev/null | head -3 | awk '{printf "  %s (%s)\n", $9, $5}'
    else
        echo "尚未生成检查点"
    fi
else
    echo "检查点目录不存在"
fi
echo ""

# 磁盘空间
echo "[6] 磁盘空间"
echo "----------------------------------------"
echo "根分区:"
df -h / | tail -1 | awk '{printf "  %s/%s (%s 使用, %s 可用)\n", $3, $2, $5, $4}'
echo "数据盘:"
df -h /data | tail -1 | awk '{printf "  %s/%s (%s 使用, %s 可用)\n", $3, $2, $5, $4}'
echo ""

# 训练输出目录大小
if [ -d "$WORK_DIR" ]; then
    SIZE=$(du -sh "$WORK_DIR" 2>/dev/null | cut -f1)
    echo "训练输出目录大小: $SIZE"
fi
echo ""

echo "=========================================="
echo "监控完成"
echo "=========================================="
echo ""
echo "查看完整日志: tail -f $LOG_FILE"
echo "查看训练日志: tail -f $WORK_DIR/log.json"
echo ""

