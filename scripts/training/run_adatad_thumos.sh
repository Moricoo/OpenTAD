#!/bin/bash
# AdaTAD THUMOS-14 训练脚本
# 从检查点恢复训练

set -e

# 激活conda环境
source /root/miniconda3/bin/activate opentad

# 进入OpenTAD目录
cd /root/OpenTAD

# 配置文件路径
CONFIG="configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py"

# 检查点目录
CHECKPOINT_DIR="exps/thumos/adatad/e2e_actionformer_videomae_s_768x1_160_adapter/gpu1_id0/checkpoint"

# 查找最新检查点
LATEST_CHECKPOINT=$(ls -t "$CHECKPOINT_DIR"/epoch_*.pth 2>/dev/null | head -1)

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "错误: 未找到检查点文件！"
    echo "检查点目录: $CHECKPOINT_DIR"
    echo "请先运行一次训练以生成检查点，或检查路径是否正确。"
    exit 1
fi

# 显示检查点信息
EPOCH=$(basename "$LATEST_CHECKPOINT" | grep -oP 'epoch_\K[0-9]+')
echo "=========================================="
echo "从检查点恢复训练"
echo "=========================================="
echo "配置文件: $CONFIG"
echo "检查点: $LATEST_CHECKPOINT"
echo "Epoch: $EPOCH"
echo "日志文件: /tmp/adatad_training.log"
echo "=========================================="

# 检查是否有正在运行的训练
if pgrep -f "train.py.*adatad.*thumos" > /dev/null; then
    echo "警告: 检测到正在运行的训练进程！"
    ps aux | grep "train.py.*adatad" | grep -v grep | head -3
    read -p "是否要停止现有训练并重新开始? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "停止现有训练..."
        pkill -f "train.py.*adatad.*thumos"
        sleep 2
    else
        echo "取消操作"
        exit 0
    fi
fi

# 启动训练
echo "启动训练..."
CUDA_VISIBLE_DEVICES=0 torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    tools/train.py \
    "$CONFIG" \
    --resume "$LATEST_CHECKPOINT" \
    > /tmp/adatad_training.log 2>&1 &

TRAIN_PID=$!
echo "训练已启动，PID: $TRAIN_PID"
echo ""
echo "监控命令:"
echo "  查看日志: tail -f /tmp/adatad_training.log"
echo "  查看进程: ps aux | grep train.py"
echo "  查看GPU:  watch -n 1 nvidia-smi"
echo "  停止训练: pkill -f 'train.py.*adatad.*thumos'"
echo ""
echo "等待5秒后显示初始日志..."
sleep 5
tail -20 /tmp/adatad_training.log

