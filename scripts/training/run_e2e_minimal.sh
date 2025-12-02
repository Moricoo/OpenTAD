#!/bin/bash
# 运行最小的端到端模型 (AdaTAD with VideoMAE-S)
# 使用 adapter 模式，只训练 adapter 参数，内存占用最小

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

# 检查并设置视频数据路径
VIDEO_CONFIG_PATH="data/thumos-14/raw_data/video"
if [ ! -e "$VIDEO_CONFIG_PATH" ]; then
    echo "=========================================="
    echo "⚠️  警告: 未找到原始视频数据"
    echo "=========================================="
    echo ""
    echo "端到端训练需要原始视频文件，而不是预提取的特征。"
    echo ""
    echo "请准备 THUMOS-14 原始视频："
    echo "1. 下载 THUMOS-14 视频文件"
    echo "2. 将视频放在: data/thumos-14/videos/"
    echo ""
    echo "然后运行设置脚本:"
    echo "  bash scripts/utils/setup_video_data.sh"
    echo ""
    echo "参考文档: tools/prepare_data/thumos/README.md"
    echo ""
    
    # 尝试自动设置（如果 videos 目录存在）
    if [ -d "data/thumos-14/videos" ] || [ -L "data/thumos-14/videos" ]; then
        echo "检测到 videos 目录，尝试自动设置..."
        bash scripts/utils/setup_video_data.sh
        if [ ! -e "$VIDEO_CONFIG_PATH" ]; then
            echo "自动设置失败，请手动运行: bash scripts/utils/setup_video_data.sh"
            exit 1
        fi
    else
        read -p "是否继续（不检查视频数据）？(y/N): " continue_anyway
        if [ "$continue_anyway" != "y" ] && [ "$continue_anyway" != "Y" ]; then
            echo "已取消"
            exit 1
        fi
    fi
fi

# 验证视频文件数量
VIDEO_COUNT=$(find "$VIDEO_CONFIG_PATH" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mkv" \) 2>/dev/null | wc -l)
if [ "$VIDEO_COUNT" -eq 0 ]; then
    echo "⚠️  警告: 未找到视频文件"
    echo "  请检查视频数据是否正确放置"
    read -p "是否继续？(y/N): " continue_anyway
    if [ "$continue_anyway" != "y" ] && [ "$continue_anyway" != "Y" ]; then
        echo "已取消"
        exit 1
    fi
else
    echo "✓ 找到 $VIDEO_COUNT 个视频文件"
fi

# 检查预训练模型
PRETRAINED_DIR="pretrained"
PRETRAINED_MODEL="vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-860a3cd3.pth"
if [ ! -f "$PRETRAINED_DIR/$PRETRAINED_MODEL" ]; then
    echo "=========================================="
    echo "⚠️  警告: 未找到预训练模型"
    echo "=========================================="
    echo ""
    echo "需要下载 VideoMAE-S 预训练模型："
    echo "  $PRETRAINED_MODEL"
    echo ""
    echo "推荐使用自动下载脚本:"
    echo "  bash scripts/utils/download_videomae_s.sh"
    echo ""
    echo "或手动下载:"
    echo "  Google Drive: https://drive.google.com/file/d/1BH5BZmdImaZesUfqtW23eBGC341Gui1D/view?usp=sharing"
    echo ""
    echo "下载后请放在: $PRETRAINED_DIR/"
    echo ""
    read -p "是否尝试自动下载？(Y/n): " auto_download
    if [ "$auto_download" != "n" ] && [ "$auto_download" != "N" ]; then
        echo "尝试自动下载..."
        bash scripts/utils/download_videomae_s.sh || {
            echo ""
            echo "自动下载失败，请手动下载后重试"
            exit 1
        }
    else
        read -p "是否继续（不检查预训练模型）？(y/N): " continue_anyway
        if [ "$continue_anyway" != "y" ] && [ "$continue_anyway" != "Y" ]; then
            echo "已取消"
            exit 1
        fi
    fi
fi

echo "=========================================="
echo "开始训练最小的端到端模型"
echo "=========================================="
echo "模型: AdaTAD with VideoMAE-S (Adapter)"
echo "配置: configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py"
echo "模式: Adapter (只训练 adapter 参数，内存占用最小)"
echo ""
echo "预期性能 (参考论文):"
echo "  mAP@0.3: ~83.90%"
echo "  mAP@0.4: ~79.01%"
echo "  mAP@0.5: ~72.38%"
echo "  mAP@0.6: ~61.57%"
echo "  mAP@0.7: ~48.27%"
echo "  平均 mAP: ~69.03%"
echo ""
echo "资源需求:"
echo "  - GPU: 2 个 GPU (推荐)"
echo "  - 显存: 每个 GPU 约 10-12 GB"
echo "  - 训练时间: 约 10-15 小时 (取决于硬件)"
echo ""

# 运行训练
# 使用 2 个 GPU（根据 README 推荐）
echo "开始训练..."
torchrun \
    --nnodes=1 \
    --nproc_per_node=2 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    tools/train.py \
    configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py

echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="
echo ""
echo "结果保存在: exps/thumos/e2e_thumos_videomae_s_768x1_160_adapter/"
echo ""
echo "测试模型:"
echo "  bash scripts/testing/test_e2e_minimal.sh"

