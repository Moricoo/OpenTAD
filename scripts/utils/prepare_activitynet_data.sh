#!/bin/bash
# ActivityNet-1.3 数据准备脚本
# 位置: scripts/utils/prepare_activitynet_data.sh

set -e

cd /root/OpenTAD

DATA_DIR="data/activitynet-1.3"
ANNOTATION_DIR="$DATA_DIR/annotations"
RAW_DATA_DIR="$DATA_DIR/raw_data"

echo "=========================================="
echo "ActivityNet-1.3 数据准备"
echo "=========================================="
echo ""

# 1. 下载注释文件
echo "[1/2] 下载注释文件"
echo "=========================================="

if [ ! -f "$ANNOTATION_DIR/activity_net.v1-3.min.json" ] && [ ! -f "$ANNOTATION_DIR/anet_anno_action.json" ]; then
    echo "注释文件不存在，开始下载..."
    echo ""
    echo "方法 1: 使用 gdown 下载（推荐）"

    if command -v gdown &> /dev/null; then
        echo "使用 gdown 下载..."
        mkdir -p "$ANNOTATION_DIR"
        cd "$ANNOTATION_DIR"

        # 尝试下载单个文件
        echo "尝试下载注释文件..."
        gdown "https://drive.google.com/uc?id=1HpTc6FbYnm-s9tY4aZljjZnYnThICcNq" -O activitynet_annotations.zip 2>/dev/null || {
            echo "⚠️  直接下载失败，请手动下载"
            echo ""
            echo "请访问以下链接下载注释文件："
            echo "https://drive.google.com/drive/folders/1HpTc6FbYnm-s9tY4aZljjZnYnThICcNq"
            echo ""
            echo "下载后解压到: $ANNOTATION_DIR"
            echo ""
            read -p "是否已手动下载并解压？(y/N): " manual_download
            if [ "$manual_download" != "y" ] && [ "$manual_download" != "Y" ]; then
                echo "请先下载注释文件"
                exit 1
            fi
        }

        if [ -f "activitynet_annotations.zip" ]; then
            echo "解压注释文件..."
            unzip -q activitynet_annotations.zip -d . 2>/dev/null || true
            rm -f activitynet_annotations.zip
        fi

        cd /root/OpenTAD
    else
        echo "⚠️  gdown 未安装"
        echo "安装 gdown: pip install gdown"
        echo ""
        echo "或者手动下载注释文件："
        echo "https://drive.google.com/drive/folders/1HpTc6FbYnm-s9tY4aZljjZnYnThICcNq"
        echo ""
        read -p "是否已手动下载？(y/N): " manual_download
        if [ "$manual_download" != "y" ] && [ "$manual_download" != "Y" ]; then
            echo "请先下载注释文件"
            exit 1
        fi
    fi
else
    echo "✓ 注释文件已存在"
fi

# 验证注释文件
if [ -f "$ANNOTATION_DIR/activity_net.v1-3.min.json" ] || [ -f "$ANNOTATION_DIR/anet_anno_action.json" ]; then
    echo "✓ 注释文件验证成功"
    ls -lh "$ANNOTATION_DIR"/*.json 2>/dev/null | head -5
else
    echo "⚠️  警告: 未找到标准注释文件"
    echo "请确保以下文件之一存在："
    echo "  - activity_net.v1-3.min.json"
    echo "  - anet_anno_action.json"
fi
echo ""

# 2. 准备原始视频数据
echo "[2/2] 准备原始视频数据"
echo "=========================================="
echo ""
echo "ActivityNet 原始视频需要从官方网站下载："
echo "  URL: https://docs.google.com/forms/d/e/1FAIpQLSdxhNVeeSCwB2USAfeNWCaI9saVT6i2hpiiizVYfa3MsTyamg/viewform"
echo ""
echo "或者使用处理后的版本（推荐）："
echo "  - Anet_videos_15fps_short256.zip"
echo "  - 已转换为 15fps，短边调整为 256 像素"
echo ""

# 检查视频目录
if [ ! -d "$RAW_DATA_DIR/video" ] || [ -z "$(ls -A $RAW_DATA_DIR/video 2>/dev/null)" ]; then
    echo "⚠️  视频目录为空或不存在"
    echo ""
    echo "请将视频文件放在以下位置之一："
    echo "  1. $RAW_DATA_DIR/video/  (推荐)"
    echo "  2. /data/opentad/data/activitynet-1.3/raw_data/video/"
    echo ""
    echo "视频文件命名格式："
    echo "  - 前缀: v_ (如 v_xxx.mp4)"
    echo "  - 格式: .mp4, .avi, .mkv"
    echo ""

    # 检查数据盘是否有视频
    DATA_DISK_VIDEOS="/data/opentad/data/activitynet-1.3/raw_data/video"
    if [ -d "$DATA_DISK_VIDEOS" ] && [ "$(ls -A $DATA_DISK_VIDEOS 2>/dev/null)" ]; then
        VIDEO_COUNT=$(find "$DATA_DISK_VIDEOS" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mkv" \) 2>/dev/null | wc -l)
        echo "✓ 在数据盘找到 $VIDEO_COUNT 个视频文件"

        # 创建符号链接
        if [ ! -e "$RAW_DATA_DIR/video" ]; then
            echo "创建符号链接..."
            mkdir -p "$RAW_DATA_DIR"
            ln -sf "$DATA_DISK_VIDEOS" "$RAW_DATA_DIR/video"
            echo "✓ 符号链接已创建"
        fi
    else
        echo "请上传视频文件后重新运行此脚本"
    fi
else
    VIDEO_COUNT=$(find "$RAW_DATA_DIR/video" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mkv" \) 2>/dev/null | wc -l)
    echo "✓ 找到 $VIDEO_COUNT 个视频文件"
fi
echo ""

# 显示数据目录结构
echo "=========================================="
echo "数据目录结构"
echo "=========================================="
echo ""
echo "ActivityNet 数据目录:"
tree -L 2 "$DATA_DIR" 2>/dev/null || {
    echo "$DATA_DIR/"
    ls -lh "$DATA_DIR" 2>/dev/null | head -10
    echo ""
    if [ -d "$ANNOTATION_DIR" ]; then
        echo "  annotations/"
        ls -lh "$ANNOTATION_DIR" 2>/dev/null | head -5
    fi
    echo ""
    if [ -d "$RAW_DATA_DIR" ]; then
        echo "  raw_data/"
        ls -lh "$RAW_DATA_DIR" 2>/dev/null | head -5
    fi
}
echo ""

# 显示数据盘使用情况
echo "=========================================="
echo "数据存储位置"
echo "=========================================="
echo ""
echo "实际存储位置: $(readlink -f $DATA_DIR)"
echo "数据盘空间:"
df -h /data | tail -1 | awk '{printf "  %s/%s (%s 使用, %s 可用)\n", $3, $2, $5, $4}'
echo ""

# 验证数据完整性
echo "=========================================="
echo "数据验证"
echo "=========================================="
echo ""

check_count=0
total_checks=2

# 检查注释文件
if [ -f "$ANNOTATION_DIR/activity_net.v1-3.min.json" ] || [ -f "$ANNOTATION_DIR/anet_anno_action.json" ]; then
    echo "✓ [1/2] 注释文件存在"
    check_count=$((check_count + 1))
else
    echo "✗ [1/2] 注释文件缺失"
fi

# 检查视频文件
if [ -d "$RAW_DATA_DIR/video" ] && [ "$(ls -A $RAW_DATA_DIR/video 2>/dev/null)" ]; then
    VIDEO_COUNT=$(find "$RAW_DATA_DIR/video" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mkv" \) 2>/dev/null | wc -l)
    if [ "$VIDEO_COUNT" -gt 0 ]; then
        echo "✓ [2/2] 视频文件存在 ($VIDEO_COUNT 个)"
        check_count=$((check_count + 1))
    else
        echo "✗ [2/2] 视频文件目录为空"
    fi
else
    echo "✗ [2/2] 视频文件目录不存在或为空"
fi

echo ""
if [ "$check_count" -eq "$total_checks" ]; then
    echo "=========================================="
    echo "✓ 数据准备完成！"
    echo "=========================================="
    echo ""
    echo "现在可以开始训练 ActivityNet 模型："
    echo ""
    echo "  # 训练 VideoMAE-S 小模型"
    echo "  torchrun --nnodes=1 --nproc_per_node=4 \\"
    echo "      tools/train.py \\"
    echo "      configs/adatad/anet/e2e_anet_videomae_s_192x4_160_adapter.py"
    echo ""
else
    echo "=========================================="
    echo "⚠️  数据准备未完成"
    echo "=========================================="
    echo ""
    echo "请完成以下步骤："
    if [ ! -f "$ANNOTATION_DIR/activity_net.v1-3.min.json" ] && [ ! -f "$ANNOTATION_DIR/anet_anno_action.json" ]; then
        echo "  1. 下载注释文件到: $ANNOTATION_DIR"
    fi
    if [ ! -d "$RAW_DATA_DIR/video" ] || [ -z "$(ls -A $RAW_DATA_DIR/video 2>/dev/null)" ]; then
        echo "  2. 下载视频文件到: $RAW_DATA_DIR/video"
    fi
    echo ""
fi

