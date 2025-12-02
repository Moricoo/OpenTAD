#!/bin/bash
# 设置视频数据路径，创建符号链接以匹配配置

set -e

cd /root/OpenTAD

echo "=========================================="
echo "设置 THUMOS-14 视频数据路径"
echo "=========================================="
echo ""

# 配置期望的路径
CONFIG_PATH="data/thumos-14/raw_data/video"
VIDEOS_DIR="data/thumos-14/videos"
RAW_DATA_DIR="data/thumos-14/raw_data"

# 检查 videos 目录是否存在
if [ ! -d "$VIDEOS_DIR" ] && [ ! -L "$VIDEOS_DIR" ]; then
    echo "⚠️  未找到: $VIDEOS_DIR"
    echo ""
    echo "请确保视频数据已上传到以下位置之一："
    echo "  1. $VIDEOS_DIR/"
    echo "  2. $RAW_DATA_DIR/video/"
    echo ""
    echo "视频文件格式: .mp4, .avi, .mkv"
    echo ""
    read -p "是否继续检查其他位置？(y/N): " continue_check
    if [ "$continue_check" != "y" ] && [ "$continue_check" != "Y" ]; then
        exit 1
    fi
fi

# 如果 videos 目录存在，创建符号链接
if [ -d "$VIDEOS_DIR" ] || [ -L "$VIDEOS_DIR" ]; then
    echo "✓ 找到视频目录: $VIDEOS_DIR"
    
    # 统计视频文件
    VIDEO_COUNT=$(find "$VIDEOS_DIR" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mkv" \) 2>/dev/null | wc -l)
    echo "  视频文件数: $VIDEO_COUNT"
    
    if [ "$VIDEO_COUNT" -eq 0 ]; then
        echo "⚠️  警告: 未找到视频文件"
        echo "  请检查视频文件是否已正确上传"
    fi
    
    # 创建 raw_data 目录（如果不存在）
    if [ ! -d "$RAW_DATA_DIR" ]; then
        echo "创建目录: $RAW_DATA_DIR"
        mkdir -p "$RAW_DATA_DIR"
    fi
    
    # 创建符号链接
    if [ ! -e "$CONFIG_PATH" ]; then
        echo "创建符号链接: $CONFIG_PATH -> $VIDEOS_DIR"
        ln -sf ../videos "$CONFIG_PATH"
        echo "✓ 符号链接已创建"
    elif [ -L "$CONFIG_PATH" ]; then
        CURRENT_LINK=$(readlink -f "$CONFIG_PATH")
        EXPECTED_LINK=$(readlink -f "$VIDEOS_DIR")
        if [ "$CURRENT_LINK" != "$EXPECTED_LINK" ]; then
            echo "更新符号链接: $CONFIG_PATH -> $VIDEOS_DIR"
            rm -f "$CONFIG_PATH"
            ln -sf ../videos "$CONFIG_PATH"
            echo "✓ 符号链接已更新"
        else
            echo "✓ 符号链接已正确设置"
        fi
    else
        echo "⚠️  $CONFIG_PATH 已存在但不是符号链接"
        echo "  请手动处理或删除后重新运行此脚本"
    fi
fi

# 如果 raw_data/video 直接存在
if [ -d "$CONFIG_PATH" ] && [ ! -L "$CONFIG_PATH" ]; then
    echo "✓ 找到视频目录: $CONFIG_PATH"
    VIDEO_COUNT=$(find "$CONFIG_PATH" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mkv" \) 2>/dev/null | wc -l)
    echo "  视频文件数: $VIDEO_COUNT"
    echo "✓ 配置路径已存在，无需创建符号链接"
fi

echo ""
echo "=========================================="
echo "验证配置"
echo "=========================================="

# 验证路径
if [ -e "$CONFIG_PATH" ]; then
    echo "✓ 配置路径存在: $CONFIG_PATH"
    if [ -L "$CONFIG_PATH" ]; then
        echo "  类型: 符号链接 -> $(readlink "$CONFIG_PATH")"
    else
        echo "  类型: 目录"
    fi
    
    VIDEO_COUNT=$(find "$CONFIG_PATH" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mkv" \) 2>/dev/null | wc -l)
    echo "  视频文件数: $VIDEO_COUNT"
    
    if [ "$VIDEO_COUNT" -gt 0 ]; then
        echo ""
        echo "示例视频文件（前3个）:"
        find "$CONFIG_PATH" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mkv" \) 2>/dev/null | head -3 | while read f; do
            size=$(du -h "$f" | cut -f1)
            echo "  - $(basename "$f") ($size)"
        done
    fi
else
    echo "✗ 配置路径不存在: $CONFIG_PATH"
    echo "  请确保视频数据已正确放置"
fi

echo ""
echo "=========================================="
echo "设置完成"
echo "=========================================="

