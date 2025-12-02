#!/bin/bash
# 解压 THUMOS-14 视频文件

set -e

cd /root/OpenTAD

echo "=========================================="
echo "解压 THUMOS-14 视频文件"
echo "=========================================="
echo ""

# 检查数据盘上的视频文件
VIDEO_SOURCE="/data/thumos-14/videos"
TARGET_DIR="/data/thumos-14/videos_extracted"

if [ ! -d "$VIDEO_SOURCE" ]; then
    echo "✗ 未找到源目录: $VIDEO_SOURCE"
    exit 1
fi

# 查找压缩文件
COMPRESSED_FILE=$(find "$VIDEO_SOURCE" -maxdepth 1 -type f \( -name "*.tar" -o -name "*.tar.gz" -o -name "*.zip" -o -name "*.7z" \) | head -1)

if [ -z "$COMPRESSED_FILE" ]; then
    echo "检查目录内容..."
    ls -lh "$VIDEO_SOURCE"
    echo ""
    echo "未找到压缩文件。可能的情况："
    echo "1. 文件已解压"
    echo "2. 文件格式不常见"
    echo ""
    read -p "请输入压缩文件的完整路径，或按 Enter 跳过: " manual_file
    if [ -n "$manual_file" ] && [ -f "$manual_file" ]; then
        COMPRESSED_FILE="$manual_file"
    else
        echo "跳过解压步骤"
        exit 0
    fi
fi

echo "找到压缩文件: $COMPRESSED_FILE"
FILE_SIZE=$(du -h "$COMPRESSED_FILE" | cut -f1)
echo "文件大小: $FILE_SIZE"
echo ""

# 检查磁盘空间
AVAILABLE_SPACE=$(df -h /data | tail -1 | awk '{print $4}')
echo "可用空间: $AVAILABLE_SPACE"
echo ""

# 确定解压目录
if [ -d "$TARGET_DIR" ] && [ "$(ls -A $TARGET_DIR 2>/dev/null)" ]; then
    echo "⚠️  目标目录已存在且不为空: $TARGET_DIR"
    read -p "是否继续解压到此目录？(y/N): " continue_extract
    if [ "$continue_extract" != "y" ] && [ "$continue_extract" != "Y" ]; then
        echo "已取消"
        exit 0
    fi
else
    mkdir -p "$TARGET_DIR"
    echo "创建解压目录: $TARGET_DIR"
fi

echo ""
echo "开始解压..."
echo "这可能需要一些时间，请耐心等待..."
echo ""

# 根据文件类型选择解压命令
if [[ "$COMPRESSED_FILE" == *.tar.gz ]] || [[ "$COMPRESSED_FILE" == *.tgz ]]; then
    echo "使用 tar -xzf 解压..."
    cd "$TARGET_DIR"
    tar -xzf "$COMPRESSED_FILE"
elif [[ "$COMPRESSED_FILE" == *.tar ]]; then
    echo "使用 tar -xf 解压..."
    cd "$TARGET_DIR"
    tar -xf "$COMPRESSED_FILE"
elif [[ "$COMPRESSED_FILE" == *.zip ]]; then
    echo "使用 unzip 解压..."
    cd "$TARGET_DIR"
    unzip -q "$COMPRESSED_FILE"
elif [[ "$COMPRESSED_FILE" == *.7z ]]; then
    echo "使用 7z 解压..."
    if ! command -v 7z &> /dev/null; then
        echo "✗ 未安装 7z，请先安装: apt-get install p7zip-full"
        exit 1
    fi
    cd "$TARGET_DIR"
    7z x "$COMPRESSED_FILE"
else
    echo "✗ 不支持的文件格式"
    exit 1
fi

echo ""
echo "=========================================="
echo "解压完成"
echo "=========================================="
echo ""

# 统计解压的文件
VIDEO_COUNT=$(find "$TARGET_DIR" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mkv" \) 2>/dev/null | wc -l)
echo "解压的视频文件数: $VIDEO_COUNT"

if [ "$VIDEO_COUNT" -gt 0 ]; then
    echo ""
    echo "示例文件（前5个）:"
    find "$TARGET_DIR" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mkv" \) 2>/dev/null | head -5 | while read f; do
        size=$(du -h "$f" | cut -f1)
        echo "  - $(basename "$f") ($size)"
    done
    
    echo ""
    echo "=========================================="
    echo "设置符号链接"
    echo "=========================================="
    echo ""
    
    # 创建符号链接到 OpenTAD 目录
    OPEN_TAD_VIDEOS="/root/OpenTAD/data/thumos-14/videos"
    if [ ! -e "$OPEN_TAD_VIDEOS" ]; then
        echo "创建符号链接: $OPEN_TAD_VIDEOS -> $TARGET_DIR"
        mkdir -p "$(dirname "$OPEN_TAD_VIDEOS")"
        ln -sf "$TARGET_DIR" "$OPEN_TAD_VIDEOS"
        echo "✓ 符号链接已创建"
    else
        echo "⚠️  目标已存在: $OPEN_TAD_VIDEOS"
        echo "  请手动处理或删除后重新运行此脚本"
    fi
    
    # 运行设置脚本
    echo ""
    echo "运行视频数据设置脚本..."
    cd /root/OpenTAD
    bash scripts/utils/setup_video_data.sh
else
    echo "⚠️  警告: 未找到视频文件"
    echo "  请检查解压是否成功"
fi

echo ""
echo "=========================================="
echo "完成"
echo "=========================================="

