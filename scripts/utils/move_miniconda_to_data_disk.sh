#!/bin/bash
# 移动 Miniconda3 到数据盘
# 位置: scripts/utils/move_miniconda_to_data_disk.sh

set -e

DATA_DISK="/data"
MINICONDA_PATH="/root/miniconda3"
MINICONDA_DEST="$DATA_DISK/miniconda3"

echo "=========================================="
echo "移动 Miniconda3 到数据盘"
echo "=========================================="
echo ""

# 检查数据盘
if [ ! -d "$DATA_DISK" ]; then
    echo "✗ 错误: 数据盘 $DATA_DISK 不存在"
    exit 1
fi

# 检查 miniconda 是否存在
if [ ! -d "$MINICONDA_PATH" ]; then
    echo "✗ 错误: Miniconda3 不存在于 $MINICONDA_PATH"
    exit 1
fi

# 检查是否已经是符号链接
if [ -L "$MINICONDA_PATH" ]; then
    CURRENT_LINK=$(readlink -f "$MINICONDA_PATH")
    echo "⚠️  Miniconda3 已经是符号链接: $MINICONDA_PATH -> $CURRENT_LINK"
    if [ "$CURRENT_LINK" = "$(readlink -f $MINICONDA_DEST)" ]; then
        echo "✓ 已经指向数据盘，无需移动"
        exit 0
    else
        echo "  当前指向: $CURRENT_LINK"
        echo "  期望指向: $MINICONDA_DEST"
        read -p "是否更新符号链接? (y/N): " update_link
        if [ "$update_link" = "y" ] || [ "$update_link" = "Y" ]; then
            rm -f "$MINICONDA_PATH"
            ln -sf "$MINICONDA_DEST" "$MINICONDA_PATH"
            echo "✓ 符号链接已更新"
        fi
        exit 0
    fi
fi

# 检查目标目录是否已存在
if [ -d "$MINICONDA_DEST" ]; then
    echo "⚠️  目标目录已存在: $MINICONDA_DEST"
    read -p "是否覆盖? (y/N): " overwrite
    if [ "$overwrite" != "y" ] && [ "$overwrite" != "Y" ]; then
        echo "已取消"
        exit 0
    fi
    echo "  备份现有目录..."
    mv "$MINICONDA_DEST" "${MINICONDA_DEST}.backup.$(date +%Y%m%d_%H%M%S)"
fi

# 显示当前空间使用
echo "当前磁盘空间:"
echo "  根分区:"
df -h "/" | tail -1 | awk '{printf "    %s/%s (%s 使用, %s 可用)\n", $3, $2, $5, $4}'
echo "  数据盘:"
df -h "$DATA_DISK" | tail -1 | awk '{printf "    %s/%s (%s 使用, %s 可用)\n", $3, $2, $5, $4}'
echo ""

# 显示 miniconda 大小
MINICONDA_SIZE=$(du -sh "$MINICONDA_PATH" 2>/dev/null | cut -f1)
echo "Miniconda3 大小: $MINICONDA_SIZE"
echo "  源位置: $MINICONDA_PATH"
echo "  目标位置: $MINICONDA_DEST"
echo ""

# 确认移动（非交互式模式，自动确认）
if [ -t 0 ]; then
    # 交互式模式
    read -p "确认移动 Miniconda3 到数据盘? (y/N): " confirm
    if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
        echo "已取消"
        exit 0
    fi
else
    # 非交互式模式，自动确认
    echo "非交互式模式，自动确认移动..."
    confirm="y"
fi

echo ""
echo "开始移动 Miniconda3..."
echo "这可能需要一些时间，请耐心等待..."
echo ""

# 尝试直接移动（同一文件系统）
if mv "$MINICONDA_PATH" "$MINICONDA_DEST" 2>/dev/null; then
    echo "✓ 移动完成（直接移动）"
else
    # 如果移动失败（跨文件系统），使用 rsync
    echo "使用 rsync 复制（跨文件系统）..."
    echo "这可能需要更长时间..."

    # 创建目标目录
    mkdir -p "$MINICONDA_DEST"

    # 使用 rsync 复制
    rsync -av --progress "$MINICONDA_PATH/" "$MINICONDA_DEST/" || {
        echo "✗ 复制失败"
        exit 1
    }

    # 删除源目录
    echo "删除源目录..."
    rm -rf "$MINICONDA_PATH"
    echo "✓ 复制完成（rsync）"
fi

# 创建符号链接
echo ""
echo "创建符号链接..."
ln -sf "$MINICONDA_DEST" "$MINICONDA_PATH"
echo "✓ 符号链接已创建: $MINICONDA_PATH -> $MINICONDA_DEST"

# 验证
echo ""
echo "验证..."
if [ -L "$MINICONDA_PATH" ] && [ -d "$MINICONDA_DEST" ]; then
    ACTUAL_LINK=$(readlink -f "$MINICONDA_PATH")
    if [ "$ACTUAL_LINK" = "$(readlink -f $MINICONDA_DEST)" ]; then
        echo "✓ 验证成功"
    else
        echo "⚠️  警告: 符号链接指向可能不正确"
        echo "  实际: $ACTUAL_LINK"
        echo "  期望: $MINICONDA_DEST"
    fi
else
    echo "✗ 验证失败"
    exit 1
fi

# 显示最终空间使用
echo ""
echo "=========================================="
echo "最终空间使用情况"
echo "=========================================="
echo "根分区:"
df -h "/" | tail -1 | awk '{printf "  %s/%s (%s 使用, %s 可用)\n", $3, $2, $5, $4}'
echo ""
echo "数据盘:"
df -h "$DATA_DISK" | tail -1 | awk '{printf "  %s/%s (%s 使用, %s 可用)\n", $3, $2, $5, $4}'
echo ""

echo "=========================================="
echo "Miniconda3 移动完成！"
echo "=========================================="
echo ""
echo "✓ Miniconda3 已移动到: $MINICONDA_DEST"
echo "✓ 符号链接已创建: $MINICONDA_PATH -> $MINICONDA_DEST"
echo ""
echo "⚠️  重要提示:"
echo "  1. 如果当前有激活的 conda 环境，请重新激活:"
echo "     source $MINICONDA_PATH/bin/activate opentad"
echo ""
echo "  2. 如果 PATH 环境变量中硬编码了路径，可能需要更新"
echo ""
echo "  3. 建议重新登录或重新打开终端以确保环境变量正确"
echo ""

