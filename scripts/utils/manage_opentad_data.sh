#!/bin/bash
# 完整的数据管理脚本 - 将所有数据相关目录链接到数据盘
# 目标：根分区只放代码，所有数据都在数据盘
# 位置: scripts/utils/manage_opentad_data.sh

set -e

DATA_DISK="/data"
ROOT_PARTITION="/"
OPENTAD_ROOT="/root/OpenTAD"

echo "=========================================="
echo "OpenTAD 数据管理 - 链接所有数据到数据盘"
echo "=========================================="
echo ""

# 检查数据盘
if [ ! -d "$DATA_DISK" ]; then
    echo "✗ 错误: 数据盘 $DATA_DISK 不存在"
    exit 1
fi

# 显示当前空间使用
echo "当前磁盘空间:"
echo "  根分区:"
df -h "$ROOT_PARTITION" | tail -1 | awk '{printf "    %s/%s (%s 使用, %s 可用)\n", $3, $2, $5, $4}'
echo "  数据盘:"
df -h "$DATA_DISK" | tail -1 | awk '{printf "    %s/%s (%s 使用, %s 可用)\n", $3, $2, $5, $4}'
echo ""

# 创建数据盘目录结构
echo "创建数据盘目录结构..."
mkdir -p "$DATA_DISK/opentad/data"
mkdir -p "$DATA_DISK/opentad/pretrained"
mkdir -p "$DATA_DISK/opentad-exps/exps"
echo "✓ 目录结构已创建"
echo ""

# 1. 管理 data 目录
echo "=========================================="
echo "[1/3] 管理数据目录 (data/)"
echo "=========================================="
OPENTAD_DATA="$OPENTAD_ROOT/data"
DATA_DEST="$DATA_DISK/opentad/data"

if [ -d "$OPENTAD_DATA" ] && [ ! -L "$OPENTAD_DATA" ]; then
    DATA_SIZE=$(du -sh "$OPENTAD_DATA" 2>/dev/null | cut -f1)
    echo "找到数据目录: $OPENTAD_DATA ($DATA_SIZE)"

    # 检查目标目录是否已有内容
    if [ -d "$DATA_DEST" ] && [ "$(ls -A $DATA_DEST 2>/dev/null)" ]; then
        echo "⚠️  目标目录已存在内容: $DATA_DEST"
        echo "  将合并现有内容..."
    fi

    echo "  移动数据目录到: $DATA_DEST"
    # 如果目标目录存在，先移动内容
    if [ -d "$DATA_DEST" ] && [ "$(ls -A $DATA_DEST 2>/dev/null)" ]; then
        echo "  合并现有内容..."
        rsync -av "$OPENTAD_DATA/" "$DATA_DEST/" 2>/dev/null || true
        rm -rf "$OPENTAD_DATA"
    else
        mv "$OPENTAD_DATA" "$DATA_DEST" 2>/dev/null || {
            # 如果移动失败（可能是跨文件系统），使用 rsync
            echo "  使用 rsync 复制（跨文件系统）..."
            rsync -av "$OPENTAD_DATA/" "$DATA_DEST/" 2>/dev/null || true
            rm -rf "$OPENTAD_DATA"
        }
    fi

    # 创建符号链接
    ln -sf "$DATA_DEST" "$OPENTAD_DATA"
    echo "  ✓ 数据目录已移动到数据盘并创建符号链接"
elif [ -L "$OPENTAD_DATA" ]; then
    CURRENT_LINK=$(readlink -f "$OPENTAD_DATA")
    if [ "$CURRENT_LINK" != "$(readlink -f $DATA_DEST)" ]; then
        echo "⚠️  符号链接指向: $CURRENT_LINK"
        echo "  更新为: $DATA_DEST"
        rm -f "$OPENTAD_DATA"
        ln -sf "$DATA_DEST" "$OPENTAD_DATA"
        echo "  ✓ 符号链接已更新"
    else
        echo "✓ 数据目录已正确链接到数据盘"
    fi
else
    echo "⚠️  数据目录不存在，创建符号链接..."
    ln -sf "$DATA_DEST" "$OPENTAD_DATA"
    echo "  ✓ 符号链接已创建（目录将在首次使用时创建）"
fi
echo ""

# 2. 管理 pretrained 目录
echo "=========================================="
echo "[2/3] 管理预训练模型目录 (pretrained/)"
echo "=========================================="
OPENTAD_PRETRAINED="$OPENTAD_ROOT/pretrained"
PRETRAINED_DEST="$DATA_DISK/opentad/pretrained"

if [ -d "$OPENTAD_PRETRAINED" ] && [ ! -L "$OPENTAD_PRETRAINED" ]; then
    PRETRAINED_SIZE=$(du -sh "$OPENTAD_PRETRAINED" 2>/dev/null | cut -f1)
    echo "找到预训练模型目录: $OPENTAD_PRETRAINED ($PRETRAINED_SIZE)"

    echo "  移动预训练模型到: $PRETRAINED_DEST"
    if [ -d "$PRETRAINED_DEST" ] && [ "$(ls -A $PRETRAINED_DEST 2>/dev/null)" ]; then
        echo "  合并现有内容..."
        rsync -av "$OPENTAD_PRETRAINED/" "$PRETRAINED_DEST/" 2>/dev/null || true
        rm -rf "$OPENTAD_PRETRAINED"
    else
        mv "$OPENTAD_PRETRAINED" "$PRETRAINED_DEST" 2>/dev/null || {
            echo "  使用 rsync 复制（跨文件系统）..."
            rsync -av "$OPENTAD_PRETRAINED/" "$PRETRAINED_DEST/" 2>/dev/null || true
            rm -rf "$OPENTAD_PRETRAINED"
        }
    fi

    ln -sf "$PRETRAINED_DEST" "$OPENTAD_PRETRAINED"
    echo "  ✓ 预训练模型目录已移动到数据盘并创建符号链接"
elif [ -L "$OPENTAD_PRETRAINED" ]; then
    CURRENT_LINK=$(readlink -f "$OPENTAD_PRETRAINED")
    if [ "$CURRENT_LINK" != "$(readlink -f $PRETRAINED_DEST)" ]; then
        echo "⚠️  符号链接指向: $CURRENT_LINK"
        echo "  更新为: $PRETRAINED_DEST"
        rm -f "$OPENTAD_PRETRAINED"
        ln -sf "$PRETRAINED_DEST" "$OPENTAD_PRETRAINED"
        echo "  ✓ 符号链接已更新"
    else
        echo "✓ 预训练模型目录已正确链接到数据盘"
    fi
else
    echo "⚠️  预训练模型目录不存在，创建符号链接..."
    ln -sf "$PRETRAINED_DEST" "$OPENTAD_PRETRAINED"
    echo "  ✓ 符号链接已创建"
fi
echo ""

# 3. 管理 exps 目录（训练结果）
echo "=========================================="
echo "[3/3] 管理训练结果目录 (exps/)"
echo "=========================================="
OPENTAD_EXPS="$OPENTAD_ROOT/exps"
EXPS_DEST="$DATA_DISK/opentad-exps/exps"

if [ -d "$OPENTAD_EXPS" ] && [ ! -L "$OPENTAD_EXPS" ]; then
    EXPS_SIZE=$(du -sh "$OPENTAD_EXPS" 2>/dev/null | cut -f1)
    echo "找到训练结果目录: $OPENTAD_EXPS ($EXPS_SIZE)"

    echo "  移动训练结果到: $EXPS_DEST"
    if [ -d "$EXPS_DEST" ] && [ "$(ls -A $EXPS_DEST 2>/dev/null)" ]; then
        echo "  合并现有内容..."
        rsync -av "$OPENTAD_EXPS/" "$EXPS_DEST/" 2>/dev/null || true
        rm -rf "$OPENTAD_EXPS"
    else
        mv "$OPENTAD_EXPS" "$EXPS_DEST" 2>/dev/null || {
            echo "  使用 rsync 复制（跨文件系统）..."
            rsync -av "$OPENTAD_EXPS/" "$EXPS_DEST/" 2>/dev/null || true
            rm -rf "$OPENTAD_EXPS"
        }
    fi

    ln -sf "$EXPS_DEST" "$OPENTAD_EXPS"
    echo "  ✓ 训练结果目录已移动到数据盘并创建符号链接"
elif [ -L "$OPENTAD_EXPS" ]; then
    CURRENT_LINK=$(readlink -f "$OPENTAD_EXPS")
    if [ "$CURRENT_LINK" != "$(readlink -f $EXPS_DEST)" ]; then
        echo "⚠️  符号链接指向: $CURRENT_LINK"
        echo "  更新为: $EXPS_DEST"
        rm -f "$OPENTAD_EXPS"
        ln -sf "$EXPS_DEST" "$OPENTAD_EXPS"
        echo "  ✓ 符号链接已更新"
    else
        echo "✓ 训练结果目录已正确链接到数据盘"
    fi
else
    echo "⚠️  训练结果目录不存在，创建符号链接..."
    ln -sf "$EXPS_DEST" "$OPENTAD_EXPS"
    echo "  ✓ 符号链接已创建"
fi
echo ""

# 验证所有链接
echo "=========================================="
echo "验证符号链接"
echo "=========================================="
echo ""

check_link() {
    local path=$1
    local expected=$2
    local name=$3

    if [ -L "$path" ]; then
        actual=$(readlink -f "$path")
        if [ "$actual" = "$(readlink -f $expected)" ]; then
            echo "✓ $name: $path -> $actual"
            return 0
        else
            echo "✗ $name: $path -> $actual (期望: $expected)"
            return 1
        fi
    elif [ -d "$path" ]; then
        echo "⚠️  $name: $path 是目录而非符号链接"
        return 1
    else
        echo "✗ $name: $path 不存在"
        return 1
    fi
}

check_link "$OPENTAD_DATA" "$DATA_DEST" "数据目录"
check_link "$OPENTAD_PRETRAINED" "$PRETRAINED_DEST" "预训练模型"
check_link "$OPENTAD_EXPS" "$EXPS_DEST" "训练结果"

echo ""

# 显示最终空间使用
echo "=========================================="
echo "最终空间使用情况"
echo "=========================================="
echo "根分区:"
df -h "$ROOT_PARTITION" | tail -1 | awk '{printf "  %s/%s (%s 使用, %s 可用)\n", $3, $2, $5, $4}'
echo ""
echo "数据盘:"
df -h "$DATA_DISK" | tail -1 | awk '{printf "  %s/%s (%s 使用, %s 可用)\n", $3, $2, $5, $4}'
echo ""

# 显示数据盘使用详情
echo "数据盘使用详情:"
du -sh "$DATA_DISK/opentad" "$DATA_DISK/opentad-exps" "$DATA_DISK/miniconda3" 2>/dev/null | while read size path; do
    echo "  $size  $path"
done

echo ""
echo "=========================================="
echo "数据管理完成！"
echo "=========================================="
echo ""
echo "所有数据相关目录已链接到数据盘:"
echo "  - data/          -> $DATA_DISK/opentad/data"
echo "  - pretrained/    -> $DATA_DISK/opentad/pretrained"
echo "  - exps/          -> $DATA_DISK/opentad-exps/exps"
if [ -L "/root/miniconda3" ]; then
    echo "  - miniconda3/   -> $DATA_DISK/miniconda3"
fi
echo ""
echo "现在可以安全地下载视频数据，所有数据都会存储在数据盘！"
echo ""

