#!/bin/bash
# 修复嵌套的 exps 目录结构

set -e

DATA_DIR="/data/opentad-exps"
EXPS_LINK="/root/OpenTAD/exps"

echo "=========================================="
echo "修复嵌套的 exps 目录结构"
echo "=========================================="

# 检查当前结构
echo ""
echo "当前目录结构:"
ls -la "$DATA_DIR" 2>/dev/null | head -10
echo ""

# 检查是否有嵌套的 exps/opentad-exps
if [ -d "$DATA_DIR/exps/opentad-exps" ]; then
    echo "发现嵌套目录: $DATA_DIR/exps/opentad-exps"
    echo ""
    
    # 检查嵌套目录中是否有实际内容
    NESTED_CONTENT=$(find "$DATA_DIR/exps/opentad-exps" -mindepth 1 -maxdepth 1 2>/dev/null | wc -l)
    EXPS_CONTENT=$(find "$DATA_DIR/exps" -mindepth 1 -maxdepth 1 ! -name "opentad-exps" 2>/dev/null | wc -l)
    
    echo "嵌套目录内容数: $NESTED_CONTENT"
    echo "exps 目录其他内容数: $EXPS_CONTENT"
    echo ""
    
    if [ "$NESTED_CONTENT" -gt 0 ]; then
        echo "合并嵌套目录内容..."
        # 将嵌套目录的内容移动到 exps
        mv "$DATA_DIR/exps/opentad-exps"/* "$DATA_DIR/exps/" 2>/dev/null || true
        rmdir "$DATA_DIR/exps/opentad-exps" 2>/dev/null || true
        echo "✓ 嵌套目录内容已合并"
    else
        echo "删除空的嵌套目录..."
        rmdir "$DATA_DIR/exps/opentad-exps" 2>/dev/null || true
        echo "✓ 空嵌套目录已删除"
    fi
fi

# 检查符号链接
echo ""
echo "检查符号链接..."
if [ -L "$EXPS_LINK" ]; then
    TARGET=$(readlink "$EXPS_LINK")
    echo "当前符号链接: $EXPS_LINK -> $TARGET"
    
    # 确保指向正确的路径
    if [ "$TARGET" != "/data/opentad-exps/exps" ]; then
        echo "修复符号链接..."
        rm "$EXPS_LINK"
        ln -sf "/data/opentad-exps/exps" "$EXPS_LINK"
        echo "✓ 符号链接已修复"
    else
        echo "✓ 符号链接正确"
    fi
else
    echo "exps 不是符号链接，创建符号链接..."
    if [ -d "$EXPS_LINK" ]; then
        # 如果 exps 是目录，先移动
        if [ "$(ls -A $EXPS_LINK 2>/dev/null)" ]; then
            echo "移动现有 exps 目录内容..."
            mkdir -p "$DATA_DIR/exps"
            mv "$EXPS_LINK"/* "$DATA_DIR/exps/" 2>/dev/null || true
        fi
        rmdir "$EXPS_LINK" 2>/dev/null || true
    fi
    ln -sf "/data/opentad-exps/exps" "$EXPS_LINK"
    echo "✓ 符号链接已创建"
fi

# 验证
echo ""
echo "验证最终结构:"
if [ -L "$EXPS_LINK" ]; then
    REAL_PATH=$(readlink -f "$EXPS_LINK")
    echo "符号链接: $EXPS_LINK"
    echo "实际路径: $REAL_PATH"
    
    if [ -d "$REAL_PATH" ]; then
        echo "✓ 目录存在"
        ITEMS=$(ls -1 "$REAL_PATH" 2>/dev/null | head -5)
        if [ -n "$ITEMS" ]; then
            echo "目录内容:"
            echo "$ITEMS" | sed 's/^/  /'
        fi
    fi
fi

echo ""
echo "=========================================="
echo "修复完成！"
echo "=========================================="

