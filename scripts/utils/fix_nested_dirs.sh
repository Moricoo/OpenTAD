#!/bin/bash
# 修复嵌套目录的脚本

set -e

TARGET="/data/opentad-exps/exps"
NESTED="/data/opentad-exps/exps/opentad-exps"
EXPS_LINK="/root/OpenTAD/exps"

echo "=========================================="
echo "修复嵌套目录"
echo "=========================================="

# 1. 合并嵌套目录内容
if [ -d "$NESTED" ]; then
    echo ""
    echo "[1/3] 合并嵌套目录内容..."
    cd "$TARGET"
    
    # 检查嵌套目录内容
    if [ "$(ls -A $NESTED 2>/dev/null)" ]; then
        echo "  嵌套目录中有内容，开始合并..."
        for item in "$NESTED"/*; do
            if [ -e "$item" ]; then
                base=$(basename "$item")
                dst="$TARGET/$base"
                
                # 特殊处理：如果 item 是 exps 目录，需要合并其内容而不是移动整个目录
                if [ "$base" = "exps" ] && [ -d "$item" ]; then
                    echo "    合并嵌套的 exps 目录内容..."
                    for sub_item in "$item"/*; do
                        if [ -e "$sub_item" ]; then
                            sub_base=$(basename "$sub_item")
                            sub_dst="$TARGET/$sub_base"
                            if [ -e "$sub_dst" ]; then
                                if [ -d "$sub_item" ] && [ -d "$sub_dst" ]; then
                                    echo "      合并子目录: $sub_base"
                                    mv "$sub_item"/* "$sub_dst/" 2>/dev/null || true
                                    rmdir "$sub_item" 2>/dev/null || true
                                else
                                    echo "      替换子文件: $sub_base"
                                    rm -rf "$sub_dst"
                                    mv "$sub_item" "$sub_dst"
                                fi
                            else
                                echo "      移动子项: $sub_base"
                                mv "$sub_item" "$sub_dst"
                            fi
                        fi
                    done
                    # 删除空的嵌套 exps 目录
                    rmdir "$item" 2>/dev/null || true
                elif [ -e "$dst" ]; then
                    if [ -d "$item" ] && [ -d "$dst" ]; then
                        echo "    合并目录: $base"
                        mv "$item"/* "$dst/" 2>/dev/null || true
                        rmdir "$item" 2>/dev/null || true
                    else
                        echo "    替换文件: $base"
                        rm -rf "$dst"
                        mv "$item" "$dst"
                    fi
                else
                    echo "    移动: $base"
                    mv "$item" "$dst"
                fi
            fi
        done
    fi
    
    # 删除空的嵌套目录
    if [ -d "$NESTED" ] && [ -z "$(ls -A $NESTED 2>/dev/null)" ]; then
        rmdir "$NESTED"
        echo "  ✓ 嵌套目录已删除"
    else
        echo "  ⚠ 嵌套目录不为空，保留"
    fi
else
    echo "[1/3] ✓ 没有嵌套目录"
fi

# 2. 修复符号链接
echo ""
echo "[2/3] 修复符号链接..."
cd /root/OpenTAD

if [ -L "$EXPS_LINK" ]; then
    current=$(readlink "$EXPS_LINK")
    if [ "$current" != "$TARGET" ]; then
        echo "  更新符号链接: $current -> $TARGET"
        rm "$EXPS_LINK"
        ln -sf "$TARGET" "$EXPS_LINK"
        echo "  ✓ 符号链接已更新"
    else
        echo "  ✓ 符号链接正确"
    fi
elif [ -e "$EXPS_LINK" ]; then
    echo "  转换为符号链接..."
    if [ -d "$EXPS_LINK" ]; then
        # 移动内容
        for item in "$EXPS_LINK"/*; do
            if [ -e "$item" ]; then
                base=$(basename "$item")
                dst="$TARGET/$base"
                if [ -e "$dst" ]; then
                    if [ -d "$item" ] && [ -d "$dst" ]; then
                        mv "$item"/* "$dst/" 2>/dev/null || true
                        rmdir "$item" 2>/dev/null || true
                    else
                        rm -rf "$dst"
                        mv "$item" "$dst"
                    fi
                else
                    mv "$item" "$dst"
                fi
            fi
        done
        rmdir "$EXPS_LINK" 2>/dev/null || true
    else
        rm -f "$EXPS_LINK"
    fi
    ln -sf "$TARGET" "$EXPS_LINK"
    echo "  ✓ 符号链接已创建"
else
    ln -sf "$TARGET" "$EXPS_LINK"
    echo "  ✓ 符号链接已创建"
fi

# 3. 验证
echo ""
echo "[3/3] 验证结果..."
if [ -L "$EXPS_LINK" ]; then
    real_path=$(readlink -f "$EXPS_LINK")
    echo "  符号链接: $EXPS_LINK -> $real_path"
    if [ -d "$real_path" ]; then
        count=$(ls -1 "$real_path" 2>/dev/null | wc -l)
        echo "  ✓ 目录存在，包含 $count 项"
    else
        echo "  ✗ 目录不存在"
    fi
else
    echo "  ✗ exps 不是符号链接"
fi

# 检查是否还有嵌套
echo ""
echo "检查嵌套目录..."
nested_count=$(find /data/opentad-exps -type d -name "opentad-exps" ! -path "/data/opentad-exps/exps" 2>/dev/null | wc -l)
if [ "$nested_count" -eq 0 ]; then
    echo "  ✓ 没有嵌套目录了"
else
    echo "  ⚠ 仍有 $nested_count 个嵌套目录:"
    find /data/opentad-exps -type d -name "opentad-exps" ! -path "/data/opentad-exps/exps" 2>/dev/null
fi

echo ""
echo "=========================================="
echo "修复完成！"
echo "=========================================="

