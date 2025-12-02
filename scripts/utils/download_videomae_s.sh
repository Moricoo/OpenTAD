#!/bin/bash
# 下载 VideoMAE-S 预训练模型

set -e

PRETRAINED_DIR="pretrained"
# 配置文件中使用的文件名（带 _my 后缀）
TARGET_NAME="vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth"
# 下载的原始文件名
DOWNLOAD_NAME="vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-860a3cd3.pth"
TARGET_PATH="$PRETRAINED_DIR/$TARGET_NAME"
DOWNLOAD_PATH="$PRETRAINED_DIR/$DOWNLOAD_NAME"

echo "=========================================="
echo "下载 VideoMAE-S 预训练模型"
echo "=========================================="

# 创建目录
mkdir -p "$PRETRAINED_DIR"

# 检查是否已存在目标文件
if [ -f "$TARGET_PATH" ]; then
    echo "✓ 模型已存在: $TARGET_PATH"
    exit 0
fi

# 检查是否有下载的文件（可能文件名不同）
if [ -f "$DOWNLOAD_PATH" ]; then
    echo "发现已下载的文件，创建符号链接..."
    ln -sf "$DOWNLOAD_NAME" "$TARGET_PATH"
    echo "✓ 已创建符号链接: $TARGET_PATH -> $DOWNLOAD_PATH"
    exit 0
fi

echo ""
echo "尝试下载..."
echo ""

# 方法 1: 使用 gdown 从 Google Drive 下载（推荐）
if command -v gdown &> /dev/null; then
    echo "[1/3] 使用 gdown 从 Google Drive 下载..."
    gdown "https://drive.google.com/uc?id=1BH5BZmdImaZesUfqtW23eBGC341Gui1D" -O "$DOWNLOAD_PATH"
    if [ -f "$DOWNLOAD_PATH" ] && [ -s "$DOWNLOAD_PATH" ]; then
        echo "✓ 下载成功！"
        # 创建符号链接或重命名
        if [ "$TARGET_NAME" != "$DOWNLOAD_NAME" ]; then
            ln -sf "$DOWNLOAD_NAME" "$TARGET_PATH"
            echo "✓ 已创建符号链接: $TARGET_PATH -> $DOWNLOAD_PATH"
        else
            mv "$DOWNLOAD_PATH" "$TARGET_PATH"
        fi
        exit 0
    fi
else
    echo "  gdown 未安装，跳过"
    echo "  可以安装: pip install gdown"
fi

# 方法 2: 使用 wget 尝试备用链接
echo ""
echo "[2/3] 尝试备用下载链接..."
# 注意：mmaction2 链接可能已失效，但可以尝试
URLS=(
    "https://download.openmmlab.com/mmaction/v1.0/recognition/videomae/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-860a3cd3.pth"
)

for url in "${URLS[@]}"; do
    echo "  尝试: $url"
    if wget "$url" -O "$DOWNLOAD_PATH" 2>&1 | grep -q "200 OK"; then
        if [ -f "$DOWNLOAD_PATH" ] && [ -s "$DOWNLOAD_PATH" ]; then
            echo "✓ 下载成功！"
            if [ "$TARGET_NAME" != "$DOWNLOAD_NAME" ]; then
                ln -sf "$DOWNLOAD_NAME" "$TARGET_PATH"
                echo "✓ 已创建符号链接: $TARGET_PATH -> $DOWNLOAD_PATH"
            else
                mv "$DOWNLOAD_PATH" "$TARGET_PATH"
            fi
            exit 0
        fi
    fi
    rm -f "$DOWNLOAD_PATH"
done

# 方法 3: 手动下载提示
echo ""
echo "[3/3] 自动下载失败，请手动下载"
echo "=========================================="
echo "请手动下载 VideoMAE-S 预训练模型："
echo ""
echo "方法 1: 使用 gdown（推荐）"
echo "  pip install gdown"
echo "  cd /root/OpenTAD"
echo "  mkdir -p pretrained"
echo "  gdown 'https://drive.google.com/uc?id=1BH5BZmdImaZesUfqtW23eBGC341Gui1D' \\"
echo "    -O pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-860a3cd3.pth"
echo "  cd pretrained"
echo "  ln -sf vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-860a3cd3.pth \\"
echo "    vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth"
echo ""
echo "方法 2: 手动从 Google Drive 下载"
echo "  URL: https://drive.google.com/file/d/1BH5BZmdImaZesUfqtW23eBGC341Gui1D/view?usp=sharing"
echo "  下载后："
echo "  1. 保存到: pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-860a3cd3.pth"
echo "  2. 创建符号链接:"
echo "     cd pretrained"
echo "     ln -sf vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-860a3cd3.pth \\"
echo "       vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth"
echo ""
echo "注意: 配置文件中使用的文件名是: $TARGET_NAME"
echo "=========================================="
exit 1
