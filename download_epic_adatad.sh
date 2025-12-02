#!/bin/bash
# EPIC-KITCHENS AdaTAD 完整下载脚本

set -e

echo "=== EPIC-KITCHENS AdaTAD 下载脚本 ==="
echo ""

# 检查gdown
if ! command -v gdown &> /dev/null; then
    echo "安装gdown..."
    pip install gdown
fi

# 创建目录
mkdir -p /root/OpenTAD/pretrained
mkdir -p /root/OpenTAD/pretrained/adatad
cd /root/OpenTAD/pretrained

echo "📥 步骤1: 下载EPIC-pretrained VideoMAE权重"
echo ""

# EPIC-Verb预训练权重
echo "下载EPIC-Verb预训练权重..."
gdown https://drive.google.com/uc?id=1h7oLiNN5LTXau4HWmmzS_ekvuNdZkp-b \
    -O vit-large-p16_videomae-epic_verb.pth || echo "⚠️ 下载失败，请手动下载"

# EPIC-Noun预训练权重
echo "下载EPIC-Noun预训练权重..."
gdown https://drive.google.com/uc?id=1nRuzJI4ej90vFsKCBSugRVOmxrR8urwW \
    -O vit-large-p16_videomae-epic_noun.pth || echo "⚠️ 下载失败，请手动下载"

echo ""
echo "📥 步骤2: 下载AdaTAD训练好的模型权重"
echo ""

cd /root/OpenTAD/pretrained/adatad

# Verb模型
echo "下载AdaTAD EPIC-Verb模型..."
gdown https://drive.google.com/uc?id=16Hq3sHu0S97Ge2AewHT6DOaHSo0TqIlx \
    -O adatad_epic_verb.pth || echo "⚠️ 下载失败，请手动下载"

# Noun模型
echo "下载AdaTAD EPIC-Noun模型..."
gdown https://drive.google.com/uc?id=17k3f6wirqniLTjKOsIXbfqJPA_iLb88E \
    -O adatad_epic_noun.pth || echo "⚠️ 下载失败，请手动下载"

echo ""
echo "✅ 下载完成！"
echo ""
echo "📊 文件清单："
echo "预训练权重："
ls -lh /root/OpenTAD/pretrained/vit-large-p16_videomae-epic_*.pth 2>/dev/null || echo "  未找到"
echo ""
echo "AdaTAD模型："
ls -lh /root/OpenTAD/pretrained/adatad/adatad_epic_*.pth 2>/dev/null || echo "  未找到"
