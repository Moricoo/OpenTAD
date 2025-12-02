#!/bin/bash
# 下载AdaTAD ActivityNet预训练权重

echo "=== AdaTAD ActivityNet 权重下载脚本 ==="
echo ""

# 创建目录
mkdir -p /root/OpenTAD/pretrained/adatad
cd /root/OpenTAD/pretrained/adatad

# 检查gdown是否安装
if ! command -v gdown &> /dev/null; then
    echo "⚠️  gdown未安装，正在安装..."
    pip install gdown
fi

echo "📥 推荐下载: VideoMAE-L (cls) - 无需分类器"
echo "   下载链接: https://drive.google.com/file/d/1VYAvDrc7O7W4hDmUjjE6y32WmVNQ4ZR_/view?usp=sharing"
echo ""
read -p "是否下载VideoMAE-L (cls)版本? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "正在下载..."
    gdown https://drive.google.com/uc?id=1VYAvDrc7O7W4hDmUjjE6y32WmVNQ4ZR_ \
        -O adatad_anet_videomae_l_224_cls.pth
    echo "✅ 下载完成: adatad_anet_videomae_l_224_cls.pth"
fi

echo ""
echo "其他可选模型："
echo "  - VideoMAE-S: gdown https://drive.google.com/uc?id=1gncN-xjArNtgVoBKCwCJCH4ISA3yVqIU -O adatad_anet_videomae_s_160.pth"
echo "  - VideoMAE-L: gdown https://drive.google.com/uc?id=1GxwNLc1rRp6x5ug1zd1r_1DmYCZD_tw5 -O adatad_anet_videomae_l_160.pth"
