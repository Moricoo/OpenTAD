#!/bin/bash
# EPIC-KITCHENS AdaTAD 完整设置脚本

set -e

echo "=== EPIC-KITCHENS AdaTAD 完整设置 ==="
echo ""

# 检查目录
cd /root/OpenTAD

# 创建必要目录
echo "📁 创建目录结构..."
mkdir -p pretrained
mkdir -p pretrained/adatad
mkdir -p data/epic_kitchens-100/annotations
mkdir -p data/epic_kitchens-100/raw_data
mkdir -p /data/videos/epic_test

echo "✅ 目录创建完成"
echo ""

# 检查文件
echo "🔍 检查已下载的文件..."
echo ""

# 检查预训练权重
echo "EPIC-pretrained VideoMAE权重:"
if [ -f "pretrained/vit-large-p16_videomae-epic_verb.pth" ]; then
    echo "  ✅ vit-large-p16_videomae-epic_verb.pth ($(du -h pretrained/vit-large-p16_videomae-epic_verb.pth | cut -f1))"
else
    echo "  ❌ vit-large-p16_videomae-epic_verb.pth (未找到)"
    echo "     下载链接: https://drive.google.com/file/d/1h7oLiNN5LTXau4HWmmzS_ekvuNdZkp-b/view?usp=sharing"
fi

if [ -f "pretrained/vit-large-p16_videomae-epic_noun.pth" ]; then
    echo "  ✅ vit-large-p16_videomae-epic_noun.pth ($(du -h pretrained/vit-large-p16_videomae-epic_noun.pth | cut -f1))"
else
    echo "  ❌ vit-large-p16_videomae-epic_noun.pth (未找到)"
    echo "     下载链接: https://drive.google.com/file/d/1nRuzJI4ej90vFsKCBSugRVOmxrR8urwW/view?usp=sharing"
fi

echo ""

# 检查AdaTAD模型
echo "AdaTAD模型权重:"
if [ -f "pretrained/adatad/adatad_epic_verb.pth" ]; then
    echo "  ✅ adatad_epic_verb.pth ($(du -h pretrained/adatad/adatad_epic_verb.pth | cut -f1))"
else
    echo "  ❌ adatad_epic_verb.pth (未找到)"
    echo "     下载链接: https://drive.google.com/file/d/16Hq3sHu0S97Ge2AewHT6DOaHSo0TqIlx/view?usp=sharing"
fi

if [ -f "pretrained/adatad/adatad_epic_noun.pth" ]; then
    echo "  ✅ adatad_epic_noun.pth ($(du -h pretrained/adatad/adatad_epic_noun.pth | cut -f1))"
else
    echo "  ❌ adatad_epic_noun.pth (未找到)"
    echo "     下载链接: https://drive.google.com/file/d/17k3f6wirqniLTjKOsIXbfqJPA_iLb88E/view?usp=sharing"
fi

echo ""

# 检查标注文件
echo "EPIC-KITCHENS标注文件:"
if [ -d "data/epic_kitchens-100/annotations" ] && [ "$(ls -A data/epic_kitchens-100/annotations 2>/dev/null)" ]; then
    echo "  ✅ 标注文件已存在"
    ls -lh data/epic_kitchens-100/annotations/ | head -5
else
    echo "  ❌ 标注文件未找到"
    echo "     运行: cd tools/prepare_data/epic && bash download_annotation.sh"
fi

echo ""

# 总结
echo "📊 准备状态总结:"
ALL_READY=true

[ ! -f "pretrained/vit-large-p16_videomae-epic_verb.pth" ] && ALL_READY=false
[ ! -f "pretrained/vit-large-p16_videomae-epic_noun.pth" ] && ALL_READY=false
[ ! -f "pretrained/adatad/adatad_epic_verb.pth" ] && ALL_READY=false
[ ! -f "pretrained/adatad/adatad_epic_noun.pth" ] && ALL_READY=false

if [ "$ALL_READY" = true ]; then
    echo "  ✅ 所有必需文件已准备就绪！"
    echo ""
    echo "🚀 可以开始推理："
    echo ""
    echo "  # Verb检测（动作）"
    echo "  CUDA_VISIBLE_DEVICES=0 torchrun \\"
    echo "      --nnodes=1 --nproc_per_node=1 \\"
    echo "      --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \\"
    echo "      tools/test.py \\"
    echo "      configs/adatad/epic/e2e_epic_videomae_l_ft_768x8_160_adapter_verb.py \\"
    echo "      --checkpoint pretrained/adatad/adatad_epic_verb.pth"
    echo ""
    echo "  # Noun检测（物体）"
    echo "  CUDA_VISIBLE_DEVICES=0 torchrun \\"
    echo "      --nnodes=1 --nproc_per_node=1 \\"
    echo "      --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \\"
    echo "      tools/test.py \\"
    echo "      configs/adatad/epic/e2e_epic_videomae_l_ft_768x8_160_adapter_noun.py \\"
    echo "      --checkpoint pretrained/adatad/adatad_epic_noun.pth"
else
    echo "  ⚠️  部分文件缺失，请先下载"
    echo ""
    echo "📥 下载方法："
    echo "  1. 查看下载清单: cat epic_download_checklist.md"
    echo "  2. 手动下载后上传，或使用: ./download_epic_adatad.sh"
fi

echo ""
echo "📚 详细指南: EPIC_KITCHENS_COMPLETE_GUIDE.md"

