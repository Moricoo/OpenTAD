#!/bin/bash
# 查找AdaTAD ActivityNet预训练权重

echo "=== 查找AdaTAD ActivityNet预训练权重 ==="
echo ""

echo "1. 检查README文件..."
if [ -f "configs/adatad/README.md" ]; then
    echo "找到README，搜索权重链接..."
    grep -i "activitynet\|anet\|checkpoint\|pretrained\|download\|google\|baidu\|hugging" configs/adatad/README.md | head -20
else
    echo "README不存在"
fi

echo ""
echo "2. 检查GitHub链接..."
grep -r "github.com\|github.io" configs/adatad/ 2>/dev/null | head -5

echo ""
echo "3. 检查配置文件中的注释..."
grep -r "#.*checkpoint\|#.*weight\|#.*download" configs/adatad/anet/ 2>/dev/null | head -10

echo ""
echo "4. 建议的查找方法："
echo "   - 访问: https://github.com/sming256/OpenTAD"
echo "   - 查看README.md中的Model Zoo部分"
echo "   - 查看configs/adatad/README.md"
echo "   - 查看GitHub Releases"
echo "   - 查看GitHub Issues中关于checkpoint的讨论"
