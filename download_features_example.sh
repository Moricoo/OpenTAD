#!/bin/bash
# 特征文件下载示例脚本

echo "=== 特征文件下载示例 ==="
echo ""

# 设置下载目录（使用/data分区）
DOWNLOAD_BASE="/data/OpenTAD/data"
mkdir -p ${DOWNLOAD_BASE}

echo "📁 下载目录: ${DOWNLOAD_BASE}"
echo ""

# THUMOS-14 I3D特征
echo "📦 THUMOS-14 I3D特征:"
echo "  目录: ${DOWNLOAD_BASE}/thumos-14/features/i3d_actionformer_stride4_thumos/"
echo "  大小: 约10-50GB"
echo "  来源: 百度网盘/官方/社区分享"
echo "  下载后解压: tar -xzf i3d_actionformer_stride4_thumos.tar"
echo ""

# ActivityNet-1.3 TSP特征
echo "📦 ActivityNet-1.3 TSP特征:"
echo "  目录: ${DOWNLOAD_BASE}/activitynet-1.3/features/anet_tsp_npy_unresize/"
echo "  大小: 约50-200GB"
echo "  来源: 官方/社区分享"
echo "  下载后解压: tar -xzf anet_tsp_npy_unresize.tar.gz"
echo ""

# EPIC-KITCHENS特征
echo "📦 EPIC-KITCHENS-100特征:"
echo "  目录: ${DOWNLOAD_BASE}/epic_kitchens-100/features/"
echo "  大小: 约50-200GB"
echo "  来源: 官方/社区分享"
echo ""

echo "💡 下载步骤："
echo "  1. 从百度网盘或其他来源下载特征文件压缩包"
echo "  2. 使用bypy或其他工具下载到服务器"
echo "  3. 解压到对应目录"
echo "  4. 验证文件完整性"
echo "  5. 使用features_*.py配置文件训练"
echo ""

echo "📋 示例命令："
echo "  # 创建目录"
echo "  mkdir -p ${DOWNLOAD_BASE}/thumos-14/features"
echo "  cd ${DOWNLOAD_BASE}/thumos-14/features"
echo ""
echo "  # 下载（使用bypy或其他工具）"
echo "  # bypy downfile <网盘路径> ."
echo ""
echo "  # 解压"
echo "  tar -xzf i3d_actionformer_stride4_thumos.tar"
echo ""
echo "  # 验证"
echo "  ls -lh i3d_actionformer_stride4_thumos/ | head -10"
