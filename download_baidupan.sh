#!/bin/bash
# 百度网盘下载脚本

SHARE_LINK="https://pan.baidu.com/s/10bR45978LoTa6ON3mPeQoQ?pwd=9527"
EXTRACT_CODE="9527"
DOWNLOAD_DIR="/root/OpenTAD/downloads"

echo "=== 百度网盘下载工具 ==="
echo ""
echo "📥 分享链接: ${SHARE_LINK}"
echo "🔑 提取码: ${EXTRACT_CODE}"
echo "📁 下载目录: ${DOWNLOAD_DIR}"
echo ""

# 创建下载目录
mkdir -p ${DOWNLOAD_DIR}
cd ${DOWNLOAD_DIR}

echo "📋 使用说明："
echo ""
echo "方法1: 使用bypy（需要先授权登录）"
echo "  1. 首次使用需要授权："
echo "     bypy info"
echo "     会显示一个授权链接，在浏览器中打开并授权"
echo ""
echo "  2. 授权后，如果文件已转存到您的网盘，使用："
echo "     bypy downfile <网盘中的文件路径> <本地保存路径>"
echo ""
echo "方法2: 使用BaiduPCS-Go（支持分享链接直接下载）"
echo "  需要安装BaiduPCS-Go工具"
echo ""
echo "方法3: 手动下载（推荐用于大文件）"
echo "  1. 在浏览器中打开分享链接"
echo "  2. 输入提取码: ${EXTRACT_CODE}"
echo "  3. 下载文件到本地"
echo "  4. 使用scp上传到服务器"
echo ""

# 检查bypy是否已授权
echo "🔍 检查bypy授权状态..."
if bypy info > /dev/null 2>&1; then
    echo "✅ bypy已授权"
    echo ""
    echo "💡 如果文件已转存到您的网盘，可以使用以下命令下载："
    echo "   bypy list                    # 查看网盘文件列表"
    echo "   bypy downfile <文件路径> .    # 下载文件到当前目录"
else
    echo "⚠️  bypy未授权，需要先授权"
    echo ""
    echo "📝 授权步骤："
    echo "  1. 运行: bypy info"
    echo "  2. 复制显示的授权链接"
    echo "  3. 在浏览器中打开链接并授权"
    echo "  4. 复制授权码并粘贴回终端"
    echo ""
    echo "或者运行以下命令开始授权："
    echo "  bypy info"
fi

echo ""
echo "📌 提示："
echo "  - 对于大文件（>1GB），建议使用浏览器下载后上传"
echo "  - bypy适合下载已转存到个人网盘的文件"
echo "  - 分享链接通常需要先转存才能用bypy下载"

