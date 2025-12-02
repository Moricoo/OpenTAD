# 百度网盘下载指南

## 📥 分享链接信息

- **链接**: https://pan.baidu.com/s/10bR45978LoTa6ON3mPeQoQ?pwd=9527
- **提取码**: 9527
- **下载目录**: `/root/OpenTAD/downloads`

---

## 方法1: 使用bypy下载（已安装）✅

### 步骤1: 授权登录（首次使用）

```bash
bypy info
```

这会显示一个授权链接，例如：
```
Please visit:
https://openapi.baidu.com/oauth/2.0/authorize?client_id=...

And authorize this app
Paste the Authorization Code here within 10 minutes.
Press [Enter] when you are done
```

**操作步骤**:
1. 复制显示的授权链接
2. 在浏览器中打开链接
3. 登录百度账号并授权
4. 复制返回的授权码
5. 粘贴到终端并按回车

### 步骤2: 转存文件到自己的网盘

由于bypy不支持直接下载分享链接，需要先转存：

1. 在浏览器中打开分享链接: https://pan.baidu.com/s/10bR45978LoTa6ON3mPeQoQ?pwd=9527
2. 输入提取码: `9527`
3. 选择文件，点击"保存到网盘"
4. 选择保存位置（建议保存到根目录或容易找到的位置）

### 步骤3: 使用bypy下载

```bash
# 查看网盘文件列表
bypy list

# 查看指定目录的文件
bypy list <目录路径>

# 下载文件到当前目录
cd /root/OpenTAD/downloads
bypy downfile <网盘中的文件路径> .

# 下载整个目录
bypy downdir <网盘中的目录路径> .
```

**示例**:
```bash
# 如果文件保存在网盘根目录，名为 dataset.zip
bypy downfile dataset.zip /root/OpenTAD/downloads/

# 如果文件在某个文件夹中
bypy downfile /我的数据集/dataset.zip /root/OpenTAD/downloads/
```

---

## 方法2: 使用BaiduPCS-Go（支持分享链接直接下载）

### 安装BaiduPCS-Go

```bash
# 下载最新版本（Linux x64）
cd /tmp
wget https://github.com/qjfoidnh/BaiduPCS-Go/releases/latest/download/BaiduPCS-Go-v3.9.4-linux-x64.zip
unzip BaiduPCS-Go-v3.9.4-linux-x64.zip
sudo mv BaiduPCS-Go /usr/local/bin/
chmod +x /usr/local/bin/BaiduPCS-Go
```

### 使用BaiduPCS-Go下载分享链接

```bash
# 登录（首次使用）
BaiduPCS-Go login

# 下载分享链接（需要提取码）
BaiduPCS-Go share transfer <分享链接> <提取码> -save <保存路径>

# 示例
cd /root/OpenTAD/downloads
BaiduPCS-Go share transfer "https://pan.baidu.com/s/10bR45978LoTa6ON3mPeQoQ?pwd=9527" 9527 -save .
```

---

## 方法3: 手动下载后上传（推荐用于大文件）

### 步骤1: 在本地电脑下载

1. 在浏览器中打开分享链接
2. 输入提取码: `9527`
3. 下载文件到本地

### 步骤2: 上传到服务器

```bash
# 在本地电脑执行（替换<服务器IP>为实际IP）
scp <本地文件路径> root@<服务器IP>:/root/OpenTAD/downloads/

# 示例
scp dataset.zip root@192.168.1.100:/root/OpenTAD/downloads/
```

---

## 方法4: 使用rclone（如果已配置）

如果已经配置了rclone的百度网盘：

```bash
# 查看配置
rclone listremotes

# 下载文件
rclone copy baidu:/<文件路径> /root/OpenTAD/downloads/
```

---

## 📋 快速开始（推荐流程）

### 最简单的方法（bypy）

```bash
# 1. 授权登录（只需一次）
bypy info
# 按照提示在浏览器中授权

# 2. 在浏览器中转存文件到自己的网盘

# 3. 查看网盘文件
bypy list

# 4. 下载文件
cd /root/OpenTAD/downloads
bypy downfile <文件路径> .
```

### 支持分享链接的方法（BaiduPCS-Go）

```bash
# 1. 安装BaiduPCS-Go（见上方安装步骤）

# 2. 登录
BaiduPCS-Go login

# 3. 直接下载分享链接
cd /root/OpenTAD/downloads
BaiduPCS-Go share transfer "https://pan.baidu.com/s/10bR45978LoTa6ON3mPeQoQ?pwd=9527" 9527 -save .
```

---

## 🔧 常用命令

### bypy常用命令

```bash
# 查看网盘信息
bypy info

# 列出文件
bypy list
bypy list <目录>

# 下载文件
bypy downfile <远程路径> <本地路径>

# 下载目录
bypy downdir <远程目录> <本地目录>

# 上传文件
bypy upload <本地文件> <远程路径>

# 同步目录
bypy syncup <本地目录> <远程目录>
```

### BaiduPCS-Go常用命令

```bash
# 登录
BaiduPCS-Go login

# 下载分享链接
BaiduPCS-Go share transfer <链接> <提取码> -save <路径>

# 列出文件
BaiduPCS-Go ls

# 下载文件
BaiduPCS-Go download <文件路径>
```

---

## ⚠️ 注意事项

1. **文件大小**: 如果文件很大（>10GB），建议使用方法3（手动下载后上传）
2. **网络稳定性**: 确保网络连接稳定，大文件下载可能需要较长时间
3. **存储空间**: 确保服务器有足够的存储空间
4. **授权安全**: bypy授权信息保存在 `~/.bypy/` 目录，注意保护

---

## 🐛 常见问题

### Q1: bypy授权失败
**A**: 确保网络可以访问百度API，可能需要代理

### Q2: 下载速度慢
**A**:
- 百度网盘限速，这是正常现象
- 可以考虑使用百度网盘会员账号
- 或者使用手动下载方式

### Q3: 分享链接无法直接下载
**A**:
- bypy不支持分享链接直接下载，需要先转存
- 使用BaiduPCS-Go可以支持分享链接直接下载

### Q4: 文件下载中断
**A**:
- bypy支持断点续传，重新运行下载命令即可
- 或者使用rsync等支持断点续传的工具

---

## 📝 当前状态

- ✅ bypy已安装: `/root/miniconda3/bin/bypy`
- ⚠️  bypy未授权: 需要运行 `bypy info` 进行授权
- 📁 下载目录: `/root/OpenTAD/downloads`（已创建）

---

## 🚀 立即开始

运行以下命令开始授权并下载：

```bash
# 1. 授权登录
bypy info

# 2. 在浏览器中转存文件后，查看文件列表
bypy list

# 3. 下载文件
cd /root/OpenTAD/downloads
bypy downfile <文件路径> .
```

