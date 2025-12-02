# ActivityNet-1.3 数据准备指南

## 📋 概述

ActivityNet-1.3 是时序动作检测的常用数据集，包含约 20,000 个视频和 200 个动作类别。

## 📁 数据目录结构

```
data/activitynet-1.3/
├── annotations/          # 注释文件（JSON）
│   ├── activity_net.v1-3.min.json
│   └── anet_anno_action.json
└── raw_data/
    └── video/           # 原始视频文件
        ├── v_xxx.mp4
        ├── v_yyy.mp4
        └── ...
```

## 🔽 步骤 1: 下载注释文件

### 方法 1: 使用 gdown（如果网络允许）

```bash
cd /root/OpenTAD
source /root/miniconda3/bin/activate opentad

# 安装 gdown（如果未安装）
pip install gdown

# 下载注释文件
cd tools/prepare_data/activitynet
bash download_annotation.sh
```

### 方法 2: 手动下载（推荐）

1. **访问 Google Drive 链接**：
   - https://drive.google.com/drive/folders/1HpTc6FbYnm-s9tY4aZljjZnYnThICcNq

2. **下载注释文件**：
   - 下载整个文件夹或单个 JSON 文件
   - 主要文件：`activity_net.v1-3.min.json` 或 `anet_anno_action.json`

3. **上传到服务器**：
   ```bash
   # 在本地机器上执行
   scp /path/to/activity_net.v1-3.min.json \
       root@your-server:/root/OpenTAD/data/activitynet-1.3/annotations/
   ```

4. **验证**：
   ```bash
   ls -lh /root/OpenTAD/data/activitynet-1.3/annotations/
   ```

## 🎬 步骤 2: 下载原始视频

### 方法 1: 官方网站（需要申请访问）

1. **访问官方网站**：
   - https://docs.google.com/forms/d/e/1FAIpQLSdxhNVeeSCwB2USAfeNWCaI9saVT6i2hpiiizVYfa3MsTyamg/viewform
   - 填写表单申请访问权限（7 天有效期）

2. **下载视频**：
   - 下载所有训练集和验证集视频
   - 视频文件命名格式：`v_xxx.mp4`

### 方法 2: 使用处理后的版本（推荐）

**Anet_videos_15fps_short256.zip**
- 已转换为 15fps
- 短边调整为 256 像素
- 适合端到端训练

下载链接：在 ActivityNet 官方 Google Drive 文件夹中查找

### 上传视频文件

```bash
# 方法 1: 上传到数据盘（推荐）
# 在本地机器上执行
scp -r /path/to/videos/* \
    root@your-server:/data/opentad/data/activitynet-1.3/raw_data/video/

# 方法 2: 如果上传到根分区，会自动链接到数据盘
scp -r /path/to/videos/* \
    root@your-server:/root/OpenTAD/data/activitynet-1.3/raw_data/video/
```

### 解压视频（如果是压缩包）

```bash
cd /data/opentad/data/activitynet-1.3/raw_data/video

# 如果是 zip 文件
unzip Anet_videos_15fps_short256.zip

# 如果是 tar.gz 文件
tar -xzf Anet_videos_15fps_short256.tar.gz
```

## ✅ 步骤 3: 验证数据

运行验证脚本：

```bash
cd /root/OpenTAD
bash scripts/utils/prepare_activitynet_data.sh
```

或者手动验证：

```bash
# 检查注释文件
ls -lh data/activitynet-1.3/annotations/*.json

# 检查视频文件数量
find data/activitynet-1.3/raw_data/video/ -type f \( -name "*.mp4" -o -name "*.avi" \) | wc -l

# 检查数据目录结构
tree -L 2 data/activitynet-1.3/ 2>/dev/null || ls -R data/activitynet-1.3/
```

## 📊 数据统计

- **训练集**: ~10,024 个视频
- **验证集**: ~4,728 个视频（部分视频已失效）
- **测试集**: ~5,044 个视频
- **动作类别**: 200 个
- **视频格式**: MP4（推荐 15fps, 短边 256px）

## 🚀 开始训练

数据准备完成后，可以开始训练：

```bash
cd /root/OpenTAD
source /root/miniconda3/bin/activate opentad

# 训练 VideoMAE-S 小模型（4 个 GPU）
torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    tools/train.py \
    configs/adatad/anet/e2e_anet_videomae_s_192x4_160_adapter.py
```

## 📝 配置文件说明

ActivityNet 的 AdaTAD 配置文件：
- `configs/adatad/anet/e2e_anet_videomae_s_192x4_160_adapter.py` - VideoMAE-S
- `configs/adatad/anet/e2e_anet_videomae_b_192x4_160_adapter.py` - VideoMAE-B
- `configs/adatad/anet/e2e_anet_videomae_l_192x4_160_adapter.py` - VideoMAE-L

### 关键参数

- **resize_length**: 192（视频长度）
- **scale_factor**: 4（实际处理 192×4=768 帧）
- **图像尺寸**: 160×160
- **视频前缀**: `v_`（配置文件中的 `prefix="v_"`）

## ⚠️ 注意事项

1. **视频命名**：确保视频文件名以 `v_` 开头（如 `v_xxx.mp4`）
2. **数据盘存储**：所有数据会自动存储在 `/data/opentad/data/`（已配置符号链接）
3. **磁盘空间**：ActivityNet 视频约 600GB，确保数据盘有足够空间
4. **网络问题**：如果无法访问 Google Drive，需要手动下载并上传

## 🔗 相关链接

- **ActivityNet 官网**: http://activity-net.org/
- **论文**: https://arxiv.org/abs/1505.04785
- **数据下载**: https://docs.google.com/forms/d/e/1FAIpQLSdxhNVeeSCwB2USAfeNWCaI9saVT6i2hpiiizVYfa3MsTyamg/viewform
- **注释文件**: https://drive.google.com/drive/folders/1HpTc6FbYnm-s9tY4aZljjZnYnThICcNq

## 📞 故障排除

### 问题 1: 注释文件下载失败

**解决**：
- 手动从 Google Drive 下载
- 使用代理或 VPN
- 在本地下载后上传到服务器

### 问题 2: 视频文件找不到

**检查**：
```bash
# 检查视频目录
ls -lh data/activitynet-1.3/raw_data/video/ | head -10

# 检查视频命名格式
ls data/activitynet-1.3/raw_data/video/ | head -5
# 应该看到 v_xxx.mp4 格式的文件
```

### 问题 3: 训练时提示找不到视频

**解决**：
1. 检查配置文件中的 `prefix="v_"`
2. 确保视频文件名以 `v_` 开头
3. 运行数据设置脚本：`bash scripts/utils/prepare_activitynet_data.sh`

---

**数据准备完成后，就可以开始训练 ActivityNet 模型了！** 🎉

