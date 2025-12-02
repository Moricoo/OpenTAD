# 特征文件下载指南

## 📊 什么是特征文件？

特征文件是**预处理后的视频特征**，通常由预训练模型（如I3D、VideoMAE、SlowFast等）提取，以`.npy`格式存储。

### 优势

✅ **存储空间小**: 通常只有原始视频的10-40%
✅ **训练速度快**: 无需视频解码，直接加载特征
✅ **下载快速**: 文件小，下载时间短

---

## 📦 各数据集特征文件信息

### THUMOS-14

**特征类型**: I3D特征
**路径**: `data/thumos-14/features/i3d_actionformer_stride4_thumos/`
**大小**: 约10-50GB
**格式**: `.npy`文件
**配置**: `configs/_base_/datasets/thumos-14/features_i3d_pad.py`

### ActivityNet-1.3

**特征类型**: TSP (TimeSformer)特征
**路径**: `data/activitynet-1.3/features/anet_tsp_npy_unresize/`
**大小**: 约50-200GB
**格式**: `.npy`文件
**配置**: `configs/_base_/datasets/activitynet-1.3/features_tsp_pad.py`

### EPIC-KITCHENS-100

**特征类型**: SlowFast或InternVideo特征
**路径**: `data/epic_kitchens-100/features/`
**大小**: 约50-200GB
**格式**: `.npy`文件
**配置**: `configs/_base_/datasets/epic_kitchens-100/features_slowfast_verb.py`

### Charades

**特征类型**: VideoMAE或I3D特征
**路径**: `data/charades/features/`
**大小**: 约5-20GB
**格式**: `.npy`文件
**配置**: `configs/_base_/datasets/charades/features_videomae_train_trunc_test_sw_s4.py`

---

## 🔽 下载方法

### 方法1: 从官方/社区分享下载（推荐）

#### THUMOS-14 I3D特征

**来源**:
- 官方提供
- 社区分享（GitHub、百度网盘等）

**下载步骤**:

```bash
# 1. 创建特征文件目录
mkdir -p /data/OpenTAD/data/thumos-14/features

# 2. 下载特征文件（假设从百度网盘下载）
cd /data/OpenTAD/data/thumos-14/features

# 3. 如果下载的是压缩包，解压
tar -xzf i3d_actionformer_stride4_thumos.tar

# 4. 验证文件结构
ls -lh i3d_actionformer_stride4_thumos/ | head -10
```

**文件结构**:
```
data/thumos-14/features/i3d_actionformer_stride4_thumos/
├── video_name_1.npy
├── video_name_2.npy
└── ...
```

#### ActivityNet-1.3 TSP特征

```bash
# 1. 创建目录
mkdir -p /data/OpenTAD/data/activitynet-1.3/features

# 2. 下载特征文件
cd /data/OpenTAD/data/activitynet-1.3/features

# 3. 解压（如果下载的是压缩包）
tar -xzf anet_tsp_npy_unresize.tar.gz

# 4. 验证
ls -lh anet_tsp_npy_unresize/ | head -10
```

### 方法2: 自己提取特征（如果下载不到）

如果无法下载预提取的特征文件，可以自己从原始视频提取：

#### 使用I3D提取特征

```bash
# 1. 安装依赖
pip install mmaction2

# 2. 下载I3D预训练模型
# 3. 使用提取脚本提取特征
python tools/extract_features.py \
    --config configs/recognition/i3d/i3d_r50_32x2x1_100e_kinetics400_rgb.py \
    --checkpoint pretrained/i3d_r50_256p_32x2x1_100e_kinetics400_rgb_20200801-aa2c523f.pth \
    --data-path data/thumos-14/raw_data/video \
    --output-path data/thumos-14/features/i3d/
```

#### 使用VideoMAE提取特征

```bash
python tools/extract_videomae_features.py \
    --model pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth \
    --data-path data/thumos-14/raw_data/video \
    --output-path data/thumos-14/features/videomae/
```

---

## 📋 完整下载流程示例

### 示例1: 下载THUMOS-14 I3D特征

```bash
# 1. 创建数据目录（如果使用/data分区）
mkdir -p /data/OpenTAD/data/thumos-14/features
cd /data/OpenTAD/data/thumos-14/features

# 2. 下载特征文件（从百度网盘或其他来源）
# 假设文件名为: i3d_actionformer_stride4_thumos.tar
# 使用bypy或其他工具下载

# 3. 解压
tar -xzf i3d_actionformer_stride4_thumos.tar

# 4. 验证文件
ls -lh i3d_actionformer_stride4_thumos/ | wc -l
# 应该看到很多.npy文件

# 5. 检查文件大小
du -sh i3d_actionformer_stride4_thumos/
# 应该约10-50GB
```

### 示例2: 下载ActivityNet-1.3 TSP特征

```bash
# 1. 创建目录
mkdir -p /data/OpenTAD/data/activitynet-1.3/features
cd /data/OpenTAD/data/activitynet-1.3/features

# 2. 下载特征文件
# 文件名可能为: anet_tsp_npy_unresize.tar.gz

# 3. 解压
tar -xzf anet_tsp_npy_unresize.tar.gz

# 4. 验证
ls -lh anet_tsp_npy_unresize/ | head -10
du -sh anet_tsp_npy_unresize/
```

---

## 🔗 特征文件来源

### 官方来源

1. **THUMOS Challenge官网**: http://www.thumos.info/
2. **ActivityNet官网**: http://activity-net.org/
3. **EPIC-KITCHENS官网**: https://epic-kitchens.github.io/

### 社区分享

1. **GitHub**: 搜索 "thumos features" 或 "activitynet features"
2. **百度网盘**: 社区分享的链接
3. **Google Drive**: 研究团队分享
4. **学术论文**: 论文作者通常会提供特征文件下载链接

---

## 📁 目录结构要求

### THUMOS-14

```
data/thumos-14/
├── annotations/
│   ├── thumos_14_anno.json
│   └── category_idx.txt
└── features/
    └── i3d_actionformer_stride4_thumos/
        ├── video_validation_0000001.npy
        ├── video_validation_0000002.npy
        └── ...
```

### ActivityNet-1.3

```
data/activitynet-1.3/
├── annotations/
│   ├── activity_net.v1-3.min.json
│   └── category_idx.txt
└── features/
    └── anet_tsp_npy_unresize/
        ├── v_---.npy
        ├── v_---.npy
        └── ...
```

---

## ✅ 验证特征文件

### 检查文件完整性

```bash
# 1. 检查文件数量
find data/thumos-14/features/i3d_actionformer_stride4_thumos/ -name "*.npy" | wc -l

# 2. 检查文件大小（应该大致相同）
ls -lh data/thumos-14/features/i3d_actionformer_stride4_thumos/*.npy | head -10

# 3. 尝试加载一个特征文件
python3 << 'EOF'
import numpy as np
feat = np.load('data/thumos-14/features/i3d_actionformer_stride4_thumos/video_validation_0000001.npy')
print(f"特征形状: {feat.shape}")
print(f"特征类型: {feat.dtype}")
print("✅ 特征文件可以正常加载")
EOF
```

---

## 🚀 使用特征文件训练

### 修改配置文件

使用特征文件时，需要引用特征文件的配置：

```python
# 使用THUMOS-14 I3D特征
_base_ = [
    "../../_base_/datasets/thumos-14/features_i3d_pad.py",  # 注意：使用features配置
    "../../_base_/models/actionformer.py",
]

# 其他配置保持不变
model = dict(...)
solver = dict(...)
```

### 关键区别

**原始视频配置** (`e2e_train_trunc_test_sw_256x224x224.py`):
- `data_path = "data/thumos-14/raw_data/video"`
- pipeline包含: `DecordDecode`, `Resize`, `Crop`等视频处理

**特征文件配置** (`features_i3d_pad.py`):
- `data_path = "data/thumos-14/features/i3d_actionformer_stride4_thumos/"`
- pipeline包含: `LoadFeats`（加载.npy文件）

---

## 📊 特征文件大小对比

| 数据集 | 原始视频 | I3D特征 | TSP特征 | VideoMAE特征 |
|--------|---------|---------|---------|-------------|
| THUMOS-14 | 100-200GB | 10-50GB | - | 20-100GB |
| ActivityNet-1.3 | 500GB-1TB | 50-200GB | 50-200GB | 100-300GB |
| EPIC-KITCHENS | 500GB-1TB | 50-200GB | - | 100-300GB |
| Charades | 50-100GB | 5-20GB | - | 10-30GB |

---

## 💡 推荐下载流程

### 步骤1: 确定需要的特征类型

根据您的模型选择：
- **I3D特征**: 适合大多数模型
- **TSP特征**: 适合TimeSformer相关模型
- **VideoMAE特征**: 适合VideoMAE backbone
- **SlowFast特征**: 适合SlowFast模型

### 步骤2: 查找特征文件来源

1. 检查OpenTAD项目README
2. 查看数据集官网
3. 搜索GitHub/社区分享
4. 联系论文作者

### 步骤3: 下载并验证

```bash
# 下载 → 解压 → 验证 → 使用
```

### 步骤4: 更新配置文件

使用对应的`features_*.py`配置文件

---

## ⚠️ 注意事项

1. **特征版本匹配**: 确保特征文件与配置文件的版本匹配
2. **文件命名**: 特征文件名需要与标注文件中的视频名对应
3. **特征维度**: 不同特征提取器的维度可能不同
4. **存储位置**: 建议将特征文件放在`/data`分区以节省空间

---

## 🔧 故障排除

### Q1: 特征文件加载失败
**A**: 检查文件路径、文件名格式、文件完整性

### Q2: 特征维度不匹配
**A**: 确保使用的特征类型与模型配置匹配

### Q3: 找不到特征文件
**A**: 检查`data_path`配置是否正确，文件是否已下载

---

## 📝 总结

1. **优先使用特征文件**: 节省存储空间和训练时间
2. **从官方/社区下载**: 比自己提取更方便
3. **验证文件完整性**: 下载后务必验证
4. **使用正确的配置**: 引用`features_*.py`配置文件

当前您已有THUMOS-14的I3D特征文件（在`/data/thumos-14-features/`），可以直接使用！

