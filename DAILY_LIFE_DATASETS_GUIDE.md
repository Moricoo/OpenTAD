# 日常生活类动作定位数据集推荐指南

## 📊 适合日常生活类动作的数据集

根据您的需求（日常生活类动作定位），以下是OpenTAD框架支持的最佳数据集选择：

---

## 🥇 推荐1: ActivityNet-1.3（最推荐）⭐⭐⭐⭐⭐

### 数据集特点

- **动作类别数**: **200类**
- **视频数量**: 约20,000个视频
- **场景**: 广泛的日常生活场景
- **动作类型**:
  - 日常活动（做饭、清洁、运动等）
  - 社交活动（聚会、会议等）
  - 娱乐活动（游戏、表演等）
  - 工作活动（办公、手工等）
  - 体育活动（但比THUMOS-14更日常化）

### 为什么推荐

✅ **类别丰富**: 200个类别覆盖大部分日常活动
✅ **场景多样**: 家庭、办公室、户外等多种场景
✅ **视频质量高**: 来自YouTube，质量较好
✅ **标注完整**: 时序边界标注准确
✅ **广泛应用**: 业界标准数据集，研究充分

### 配置文件位置

```bash
configs/_base_/datasets/activitynet-1.3/e2e_resize_768_1x224x224.py
```

### 使用示例

```python
# 在配置文件中引用
_base_ = [
    "../../_base_/datasets/activitynet-1.3/e2e_resize_768_1x224x224.py",
    "../../_base_/models/actionformer.py",
]
```

### 数据准备

```bash
# 数据目录结构
data/activitynet-1.3/
├── annotations/
│   ├── activity_net.v1-3.min.json  # 标注文件
│   └── category_idx.txt            # 类别映射
└── raw_data/
    └── Anet_videos_15fps_short256/  # 视频文件
```

---

## 🥈 推荐2: EPIC-KITCHENS-100（厨房场景专用）⭐⭐⭐⭐

### 数据集特点

- **动作类别数**:
  - **100个动词** (verb) - 如：take, put, open, close等
  - **300个名词** (noun) - 如：cup, plate, knife等
- **视频数量**: 约55,000个视频片段
- **场景**: **第一人称视角的厨房活动**
- **特点**:
  - 第一人称视角（ego-centric）
  - 细粒度动作（如"拿起杯子"、"打开冰箱"）
  - 真实日常厨房场景

### 为什么推荐

✅ **真实场景**: 第一人称视角，贴近实际应用
✅ **细粒度**: 动作划分细致，适合精细检测
✅ **日常化**: 完全聚焦日常生活场景
✅ **双任务**: 可同时检测动词和名词

### 适用场景

- 智能家居中的厨房活动监控
- 烹饪教学视频分析
- 日常活动辅助系统
- 第一人称视角视频分析

### 配置文件位置

```bash
# 动词检测
configs/_base_/datasets/epic_kitchens-100/e2e_verb_train_trunc_test_sw_s16_768x1_224.py

# 名词检测
configs/_base_/datasets/epic_kitchens-100/e2e_noun_train_trunc_test_sw_s16_768x1_224.py
```

### 使用示例

```python
# 动词检测配置
_base_ = [
    "../../_base_/datasets/epic_kitchens-100/e2e_verb_train_trunc_test_sw_s16_768x1_224.py",
    "../../_base_/models/actionformer.py",
]

# 名词检测配置
_base_ = [
    "../../_base_/datasets/epic_kitchens-100/e2e_noun_train_trunc_test_sw_s16_768x1_224.py",
    "../../_base_/models/actionformer.py",
]
```

---

## 🥉 推荐3: Charades（日常活动）⭐⭐⭐

### 数据集特点

- **动作类别数**: **157类**
- **视频数量**: 约10,000个视频
- **场景**: 家庭日常活动
- **特点**:
  - 多动作实例（每个视频平均6.8个动作）
  - 动作可能重叠
  - 真实家庭场景

### 为什么推荐

✅ **家庭场景**: 完全聚焦家庭日常活动
✅ **多实例**: 适合检测视频中的多个动作
✅ **真实数据**: 用户自己拍摄的真实场景

### 配置文件位置

```bash
configs/_base_/datasets/charades/e2e_train_trunc_test_sw_s4_512x1_224.py
```

---

## 📋 数据集对比

| 数据集 | 类别数 | 视频数 | 场景 | 视角 | 推荐度 |
|--------|--------|--------|------|------|--------|
| **ActivityNet-1.3** | 200 | ~20K | 广泛日常 | 第三人称 | ⭐⭐⭐⭐⭐ |
| **EPIC-KITCHENS** | 100动词+300名词 | ~55K | 厨房 | 第一人称 | ⭐⭐⭐⭐ |
| **Charades** | 157 | ~10K | 家庭 | 第三人称 | ⭐⭐⭐ |
| **THUMOS-14** | 20 | ~20K | 体育 | 第三人称 | ⭐ (不推荐) |

---

## 🚀 快速开始：使用ActivityNet-1.3训练

### 步骤1: 准备数据

```bash
# 1. 下载ActivityNet-1.3数据集
# 访问: http://activity-net.org/

# 2. 组织数据目录
mkdir -p data/activitynet-1.3/raw_data
mkdir -p data/activitynet-1.3/annotations

# 3. 放置文件
# - 视频文件放到: data/activitynet-1.3/raw_data/Anet_videos_15fps_short256/
# - 标注文件放到: data/activitynet-1.3/annotations/activity_net.v1-3.min.json
```

### 步骤2: 创建训练配置

基于您的THUMOS配置，创建ActivityNet版本：

```python
# configs/adatad/activitynet/e2e_activitynet_videomae_s_768x1_160_adapter.py

_base_ = [
    "../../_base_/datasets/activitynet-1.3/e2e_resize_768_1x224x224.py",
    "../../_base_/models/actionformer.py",
]

window_size = 768
scale_factor = 1
chunk_num = window_size * scale_factor // 16

dataset = dict(
    train=dict(
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4", prefix="v_"),
            dict(type="mmaction.DecordInit", num_threads=12),
            dict(type="LoadFrames", num_clips=1, method="resize"),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 160)),
            dict(type="mmaction.CenterCrop", crop_size=160),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    # ... val和test配置类似
)

model = dict(
    backbone=dict(
        type="mmaction.Recognizer3D",
        backbone=dict(
            type="VisionTransformerAdapter",
            img_size=224,
            patch_size=16,
            embed_dims=384,
            depth=12,
            num_heads=6,
            mlp_ratio=4,
            qkv_bias=True,
            num_frames=16,
            drop_path_rate=0.1,
            norm_cfg=dict(type="LN", eps=1e-6),
            return_feat_map=True,
            with_cp=True,
            total_frames=window_size * scale_factor,
            adapter_index=list(range(12)),
        ),
        # ... 其他配置
    ),
    projection=dict(
        in_channels=384,
        max_seq_len=window_size,
        attn_cfg=dict(n_mha_win_size=-1),
    ),
)

# ... 其他配置（optimizer, scheduler等）
```

### 步骤3: 开始训练

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 tools/train.py \
    configs/adatad/activitynet/e2e_activitynet_videomae_s_768x1_160_adapter.py
```

---

## 💡 选择建议

### 如果您的应用场景是：

1. **通用日常生活场景** → 选择 **ActivityNet-1.3**
   - 覆盖范围最广
   - 200个类别
   - 适合大多数应用

2. **厨房/烹饪相关** → 选择 **EPIC-KITCHENS**
   - 第一人称视角
   - 细粒度动作
   - 真实厨房场景

3. **家庭场景监控** → 选择 **Charades**
   - 家庭日常活动
   - 多动作实例检测

4. **需要快速验证** → 可以先用 **ActivityNet-1.3** 的子集训练

---

## 📚 数据集下载链接

- **ActivityNet-1.3**: http://activity-net.org/
- **EPIC-KITCHENS-100**: https://epic-kitchens.github.io/
- **Charades**: https://prior.allenai.org/projects/charades

---

## ⚠️ 注意事项

1. **数据量**: ActivityNet和EPIC-KITCHENS数据量较大，需要足够的存储空间
2. **标注格式**: 不同数据集的标注格式可能略有不同，需要适配
3. **类别映射**: 如果您的应用场景有特定类别，可能需要：
   - 选择最接近的数据集
   - 或者进行迁移学习（从ActivityNet预训练，然后在您的数据上微调）

---

## 🔄 迁移学习建议

如果您已经有THUMOS-14训练的模型，可以：

1. **使用ActivityNet继续训练**（迁移学习）
   - 从THUMOS checkpoint开始
   - 只训练adapter部分
   - 或全模型微调

2. **多数据集联合训练**
   - 同时使用ActivityNet和EPIC-KITCHENS
   - 提高模型泛化能力

---

## 📝 总结

**对于日常生活类动作定位，强烈推荐使用 ActivityNet-1.3**：

✅ 200个类别，覆盖广泛
✅ 场景多样，贴近实际应用
✅ 数据质量高，标注准确
✅ OpenTAD框架已支持，配置简单

如果需要更细粒度的厨房场景，可以额外使用EPIC-KITCHENS进行训练或微调。

