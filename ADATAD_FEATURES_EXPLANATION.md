# AdaTAD与特征文件使用说明

## ❓ 问题：AdaTAD是端到端模型，能否仅使用特征文件？

### 简短回答

**AdaTAD本身设计为端到端模型，需要原始视频作为输入，不能直接使用预提取的特征文件。**

但是，有**两种替代方案**：

---

## 🔍 技术分析

### AdaTAD的架构特点

AdaTAD使用**VideoMAE**作为backbone，其工作流程是：

```
原始视频帧 → Patch Embedding → VideoMAE Encoder → Adapter → 特征
```

关键点：
1. **VideoMAE需要原始视频帧**：它会对视频帧进行patch分割和位置编码
2. **端到端训练**：backbone和adapter一起训练，学习视频的时空特征
3. **输入格式**：需要`NCTHW`格式的视频张量（Batch, Channels, Time, Height, Width）

### 当前配置分析

查看您的AdaTAD配置：

```python
# configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py
pipeline=[
    dict(type="PrepareVideoInfo", format="mp4"),
    dict(type="mmaction.DecordInit", num_threads=12),
    dict(type="LoadFrames", ...),  # 加载视频帧
    dict(type="mmaction.DecordDecode"),  # 解码视频
    dict(type="mmaction.Resize", ...),
    # ... 其他视频处理步骤
    dict(type="mmaction.FormatShape", input_format="NCTHW"),
]
```

这明确表明AdaTAD需要**原始视频**作为输入。

---

## 💡 解决方案

### 方案1: 使用原始视频（当前方案）✅

**优点**：
- ✅ 完全端到端训练
- ✅ 可以微调VideoMAE backbone
- ✅ 性能最佳

**缺点**：
- ❌ 需要大量存储空间（100GB-1TB）
- ❌ 训练速度较慢（需要视频解码）

**适用场景**：
- 有足够存储空间
- 需要最佳性能
- 可以微调backbone

---

### 方案2: 两阶段训练（推荐用于节省存储）⭐

**思路**：
1. **第一阶段**：使用其他模型（如I3D）提取特征，训练adapter部分
2. **第二阶段**：使用原始视频微调整个模型

**步骤**：

#### 步骤1: 使用特征文件预训练Adapter

```python
# 创建新配置文件：configs/adatad/thumos/features_thumos_videomae_s_adapter.py
_base_ = [
    "../../_base_/datasets/thumos-14/features_i3d_pad.py",  # 使用特征文件配置
    "../../_base_/models/actionformer.py",
]

# 修改模型配置，只训练adapter部分
model = dict(
    backbone=dict(
        type="mmaction.Recognizer3D",
        backbone=dict(
            type="VisionTransformerAdapter",
            # ... adapter配置
        ),
        # 注意：这里需要修改，因为特征文件已经是提取好的特征
        # 可能需要添加一个特征投影层
    ),
    projection=dict(
        in_channels=1024,  # I3D特征维度，不是384
        # ...
    ),
)
```

**问题**：这个方案比较复杂，因为：
- I3D特征维度（1024）与VideoMAE输出维度（384）不匹配
- 需要额外的投影层
- 不是真正的端到端训练

---

### 方案3: 使用轻量级视频预处理（折中方案）⭐

**思路**：
- 下载**压缩后的视频**（降低分辨率/帧率）
- 或使用**视频特征缓存**（第一次解码后缓存）

**步骤**：

```python
# 修改配置，使用更小的输入尺寸
dataset = dict(
    train=dict(
        pipeline=[
            # ...
            dict(type="mmaction.Resize", scale=(112, 112), keep_ratio=False),  # 更小的尺寸
            # ...
        ],
    ),
)
```

**优点**：
- ✅ 保持端到端训练
- ✅ 减少存储需求（约减少75%）
- ✅ 训练速度更快

**缺点**：
- ⚠️ 可能略微影响性能

---

## 📊 方案对比

| 方案 | 存储需求 | 训练速度 | 性能 | 端到端 | 推荐度 |
|------|---------|---------|------|--------|--------|
| **原始视频（当前）** | 100GB-1TB | 慢 | ⭐⭐⭐⭐⭐ | ✅ | ⭐⭐⭐⭐⭐ |
| **两阶段训练** | 10-50GB | 快 | ⭐⭐⭐ | ❌ | ⭐⭐⭐ |
| **压缩视频** | 25-250GB | 中等 | ⭐⭐⭐⭐ | ✅ | ⭐⭐⭐⭐ |

---

## 🎯 推荐方案

### 如果存储空间充足（>500GB）

**使用原始视频**（当前方案）
- 完全端到端训练
- 最佳性能
- 无需修改配置

### 如果存储空间有限（<300GB）

**使用压缩视频**（方案3）
- 降低输入分辨率（如112x112而不是160x160）
- 保持端到端训练
- 性能损失较小

### 如果存储空间非常有限（<100GB）

**考虑其他模型**：
- 使用**ActionFormer**（支持特征文件）
- 或使用**其他支持特征的模型**

---

## ⚠️ 重要说明

### AdaTAD不能直接使用特征文件的原因

1. **VideoMAE需要原始帧**：
   - VideoMAE会对视频帧进行patch分割
   - 需要空间信息（H, W）来构建patch embeddings
   - 预提取的特征已经丢失了空间结构

2. **端到端设计**：
   - AdaTAD的设计理念是端到端学习
   - Backbone和adapter需要一起训练
   - 使用预提取特征会破坏这种设计

3. **特征维度不匹配**：
   - VideoMAE输出：384维（或768/1024，取决于模型大小）
   - I3D特征：1024维
   - TSP特征：768维
   - 直接替换会导致维度不匹配

---

## 🔧 如果必须使用特征文件

### 方案：修改模型架构

如果必须使用特征文件，需要：

1. **添加特征投影层**：
```python
model = dict(
    backbone=dict(
        # 添加一个投影层，将I3D特征投影到VideoMAE维度
        feature_projection=dict(
            type="Linear",
            in_features=1024,  # I3D特征维度
            out_features=384,  # VideoMAE维度
        ),
        # ...
    ),
)
```

2. **修改数据加载**：
```python
pipeline=[
    dict(type="LoadFeats", feat_format="npy"),  # 加载特征文件
    dict(type="FeatureProjection", ...),  # 投影到VideoMAE维度
    # ... 后续处理
]
```

**但这会失去端到端的优势，不推荐。**

---

## 📝 总结

1. **AdaTAD是端到端模型**，设计上需要原始视频输入
2. **不能直接使用特征文件**，因为VideoMAE需要原始帧进行patch embedding
3. **推荐方案**：
   - 存储充足 → 使用原始视频（当前方案）
   - 存储有限 → 使用压缩视频（降低分辨率）
   - 存储非常有限 → 考虑其他支持特征的模型

4. **如果必须使用特征文件**：
   - 需要修改模型架构
   - 添加特征投影层
   - 失去端到端训练的优势

---

## 💡 实际建议

基于您的存储情况（/data分区有294GB可用）：

**推荐**：继续使用原始视频，但可以考虑：
1. 只下载一个数据集（如ActivityNet-1.3）
2. 使用压缩视频（降低分辨率到112x112）
3. 训练完成后删除原始视频，只保留checkpoint

这样可以在保持端到端训练的同时，节省存储空间。

