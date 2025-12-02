# AdaTAD 初始模型说明

## 📦 初始模型：VideoMAE-S

AdaTAD 使用的初始模型是 **VideoMAE-S (Video Masked Autoencoder - Small)**，这是一个在 Kinetics-400 数据集上预训练的 Vision Transformer 模型。

---

## 🔍 模型详细信息

### 模型架构

- **类型**: Vision Transformer (ViT)
- **规模**: Small
- **嵌入维度**: 384
- **Transformer 层数**: 12 层
- **注意力头数**: 6
- **MLP 比例**: 4
- **Patch 大小**: 16×16
- **位置编码**: 可学习的位置嵌入

### 预训练信息

- **预训练数据集**: Kinetics-400 (K400)
  - 包含 400 个动作类别
  - 约 30 万段视频
- **预训练方法**: VideoMAE (Video Masked Autoencoder)
  - 自监督学习方法
  - 通过掩码重建任务学习视频表示
- **预训练配置**: 16×4×1
  - 16 帧输入
  - 4 倍时间采样
  - 1 个 clip

### 模型文件

- **文件名**: `vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth`
- **位置**: `pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth`
- **大小**: 87 MB
- **参数数量**: 163 个主要参数组

---

## 🎯 在 AdaTAD 中的使用

### 1. 模型加载

```python
# 配置文件中的设置
model = dict(
    backbone=dict(
        custom=dict(
            pretrain="pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth",
            # ...
        ),
    ),
)
```

### 2. 适配器插入

- **位置**: 在每个 Transformer 层的 12 层中插入 TIA 适配器
- **配置**: `adapter_index=list(range(12))`
- **作用**: 时序信息聚合和任务适配

### 3. 训练策略

```python
optimizer = dict(
    type="AdamW",
    lr=1e-4,  # 检测头学习率
    backbone=dict(
        lr=0,  # 骨干网络学习率为 0（冻结）
        custom=[dict(name="adapter", lr=2e-4)],  # 只训练适配器
    ),
)
```

**关键点**：
- ✅ **VideoMAE-S 骨干网络**: 冻结（学习率为 0）
- ✅ **TIA 适配器**: 可训练（学习率 2e-4）
- ✅ **检测头**: 可训练（学习率 1e-4）

---

## 📊 模型参数统计

### VideoMAE-S 参数

- **总参数量**: 约 22M 参数
- **嵌入维度**: 384
- **层数**: 12
- **每层参数**:
  - 自注意力: QKV 投影、输出投影
  - MLP: 两个线性层
  - 层归一化

### TIA 适配器参数

- **每层适配器**: 轻量级模块
  - 深度卷积 (dwconv)
  - 1×1 卷积
  - 下投影和上投影
- **总适配器参数**: 远少于骨干网络（参数高效）

---

## 🔄 训练流程

### 初始化阶段

1. **加载预训练权重**
   ```
   VideoMAE-S (K400 预训练)
   ↓
   加载到 VisionTransformerAdapter
   ↓
   插入 TIA 适配器到每一层
   ```

2. **参数设置**
   - 骨干网络参数：冻结（不更新）
   - 适配器参数：随机初始化，然后训练
   - 检测头参数：随机初始化，然后训练

3. **开始训练**
   - 只更新适配器和检测头的参数
   - 骨干网络保持预训练权重不变

---

## 📚 相关资源

### VideoMAE 论文

- **标题**: VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training
- **作者**: Zhan Tong, Yibing Song, Jue Wang, Limin Wang
- **链接**: https://arxiv.org/abs/2203.12602
- **GitHub**: https://github.com/MCG-NJU/VideoMAE

### 预训练模型下载

- **VideoMAE-S**: https://drive.google.com/file/d/1BH5BZmdImaZesUfqtW23eBGC341Gui1D
- **其他规模**:
  - VideoMAE-B: mmaction2 官方链接
  - VideoMAE-L: mmaction2 官方链接
  - VideoMAE-H: Google Drive

### Kinetics-400 数据集

- **官网**: https://deepmind.com/research/open-source/kinetics
- **类别数**: 400
- **视频数**: ~30 万
- **用途**: 视频理解任务的常用预训练数据集

---

## 💡 为什么使用 VideoMAE-S？

1. **强大的视频表示能力**
   - 通过自监督学习学习到丰富的视频特征
   - 在 Kinetics-400 上预训练，具有通用性

2. **参数效率**
   - Small 版本参数量适中（22M）
   - 适合端到端训练
   - 内存占用相对较小

3. **适配器机制**
   - 冻结预训练权重，保持通用特征
   - 通过轻量级适配器适应 TAD 任务
   - 参数高效，训练快速

4. **性能表现**
   - 在 THUMOS-14 上达到 69.03% mAP
   - 平衡了性能和效率

---

## 🔬 技术细节

### 预训练任务

VideoMAE 使用掩码重建任务：
1. 随机掩码视频中的时空 patches
2. 使用未掩码的 patches 重建掩码的 patches
3. 学习视频的时空表示

### 适配器机制

TIA (Temporal-Informative Adapter)：
- **输入**: Transformer 层的特征
- **处理**:
  - 时序信息聚合（相邻帧上下文）
  - 任务特定适配
- **输出**: 适配后的特征

### 训练策略

- **冻结骨干**: 保持预训练的通用特征
- **训练适配器**: 学习任务特定的时序模式
- **端到端优化**: 适配器和检测头联合训练

---

## 📈 模型规模对比

| 模型 | 参数量 | 嵌入维度 | 层数 | 注意力头 | THUMOS-14 mAP |
|------|--------|---------|------|---------|---------------|
| VideoMAE-S | ~22M | 384 | 12 | 6 | 69.03% |
| VideoMAE-B | ~87M | 768 | 12 | 12 | 71.14% |
| VideoMAE-L | ~307M | 1024 | 24 | 16 | 73.51% |
| VideoMAE-H | ~632M | 1280 | 32 | 16 | 74.95% |

---

## ✅ 总结

AdaTAD 的初始模型是 **VideoMAE-S**，这是一个：
- ✅ 在 Kinetics-400 上预训练的 Vision Transformer
- ✅ 通过自监督学习获得强大的视频表示能力
- ✅ 通过 TIA 适配器机制适应时序动作检测任务
- ✅ 参数高效，训练快速，性能优秀

这种设计使得 AdaTAD 能够：
1. 利用大规模预训练模型的强大特征
2. 通过轻量级适配器快速适应新任务
3. 在保持性能的同时降低训练成本

