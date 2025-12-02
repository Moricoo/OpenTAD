# LoRA模型准确率提升指南

## 当前结果
- **Average-mAP**: 50.05%
- **mAP@0.3**: 70.25%
- **mAP@0.5**: 52.16%

## 优化策略（按优先级排序）

### 1. LoRA参数调整 ⭐⭐⭐⭐⭐
**影响**: 高 | **难度**: 低 | **预期提升**: +2-5%

#### 当前配置
```python
lora_r=16,        # rank
lora_alpha=32,    # alpha
lora_dropout=0.1  # dropout
```

#### 优化方案A：增加模型容量
```python
lora_r=32,        # 增加rank，提升模型表达能力
lora_alpha=64,    # 保持alpha/r=2的比例
lora_dropout=0.05 # 降低dropout，减少正则化
```
**预期**: 提升2-3%，但训练时间增加约20%

#### 优化方案B：精细调整
```python
lora_r=24,        # 中等rank
lora_alpha=48,    # alpha/r=2
lora_dropout=0.08 # 适中的dropout
```
**预期**: 提升1-2%，训练时间增加约10%

### 2. 学习率和训练策略 ⭐⭐⭐⭐
**影响**: 高 | **难度**: 低 | **预期提升**: +1-3%

#### 当前配置
```python
lr=2e-4,          # adapter/lora学习率
warmup_epoch=5,   # warmup轮数
max_epoch=60,     # 总训练轮数
```

#### 优化方案
```python
lr=3e-4,          # 提高学习率，加快收敛
warmup_epoch=10,  # 增加warmup，稳定训练
max_epoch=120,    # 增加训练轮数，充分训练
```
**预期**: 提升1-3%，需要更长的训练时间

### 3. 训练更多Epoch ⭐⭐⭐
**影响**: 中 | **难度**: 低 | **预期提升**: +1-2%

- 从60 epoch增加到100-120 epoch
- 使用学习率衰减策略
- 监控验证集性能，避免过拟合

### 4. 数据增强优化 ⭐⭐⭐
**影响**: 中 | **难度**: 中 | **预期提升**: +1-2%

#### 可尝试的增强
```python
# 在train pipeline中添加
dict(type="mmaction.RandAugment", num_ops=2, magnitude=9),  # RandAugment
dict(type="mmaction.Mixup", alpha=0.2),  # Mixup增强
dict(type="mmaction.CutMix", alpha=1.0),  # CutMix增强
```

### 5. 后处理参数调优 ⭐⭐
**影响**: 中 | **难度**: 低 | **预期提升**: +0.5-1.5%

#### 当前配置
```python
sigma=0.7,           # soft-NMS参数
max_seg_num=2000,    # 最大segment数量
voting_thresh=0.7,   # voting阈值
```

#### 优化方案
```python
sigma=0.6,           # 降低sigma，更严格的NMS
max_seg_num=2500,    # 增加候选数量
voting_thresh=0.65,  # 降低阈值，保留更多预测
```

### 6. 模型结构微调 ⭐⭐
**影响**: 中 | **难度**: 高 | **预期提升**: +1-2%

#### 选项A：增加Adapter层数
```python
adapter_index=list(range(12)),  # 当前：所有12层
# 可以尝试只在前几层或后几层使用
```

#### 选项B：调整drop_path_rate
```python
drop_path_rate=0.15,  # 从0.1增加到0.15，增强正则化
```

### 7. 集成学习 ⭐
**影响**: 高 | **难度**: 高 | **预期提升**: +2-4%

- 训练多个不同配置的模型
- 在推理时进行模型集成
- 需要更多计算资源

## 推荐实施顺序

### 阶段1：快速优化（1-2天）
1. ✅ 使用v2配置文件（已创建）
   - 增加LoRA rank到32
   - 提高学习率到3e-4
   - 增加训练epoch到120
   - 增加warmup到10

### 阶段2：精细调优（3-5天）
2. 调整后处理参数
3. 尝试不同的LoRA配置组合
4. 优化数据增强策略

### 阶段3：高级优化（1周+）
5. 模型结构实验
6. 集成学习
7. 超参数网格搜索

## 使用v2配置启动训练

```bash
cd /root/OpenTAD
source /root/miniconda3/bin/activate opentad
CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --nproc_per_node=1 \
    --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
    tools/train.py configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter_lora_v2.py
```

## 监控指标

训练过程中关注：
- **训练loss**: 应该持续下降
- **验证mAP**: 每2个epoch评估一次
- **过拟合**: 如果验证mAP不再提升，考虑早停
- **学习率**: 观察学习率衰减曲线

## 预期结果

- **阶段1优化后**: 52-55% Average-mAP
- **阶段2优化后**: 54-57% Average-mAP
- **阶段3优化后**: 56-60% Average-mAP

## 注意事项

1. **显存限制**: 增加LoRA rank会增加显存占用，注意调整batch_size
2. **训练时间**: 更多epoch意味着更长的训练时间
3. **过拟合**: 监控验证集性能，及时调整
4. **实验记录**: 记录每次调整的效果，便于对比

