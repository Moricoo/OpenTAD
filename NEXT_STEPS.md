# Baseline 跑通后的下一步建议

恭喜您成功运行了 ActionFormer baseline！以下是您可以尝试的下一步：

## 📊 1. 在测试集上评估模型（推荐第一步）

训练完成后，在测试集上评估模型性能：

```bash
cd /root/OpenTAD
source /root/miniconda3/bin/activate opentad

# 使用最佳检查点进行评估
torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    tools/test.py \
    configs/actionformer/thumos_i3d.py \
    --checkpoint exps/thumos/actionformer_i3d/gpu1_id0/checkpoint/best.pth
```

**预期结果**（参考论文）：
- mAP@0.3: ~83.78%
- mAP@0.4: ~80.06%
- mAP@0.5: ~73.16%
- mAP@0.6: ~60.46%
- mAP@0.7: ~44.72%
- 平均 mAP: ~68.44%

## 🔬 2. 尝试其他模型架构

OpenTAD 支持多种模型，可以尝试不同的架构：

### 2.1 BMN (Boundary-Matching Network)
```bash
torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    tools/train.py \
    configs/bmn/thumos_i3d.py
```

### 2.2 GTAD (Graph Temporal Action Detection)
```bash
torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    tools/train.py \
    configs/gtad/thumos_i3d.py
```

### 2.3 TriDet (Triple Detection)
```bash
torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    tools/train.py \
    configs/tridet/thumos_i3d.py
```

### 2.4 TemporalMaxer
```bash
torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    tools/train.py \
    configs/temporalmaxer/thumos_i3d.py
```

### 2.5 TADTR (Temporal Action Detection Transformer)
```bash
torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    tools/train.py \
    configs/tadtr/thumos_i3d.py
```

## 🎯 3. 尝试其他数据集

### 3.1 ActivityNet-1.3
需要准备 ActivityNet 数据集和 TSP 特征：
```bash
# 准备数据后
torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    tools/train.py \
    configs/actionformer/anet_tsp.py
```

### 3.2 MultiTHUMOS
支持多标签动作检测：
```bash
torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    tools/train.py \
    configs/actionformer/multithumos_i3d.py
```

## ⚙️ 4. 调整超参数和配置

### 4.1 修改学习率
编辑 `configs/actionformer/thumos_i3d.py`：
```python
optimizer = dict(type="AdamW", lr=2e-4, weight_decay=0.05, paramwise=True)  # 从 1e-4 改为 2e-4
```

### 4.2 调整批次大小
```python
solver = dict(
    train=dict(batch_size=4, num_workers=4),  # 从 2 改为 4（如果 GPU 内存足够）
    ...
)
```

### 4.3 修改训练轮数
```python
scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=5, max_epoch=50)  # 从 35 改为 50
```

### 4.4 调整 NMS 参数
```python
post_processing = dict(
    nms=dict(
        use_soft_nms=True,
        sigma=0.7,  # 从 0.5 改为 0.7
        max_seg_num=3000,  # 从 2000 改为 3000
        min_score=0.0005,  # 从 0.001 改为 0.0005
        ...
    ),
)
```

## 📈 5. 分析训练结果

### 5.1 查看训练日志
```bash
# 查看训练日志
tail -f exps/thumos/actionformer_i3d/gpu1_id0/log.json

# 或者使用监控脚本
bash scripts/monitoring/monitor_training.sh
```

### 5.2 可视化损失曲线
训练日志保存在 `log.json`，可以：
- 使用 Python 脚本解析并绘制损失曲线
- 分析不同 epoch 的性能变化
- 对比不同配置的效果

### 5.3 检查检查点
```bash
# 查看所有检查点
ls -lh exps/thumos/actionformer_i3d/gpu1_id0/checkpoint/

# 查看最佳检查点
ls -lh exps/thumos/actionformer_i3d/gpu1_id0/checkpoint/best.pth
```

## 🔍 6. 进行消融实验

### 6.1 测试不同的特征提取器
- I3D (当前使用)
- TSN
- SlowFast
- VideoMAE
- InternVideo

### 6.2 测试不同的后处理策略
- Soft NMS vs Hard NMS
- 不同的 IoU 阈值
- 不同的投票阈值

## 🚀 7. 端到端训练（如果资源充足）

尝试端到端训练，从原始视频帧开始：
```bash
# 需要准备原始视频数据
torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    tools/train.py \
    configs/tadtr/e2e_thumos_tadtr_slowfast50_sw128s6.py
```

## 📝 8. 创建测试脚本

为了方便后续使用，可以创建测试脚本：

```bash
# 创建测试脚本
cat > scripts/testing/test_baseline.sh << 'EOF'
#!/bin/bash
# 测试 baseline 模型

source /root/miniconda3/bin/activate opentad
cd /root/OpenTAD

CHECKPOINT="${1:-exps/thumos/actionformer_i3d/gpu1_id0/checkpoint/best.pth}"

echo "使用检查点: $CHECKPOINT"

torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    tools/test.py \
    configs/actionformer/thumos_i3d.py \
    --checkpoint "$CHECKPOINT"
EOF

chmod +x scripts/testing/test_baseline.sh
```

使用：
```bash
bash scripts/testing/test_baseline.sh
# 或指定检查点
bash scripts/testing/test_baseline.sh exps/thumos/actionformer_i3d/gpu1_id0/checkpoint/epoch_34.pth
```

## 🎓 9. 学习资源

- **论文**: 阅读 ActionFormer 原始论文了解模型原理
- **代码**: 研究 `opentad/models/detectors/actionformer.py` 了解实现细节
- **配置**: 查看不同配置文件的差异，理解各参数作用

## 💡 10. 实验建议

1. **记录实验**: 为每次实验创建独立的配置文件和输出目录
2. **对比实验**: 系统性地对比不同模型和配置
3. **错误分析**: 分析模型在哪些类别上表现较差
4. **可视化**: 可视化检测结果，理解模型的预测行为

## 📚 相关文档

- 模型 README: `configs/actionformer/README.md`
- 使用文档: `docs/en/usage.md`
- 其他模型配置: `configs/` 目录下各模型的 README

## ⚠️ 注意事项

1. **GPU 内存**: 不同模型和配置的显存需求不同
2. **训练时间**: 某些模型（如端到端）训练时间较长
3. **数据准备**: 尝试新数据集前需要先准备相应的数据和特征
4. **检查点**: 定期保存检查点，避免训练中断丢失进度

祝您实验顺利！🎉

