# CPU/GPU负载优化指南

## 问题分析

**当前情况**：
- 2张GPU卡（Tesla V100 32GB）
- 16个CPU核心
- CPU成为瓶颈，GPU空闲等待数据

**原因**：
1. 视频解码和预处理占用大量CPU资源
2. 单GPU训练，CPU负载集中在数据加载
3. num_workers配置可能不够优化

## 优化方案

### 方案1：双GPU训练（推荐）⭐⭐⭐⭐⭐

**优势**：
- 充分利用2张GPU
- CPU负载分散到2个进程
- 训练速度提升约1.8-1.9倍

**配置**：
```bash
# 使用2个GPU训练
CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes=1 --nproc_per_node=2 \
    --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
    tools/train.py configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter_lora_v2.py
```

**参数调整**：
- 每个GPU: `num_workers=6`（总共12个workers）
- `persistent_workers=True`（减少worker启动开销）
- `prefetch_factor=4`（平衡内存和速度）

### 方案2：优化单GPU数据加载 ⭐⭐⭐

**如果只能使用单GPU**：

1. **减少视频解码线程**
   ```python
   dict(type="mmaction.DecordInit", num_threads=4),  # 从12降到4
   ```

2. **优化num_workers**
   ```python
   train=dict(batch_size=8, num_workers=8, prefetch_factor=4, persistent_workers=True)
   ```
   - 16核CPU，留8核给系统和其他任务
   - 8个workers用于数据加载

3. **使用persistent_workers**
   - 减少worker启动/关闭开销
   - 提高数据加载效率

### 方案3：混合优化 ⭐⭐⭐⭐

**结合方案1和2**：
- 双GPU训练
- 优化每个GPU的数据加载参数
- 减少不必要的CPU开销

## 已优化的配置

### v2配置已优化：
1. ✅ 减少解码线程：12 → 4
2. ✅ 优化num_workers：单GPU 6，双GPU时每个6（总共12）
3. ✅ 启用persistent_workers：减少启动开销
4. ✅ 调整prefetch_factor：平衡内存和速度

## 监控指标

训练时监控：
```bash
# CPU使用率
htop  # 或 top

# GPU使用率
watch -n 1 nvidia-smi

# 数据加载时间
# 在训练日志中查看每个batch的时间
# 如果数据加载时间 > GPU计算时间，说明CPU是瓶颈
```

## 预期效果

### 双GPU训练：
- **GPU利用率**: 从~50%提升到~90%+
- **训练速度**: 提升1.8-1.9倍
- **CPU利用率**: 更均匀分布（每个进程约50%）

### 单GPU优化：
- **GPU利用率**: 从~30-40%提升到~70-80%
- **训练速度**: 提升1.3-1.5倍
- **CPU利用率**: 更高效（减少空闲等待）

## 启动命令

### 双GPU训练（推荐）：
```bash
cd /root/OpenTAD
source /root/miniconda3/bin/activate opentad
CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes=1 --nproc_per_node=2 \
    --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
    tools/train.py configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter_lora_v2.py
```

### 单GPU优化训练：
```bash
cd /root/OpenTAD
source /root/miniconda3/bin/activate opentad
CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --nproc_per_node=1 \
    --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
    tools/train.py configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter_lora_v2.py
```

## 进一步优化建议

1. **数据预处理缓存**
   - 如果数据集不大，可以预处理并缓存
   - 减少实时解码开销

2. **使用更快的解码库**
   - 考虑使用PyAV替代Decord（如果兼容）
   - 或使用硬件加速解码

3. **减少数据增强**
   - 如果CPU仍然瓶颈，可以减少一些CPU密集的数据增强
   - 如减少ColorJitter、ImgAug等

4. **调整batch_size**
   - 如果GPU内存充足，可以增加batch_size
   - 减少数据加载频率，提高GPU利用率

## 注意事项

1. **内存使用**：
   - persistent_workers会增加内存占用
   - 如果出现OOM，可以关闭persistent_workers

2. **CPU核心分配**：
   - 16核CPU，建议：
     - 系统：2-4核
     - 数据加载：8-12核
     - 其他任务：2-4核

3. **监控资源**：
   - 定期检查CPU/GPU/内存使用情况
   - 根据实际情况调整参数

