# OpenTAD 脚本目录

本目录包含 OpenTAD 项目相关的所有脚本，按功能分类组织。

## 📁 目录结构

```
scripts/
├── training/              # 训练相关脚本
│   ├── run_baseline.sh    # 运行 baseline 训练
│   ├── run_e2e_minimal.sh # 运行最小的端到端模型
│   └── resume_training.sh # 恢复训练
├── monitoring/            # 监控相关脚本
│   └── monitor_training.sh # 监控训练状态
├── testing/              # 测试相关脚本
│   ├── test_installation.py  # 测试环境安装
│   ├── test_model_forward.py # 测试模型前向传播
│   └── test_baseline.sh      # 测试 baseline 模型
└── README.md             # 本文件
```

## 📝 脚本说明

### training/ - 训练脚本

#### `run_baseline.sh`
- **功能**: 运行 ActionFormer baseline 训练
- **用法**: `bash scripts/training/run_baseline.sh`
- **说明**: 自动检查数据并启动训练

#### `run_e2e_minimal.sh`
- **功能**: 运行最小的端到端模型 (AdaTAD with VideoMAE-S)
- **用法**: `bash scripts/training/run_e2e_minimal.sh`
- **说明**: 使用 adapter 模式，内存占用最小，适合快速实验

#### `resume_training.sh`
- **功能**: 从检查点恢复训练
- **用法**: `bash scripts/training/resume_training.sh`
- **说明**: 检查最新检查点并提供恢复训练的命令

### monitoring/ - 监控脚本

#### `monitor_training.sh`
- **功能**: 监控训练状态
- **用法**: `bash scripts/monitoring/monitor_training.sh`
- **说明**: 显示训练进程、GPU状态、训练进度和检查点信息

### testing/ - 测试脚本

#### `test_installation.py`
- **功能**: 测试 OpenTAD 环境安装
- **用法**: `python scripts/testing/test_installation.py`
- **说明**: 验证 PyTorch、CUDA、模块导入和模型构建

#### `test_model_forward.py`
- **功能**: 测试模型前向传播
- **用法**: `python scripts/testing/test_model_forward.py`
- **说明**: 使用模拟数据测试模型的前向传播和推理功能

#### `test_baseline.sh`
- **功能**: 在测试集上评估训练好的 baseline 模型
- **用法**: `bash scripts/testing/test_baseline.sh [checkpoint_path]`
- **说明**: 自动检查检查点并运行评估，默认使用 best.pth

#### `test_e2e_minimal.sh`
- **功能**: 测试最小的端到端模型
- **用法**: `bash scripts/testing/test_e2e_minimal.sh [checkpoint_path]`
- **说明**: 测试 AdaTAD with VideoMAE-S 模型

## 🚀 快速使用

### 开始训练
```bash
cd /root/OpenTAD
bash scripts/training/run_baseline.sh
```

### 监控训练
```bash
cd /root/OpenTAD
bash scripts/monitoring/monitor_training.sh
```

### 恢复训练
```bash
cd /root/OpenTAD
bash scripts/training/resume_training.sh
```

### 测试环境
```bash
cd /root/OpenTAD
source /root/miniconda3/bin/activate opentad
python scripts/testing/test_installation.py
python scripts/testing/test_model_forward.py
```

### 测试训练好的模型
```bash
cd /root/OpenTAD
bash scripts/testing/test_baseline.sh
# 或指定检查点
bash scripts/testing/test_baseline.sh exps/thumos/actionformer_i3d/gpu1_id0/checkpoint/epoch_34.pth
```

### 训练端到端模型
```bash
cd /root/OpenTAD
bash scripts/training/run_e2e_minimal.sh
```

### 测试端到端模型
```bash
cd /root/OpenTAD
bash scripts/testing/test_e2e_minimal.sh
```

## ⚠️ 注意事项

- 所有脚本都假设在 `/root/OpenTAD` 目录下运行
- 训练和测试脚本需要先激活 conda 环境：`source /root/miniconda3/bin/activate opentad`
- 脚本中的路径都是绝对路径，可以从任何位置运行
- 测试脚本已修复路径问题，可以从 `scripts/testing/` 目录直接运行
