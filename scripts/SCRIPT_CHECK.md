# 脚本检查报告

## 检查时间
$(date)

## 已检查的脚本

### 1. `scripts/training/run_baseline.sh`
- **状态**: ✅ 已修复
- **修复内容**:
  - 添加了 exps 符号链接检查和自动修复
  - 确保训练前符号链接正确指向 `/data/opentad-exps/exps`
- **功能**: 运行 ActionFormer baseline 训练
- **用法**: `bash scripts/training/run_baseline.sh`

### 2. `scripts/training/resume_training.sh`
- **状态**: ✅ 已修复
- **修复内容**:
  - 添加了工作目录存在性检查
  - 使用 `readlink -f` 解析符号链接获取实际路径
  - 自动创建缺失的工作目录
- **功能**: 检查训练状态并提供恢复训练命令
- **用法**: `bash scripts/training/resume_training.sh`

### 3. `scripts/monitoring/monitor_training.sh`
- **状态**: ✅ 已修复
- **修复内容**:
  - 使用 `readlink -f` 解析符号链接获取实际路径
  - 确保所有路径都通过符号链接正确解析
- **功能**: 监控训练状态、GPU使用、训练进度和检查点
- **用法**: `bash scripts/monitoring/monitor_training.sh`

## 符号链接配置

- **符号链接**: `/root/OpenTAD/exps` → `/data/opentad-exps/exps`
- **训练输出**: 所有训练结果（检查点、日志等）都会自动保存到数据盘

## 验证方法

### 检查符号链接
```bash
cd /root/OpenTAD
ls -la exps
readlink exps
# 应该显示: /data/opentad-exps/exps
```

### 检查训练路径
```bash
# 训练工作目录（通过符号链接解析）
readlink -f /root/OpenTAD/exps/thumos/actionformer_i3d/gpu1_id0
# 应该显示: /data/opentad-exps/exps/thumos/actionformer_i3d/gpu1_id0
```

### 测试脚本
```bash
# 测试训练脚本（不实际运行训练）
cd /root/OpenTAD
bash scripts/training/run_baseline.sh --help 2>&1 | head -5

# 检查训练状态
bash scripts/training/resume_training.sh

# 监控训练（如果没有运行会显示已停止）
bash scripts/monitoring/monitor_training.sh
```

## 注意事项

1. **符号链接**: 所有脚本都通过符号链接自动解析到数据盘，无需手动处理路径
2. **嵌套目录**: 即使存在嵌套目录，训练脚本也能正常工作（使用相对路径）
3. **自动修复**: `run_baseline.sh` 会在训练前自动检查和修复符号链接
4. **路径解析**: 监控和恢复脚本使用 `readlink -f` 确保正确解析符号链接

## 下次运行

所有脚本已修复，可以直接使用：

```bash
# 开始训练
cd /root/OpenTAD
bash scripts/training/run_baseline.sh

# 监控训练
bash scripts/monitoring/monitor_training.sh

# 检查状态
bash scripts/training/resume_training.sh
```

