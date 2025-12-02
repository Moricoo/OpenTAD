# QLoRA训练日志查看指南

## 📁 日志文件位置

### 1. 主要日志文件（JSON格式，推荐）

**路径**:
```
exps/thumos/adatad/e2e_actionformer_videomae_s_768x1_160_adapter_qlora/gpu1_id0/log.json
```

**特点**:
- JSON格式，每行一条记录
- 包含训练和验证的详细信息
- 文件大小: ~56 KB
- 总记录数: ~1065 条

### 2. 临时训练日志（文本格式）

**路径**:
```
/tmp/qlora_training.log
```

**特点**:
- 文本格式，包含完整的训练输出
- 包括INFO、WARNING、ERROR等所有信息

### 3. 检查点目录

**路径**:
```
exps/thumos/adatad/e2e_actionformer_videomae_s_768x1_160_adapter_qlora/gpu1_id0/checkpoint/
```

**内容**:
- 每个epoch的模型检查点（.pth文件）
- 文件大小: ~586 MB/个
- 最新检查点: epoch_41.pth

---

## 📊 查看日志的方法

### 方法1: 使用Python脚本查看

```python
import json

log_path = "exps/thumos/adatad/e2e_actionformer_videomae_s_768x1_160_adapter_qlora/gpu1_id0/log.json"

with open(log_path, 'r') as f:
    lines = f.readlines()

# 查看最后5条记录
for line in lines[-5:]:
    data = json.loads(line.strip())
    print(data)
```

### 方法2: 使用命令行工具

```bash
# 查看最后10条记录
tail -10 exps/thumos/adatad/e2e_actionformer_videomae_s_768x1_160_adapter_qlora/gpu1_id0/log.json

# 统计总记录数
wc -l exps/thumos/adatad/e2e_actionformer_videomae_s_768x1_160_adapter_qlora/gpu1_id0/log.json

# 查看文件大小
ls -lh exps/thumos/adatad/e2e_actionformer_videomae_s_768x1_160_adapter_qlora/gpu1_id0/log.json
```

### 方法3: 提取训练统计信息

```python
import json

log_path = "exps/thumos/adatad/e2e_actionformer_videomae_s_768x1_160_adapter_qlora/gpu1_id0/log.json"

with open(log_path, 'r') as f:
    lines = f.readlines()

# 提取所有epoch信息
epochs = {}
for line in lines:
    try:
        data = json.loads(line.strip())
        epoch = data.get('epoch')
        if epoch is not None:
            if epoch not in epochs:
                epochs[epoch] = {}
            epochs[epoch].update(data)
    except:
        pass

# 显示训练统计
for epoch in sorted(epochs.keys()):
    info = epochs[epoch]
    loss = info.get('loss', 'N/A')
    mAP = info.get('mAP', 'N/A')
    print(f"Epoch {epoch}: Loss={loss}, mAP={mAP}")
```

---

## 🔍 日志内容说明

### 训练记录字段

- `epoch`: Epoch编号
- `loss`: 总损失
- `cls_loss`: 分类损失
- `reg_loss`: 回归损失
- `lr_backbone`: Backbone学习率
- `lr_det`: Detector学习率
- `mem`: 显存使用量（MB）

### 验证记录字段

- `epoch`: Epoch编号
- `mAP`: 平均精度（Mean Average Precision）
- 其他评估指标

---

## 📈 训练进度查看

### 查看最新检查点

```bash
ls -lht exps/thumos/adatad/e2e_actionformer_videomae_s_768x1_160_adapter_qlora/gpu1_id0/checkpoint/ | head -5
```

### 查看训练是否完成

```bash
# 检查是否有最新epoch的检查点
ls exps/thumos/adatad/e2e_actionformer_videomae_s_768x1_160_adapter_qlora/gpu1_id0/checkpoint/ | grep epoch | tail -1
```

---

## 💡 快速查看命令

```bash
# 进入训练目录
cd /root/OpenTAD

# 查看日志文件
cat exps/thumos/adatad/e2e_actionformer_videomae_s_768x1_160_adapter_qlora/gpu1_id0/log.json | tail -5 | python3 -m json.tool

# 查看检查点
ls -lh exps/thumos/adatad/e2e_actionformer_videomae_s_768x1_160_adapter_qlora/gpu1_id0/checkpoint/
```

---

## 📝 注意事项

1. **JSON格式**: log.json是JSON Lines格式，每行一个JSON对象
2. **文件大小**: 日志文件会随着训练进行而增长
3. **检查点**: 检查点文件较大（~586 MB），注意磁盘空间
4. **训练状态**: 如果训练中断，可以从最新检查点恢复

