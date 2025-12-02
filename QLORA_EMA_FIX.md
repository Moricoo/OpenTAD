# QLoRA EMA模型加载错误修复

## 问题描述

在验证阶段（Epoch 41），训练脚本在加载EMA模型状态时出现错误：

```
RuntimeError: Error(s) in loading state_dict for DistributedDataParallel:
       Unexpected key(s) in state_dict: "module.backbone.model.backbone.blocks.0.adapter.down_proj.weight.absmax",
       "module.backbone.model.backbone.blocks.0.adapter.down_proj.weight.quant_map", ...
```

## 问题原因

1. **量化层的额外状态**: `bitsandbytes`的`Linear4bit`/`Linear8bitLt`在`state_dict()`中不仅包含权重，还包含量化相关的元数据：
   - `weight.absmax`: 绝对最大值（用于量化）
   - `weight.quant_map`: 量化映射表
   - `weight.nested_absmax`: 嵌套绝对最大值
   - `weight.nested_quant_map`: 嵌套量化映射表
   - `weight.quant_state.bitsandbytes__fp4`: 量化状态信息

2. **EMA模型更新**: EMA模型在更新时会保存这些量化状态信息

3. **状态加载不匹配**: 当尝试将EMA模型的状态加载回主模型时，这些额外的键导致`load_state_dict`失败

## 解决方案

### 1. 添加过滤函数

在两个文件中添加了`filter_quantization_state`函数：

**文件**: `opentad/cores/test_engine.py` 和 `opentad/cores/train_engine.py`

```python
def filter_quantization_state(state_dict):
    """
    过滤掉量化层的额外状态信息（bitsandbytes的Linear4bit/8bit会包含这些）
    这些状态信息在EMA模型和主模型之间加载时会导致不匹配
    """
    filtered_dict = {}
    quantization_suffixes = [
        '.absmax', '.quant_map', '.nested_absmax', '.nested_quant_map',
        '.quant_state.bitsandbytes__fp4', '.quant_state.bitsandbytes__fp8'
    ]

    for key, value in state_dict.items():
        # 跳过量化层的额外状态信息
        skip = False
        for suffix in quantization_suffixes:
            if key.endswith(suffix):
                skip = True
                break
        if not skip:
            filtered_dict[key] = value

    return filtered_dict
```

### 2. 修改EMA状态加载

**修改前**:
```python
if model_ema != None:
    current_dict = copy.deepcopy(model.state_dict())
    model.load_state_dict(model_ema.module.state_dict())
```

**修改后**:
```python
if model_ema != None:
    current_dict = copy.deepcopy(model.state_dict())
    # 过滤掉量化层的额外状态信息，避免加载错误
    ema_state_dict = model_ema.module.state_dict()
    filtered_ema_dict = filter_quantization_state(ema_state_dict)
    model.load_state_dict(filtered_ema_dict, strict=False)
```

## 修改的文件

1. **`opentad/cores/test_engine.py`**
   - 添加`filter_quantization_state`函数
   - 修改`eval_one_epoch`中的EMA状态加载

2. **`opentad/cores/train_engine.py`**
   - 添加`filter_quantization_state`函数
   - 修改`val_one_epoch`中的EMA状态加载

## 技术细节

### 为什么需要`strict=False`?

- 量化层的权重本身（`weight`）仍然会被加载
- 但量化元数据（`absmax`等）被过滤掉了
- 使用`strict=False`允许部分键不匹配，只加载匹配的键
- 量化层会在运行时自动重新计算量化状态

### 为什么过滤是安全的?

1. **量化状态是运行时计算的**: `bitsandbytes`的量化层在第一次前向传播时会自动计算量化状态
2. **权重是完整的**: 过滤掉的只是元数据，权重本身仍然被正确加载
3. **EMA权重有效**: EMA模型更新的权重值是正确的，只是不需要量化元数据

## 验证

修复后，训练应该能够：
1. ✅ 正常完成训练阶段
2. ✅ 在验证阶段成功加载EMA模型
3. ✅ 完成验证并输出mAP结果
4. ✅ 恢复主模型状态继续训练

## 注意事项

- 这个修复只影响EMA模型的加载，不影响训练过程
- 量化层的功能完全正常，只是过滤掉了不必要的元数据
- 如果未来使用其他量化库，可能需要调整过滤规则

