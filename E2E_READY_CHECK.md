# 端到端模型准备就绪检查

## ✅ 预训练模型

**状态**: ✓ 已准备完成

- **文件**: `pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth`
- **大小**: 87 MB
- **路径**: 与配置文件中的路径匹配 ✓

## ⚠️ 原始视频数据

**状态**: 需要准备

端到端训练需要原始视频文件，而不是预提取的特征。

### 准备步骤

1. **下载 THUMOS-14 视频**:
   - 参考: `tools/prepare_data/thumos/README.md`
   - 需要下载验证集和测试集的原始视频

2. **组织视频文件**:
   ```
   data/thumos-14/videos/
   ├── validation/
   │   ├── video_validation_0000001.mp4
   │   ├── video_validation_0000002.mp4
   │   └── ...
   └── test/
       ├── video_test_0000001.mp4
       ├── video_test_0000002.mp4
       └── ...
   ```

3. **检查数据**:
   ```bash
   # 检查视频目录
   ls -lh data/thumos-14/videos/validation/ | head -10
   ls -lh data/thumos-14/videos/test/ | head -10
   
   # 检查视频文件数量
   find data/thumos-14/videos/ -name "*.mp4" | wc -l
   ```

## 🚀 开始训练

一旦视频数据准备好，就可以开始训练：

```bash
cd /root/OpenTAD
bash scripts/training/run_e2e_minimal.sh
```

## 📋 检查清单

- [x] 预训练模型已下载并放在正确位置
- [ ] 原始视频数据已准备
- [ ] 视频文件路径正确
- [ ] 注释文件已准备（应该已经有了，因为 baseline 已经跑通）

## 💡 提示

如果暂时没有原始视频数据，可以：

1. **先测试 baseline 模型**: 使用已有的特征进行训练和测试
2. **准备视频数据**: 下载 THUMOS-14 原始视频
3. **再运行端到端训练**: 视频数据准备好后再开始

## 📚 相关文档

- 数据准备: `tools/prepare_data/thumos/README.md`
- 训练指南: `E2E_MINIMAL_GUIDE.md`
- 预训练模型下载: `PRETRAINED_MODEL_DOWNLOAD.md`


