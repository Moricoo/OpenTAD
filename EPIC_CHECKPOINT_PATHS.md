# EPIC-KITCHENS AdaTAD 已训练权重路径

## 📦 已训练好的AdaTAD权重

### Verb模型（动作检测）

- **下载链接**: https://drive.google.com/file/d/16Hq3sHu0S97Ge2AewHT6DOaHSo0TqIlx/view?usp=sharing
- **保存路径**: `/root/OpenTAD/pretrained/adatad/adatad_epic_verb.pth`
- **配置文件**: `configs/adatad/epic/e2e_epic_videomae_l_ft_768x8_160_adapter_verb.py`
- **性能**: 
  - mAP@0.1: 33.02%
  - mAP@0.2: 32.43%
  - mAP@0.3: 30.51%
  - mAP@0.4: 27.80%
  - mAP@0.5: 24.69%
  - ave. mAP: 29.69%

### Noun模型（物体检测）

- **下载链接**: https://drive.google.com/file/d/17k3f6wirqniLTjKOsIXbfqJPA_iLb88E/view?usp=sharing
- **保存路径**: `/root/OpenTAD/pretrained/adatad/adatad_epic_noun.pth`
- **配置文件**: `configs/adatad/epic/e2e_epic_videomae_l_ft_768x8_160_adapter_noun.py`
- **性能**:
  - mAP@0.1: 33.88%
  - mAP@0.2: 32.41%
  - mAP@0.3: 30.58%
  - mAP@0.4: 27.66%
  - mAP@0.5: 22.67%
  - ave. mAP: 29.44%

## 📥 下载方法

### 方法1: 使用gdown

```bash
mkdir -p /root/OpenTAD/pretrained/adatad
cd /root/OpenTAD/pretrained/adatad

pip install gdown

# 下载Verb模型
gdown https://drive.google.com/uc?id=16Hq3sHu0S97Ge2AewHT6DOaHSo0TqIlx \
    -O adatad_epic_verb.pth

# 下载Noun模型
gdown https://drive.google.com/uc?id=17k3f6wirqniLTjKOsIXbfqJPA_iLb88E \
    -O adatad_epic_noun.pth
```

### 方法2: 手动下载

1. 在浏览器中打开Google Drive链接
2. 下载文件到本地
3. 使用scp上传：

```bash
# 在本地电脑执行
scp adatad_epic_verb.pth root@<服务器IP>:/root/OpenTAD/pretrained/adatad/
scp adatad_epic_noun.pth root@<服务器IP>:/root/OpenTAD/pretrained/adatad/
```

## 🔍 验证下载

```bash
# 检查文件是否存在
ls -lh /root/OpenTAD/pretrained/adatad/adatad_epic_*.pth

# 应该看到：
# adatad_epic_verb.pth  (~2GB)
# adatad_epic_noun.pth  (~2GB)
```

## 🚀 使用已训练权重进行推理

```bash
# Verb检测
CUDA_VISIBLE_DEVICES=0 torchrun \
    --nnodes=1 --nproc_per_node=1 \
    --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
    tools/test.py \
    configs/adatad/epic/e2e_epic_videomae_l_ft_768x8_160_adapter_verb.py \
    --checkpoint /root/OpenTAD/pretrained/adatad/adatad_epic_verb.pth

# Noun检测
CUDA_VISIBLE_DEVICES=0 torchrun \
    --nnodes=1 --nproc_per_node=1 \
    --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
    tools/test.py \
    configs/adatad/epic/e2e_epic_videomae_l_ft_768x8_160_adapter_noun.py \
    --checkpoint /root/OpenTAD/pretrained/adatad/adatad_epic_noun.pth
```
