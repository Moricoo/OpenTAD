# EPIC-KITCHENS AdaTAD 下载清单

## 📋 下载清单

### ✅ 必需文件（约6.4GB）

#### 1. EPIC-pretrained VideoMAE权重

- [ ] `vit-large-p16_videomae-epic_verb.pth` (~1.2GB)
  - 链接: https://drive.google.com/file/d/1h7oLiNN5LTXau4HWmmzS_ekvuNdZkp-b/view?usp=sharing
  - 保存到: `/root/OpenTAD/pretrained/`

- [ ] `vit-large-p16_videomae-epic_noun.pth` (~1.2GB)
  - 链接: https://drive.google.com/file/d/1nRuzJI4ej90vFsKCBSugRVOmxrR8urwW/view?usp=sharing
  - 保存到: `/root/OpenTAD/pretrained/`

#### 2. AdaTAD训练好的模型权重

- [ ] `adatad_epic_verb.pth` (~2GB)
  - 链接: https://drive.google.com/file/d/16Hq3sHu0S97Ge2AewHT6DOaHSo0TqIlx/view?usp=sharing
  - 保存到: `/root/OpenTAD/pretrained/adatad/`

- [ ] `adatad_epic_noun.pth` (~2GB)
  - 链接: https://drive.google.com/file/d/17k3f6wirqniLTjKOsIXbfqJPA_iLb88E/view?usp=sharing
  - 保存到: `/root/OpenTAD/pretrained/adatad/`

#### 3. EPIC-KITCHENS标注文件

- [ ] 运行: `cd /root/OpenTAD/tools/prepare_data/epic && bash download_annotation.sh`
  - 保存到: `data/epic_kitchens-100/annotations/`

### ⚠️ 可选文件（仅训练需要）

- [ ] EPIC-KITCHENS-100原始视频 (~500GB-1TB)
  - 从官网下载: https://github.com/epic-kitchens/epic-kitchens-download-scripts
  - 保存到: `data/epic_kitchens-100/raw_data/`

## 🔍 验证命令

```bash
# 检查预训练权重
ls -lh /root/OpenTAD/pretrained/vit-large-p16_videomae-epic_*.pth

# 检查AdaTAD模型
ls -lh /root/OpenTAD/pretrained/adatad/adatad_epic_*.pth

# 检查标注文件
ls -lh /root/OpenTAD/data/epic_kitchens-100/annotations/
```

## 📥 下载方法

### 方法1: 使用gdown（如果网络可用）

```bash
pip install gdown
cd /root/OpenTAD
./download_epic_adatad.sh
```

### 方法2: 手动下载（推荐）

1. 在浏览器中打开Google Drive链接
2. 下载文件到本地
3. 使用scp上传到服务器

### 方法3: 使用百度网盘（如果提供）

使用bypy或其他工具下载
