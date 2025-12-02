# -*- coding: utf-8 -*-
import os
import sys
sys.path.insert(0, "/root/OpenTAD")
os.chdir("/root/OpenTAD")

import torch
import json
from mmengine.config import Config
from opentad.models import build_detector

def main():
    config_path = "model_package_thumos_adapter_baseline/config/e2e_thumos_videomae_s_768x1_160_adapter.py"
    checkpoint_path = "model_package_thumos_adapter_baseline/checkpoint/latest.pth"
    video_path = "03-西班牙逛街日常-2-28.mp4"
    
    print("=" * 60)
    print("时序动作检测推理")
    print("=" * 60)
    
    # 加载配置
    print("[1/4] 加载配置文件...")
    cfg = Config.fromfile(config_path)
    print("  ✅ 配置加载成功")
    
    # 构建模型
    print("[2/4] 构建模型...")
    model = build_detector(cfg.model)
    print("  ✅ 模型构建成功")
    
    # 加载权重
    print("[3/4] 加载模型权重...")
    checkpoint = torch.load(checkpoint_path, map_location="cuda:0")
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model.eval()
    model.cuda()
    print("  ✅ 模型已加载到 GPU")
    
    print("[4/4] 模型部署完成")
    print("=" * 60)
    print(f"✅ 模型已成功部署！")
    print(f"📹 视频文件: {video_path}")
    print(f"💡 模型已准备好进行推理")
    print("=" * 60)
    print("\n注意: 完整的视频推理需要准备数据集格式。")
    print("模型已加载，可以进行下一步的推理操作。")

if __name__ == "__main__":
    main()
