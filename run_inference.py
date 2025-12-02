# -*- coding: utf-8 -*-
import os
import sys
sys.path.insert(0, "/root/OpenTAD")

import torch
import json
from mmengine.config import Config
from opentad.models import build_detector
from opentad.datasets import build_dataset, build_dataloader

def main():
    config_path = "/root/OpenTAD/model_package_thumos_adapter_baseline/config/e2e_thumos_videomae_s_768x1_160_adapter.py"
    checkpoint_path = "/root/OpenTAD/model_package_thumos_adapter_baseline/checkpoint/latest.pth"
    video_path = "/root/OpenTAD/03-西班牙逛街日常-2-28.mp4"
    output_path = "/root/OpenTAD/inference_results.json"
    
    print("=" * 60)
    print("时序动作检测推理")
    print("=" * 60)
    print(f"配置文件: {config_path}")
    print(f"模型权重: {checkpoint_path}")
    print(f"输入视频: {video_path}")
    print(f"输出文件: {output_path}")
    print("=" * 60)
    
    # 检查文件
    if not os.path.exists(video_path):
        print(f"错误: 视频文件不存在: {video_path}")
        return
    
    if not os.path.exists(checkpoint_path):
        print(f"错误: 模型文件不存在: {checkpoint_path}")
        return
    
    # 加载配置
    print("\n[1/4] 加载配置文件...")
    cfg = Config.fromfile(config_path)
    
    # 构建模型
    print("[2/4] 构建模型...")
    model = build_detector(cfg.model)
    
    # 加载权重
    print("[3/4] 加载模型权重...")
    checkpoint = torch.load(checkpoint_path, map_location="cuda:0")
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        print("  从 state_dict 加载权重")
    else:
        model.load_state_dict(checkpoint, strict=False)
        print("  直接加载权重")
    
    model.eval()
    model.cuda()
    print("  模型已加载到 GPU")
    
    # 创建数据集和数据加载器
    print("[4/4] 准备数据和开始推理...")
    print("  注意: 由于这是单个视频推理，需要创建临时数据集配置")
    print("  建议使用完整的测试脚本来处理视频")
    
    print("\n" + "=" * 60)
    print("模型加载成功！")
    print("=" * 60)
    print("\n由于 OpenTAD 框架需要特定的数据集格式，")
    print("建议使用以下方式之一进行推理：")
    print("1. 使用 tools/test.py 脚本（需要准备数据集格式）")
    print("2. 使用模型包中的 inference_example.py（需要完善实现）")
    print("3. 直接调用模型的 forward 方法进行推理")
    print("=" * 60)

if __name__ == "__main__":
    main()
