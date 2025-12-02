# -*- coding: utf-8 -*-
"""
完整的视频推理脚本 - 使用 OpenTAD 框架对单个 MP4 视频进行动作检测
"""
import os
import sys
sys.path.insert(0, "/root/OpenTAD")
os.chdir("/root/OpenTAD")

import torch
import json
import numpy as np
from mmengine.config import Config
from opentad.models import build_detector
from opentad.datasets import build_dataset, build_dataloader

def main():
    # 路径配置
    config_path = "model_package_thumos_adapter_baseline/config/e2e_thumos_videomae_s_768x1_160_adapter.py"
    checkpoint_path = "model_package_thumos_adapter_baseline/checkpoint/latest.pth"
    video_path = "03-西班牙逛街日常-2-28.mp4"
    ann_file = "inference_annotation.json"
    class_map = "inference_category_idx.txt"
    output_path = "inference_results.json"
    
    print("=" * 70)
    print("时序动作检测推理 - 单个视频")
    print("=" * 70)
    
    # 1. 加载配置
    print("\n[1/5] 加载配置文件...")
    cfg = Config.fromfile(config_path)
    
    # 修改数据集配置以使用我们的视频
    cfg.dataset.test.ann_file = ann_file
    cfg.dataset.test.class_map = class_map
    cfg.dataset.test.data_path = "."  # 当前目录
    cfg.dataset.test.subset_name = "test"
    
    print(f"  ✅ 配置加载成功")
    print(f"  📋 Annotation 文件: {ann_file}")
    print(f"  📋 类别映射文件: {class_map}")
    print(f"  📹 视频路径: {video_path}")
    
    # 2. 构建模型
    print("\n[2/5] 构建模型...")
    model = build_detector(cfg.model)
    print("  ✅ 模型构建成功")
    
    # 3. 加载权重
    print("\n[3/5] 加载模型权重...")
    checkpoint = torch.load(checkpoint_path, map_location="cuda:0")
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        print("  ✅ 从 state_dict 加载权重")
    else:
        model.load_state_dict(checkpoint, strict=False)
        print("  ✅ 直接加载权重")
    
    model.eval()
    model.cuda()
    print("  ✅ 模型已加载到 GPU")
    
    # 4. 构建数据集和数据加载器
    print("\n[4/5] 构建数据集和数据加载器...")
    dataset = build_dataset(cfg.dataset.test)
    print(f"  ✅ 数据集构建成功，共 {len(dataset)} 个样本")
    
    dataloader = build_dataloader(
        dataset, 
        batch_size=cfg.solver.test.batch_size,
        rank=0,
        world_size=1,
        num_workers=cfg.solver.test.num_workers
    )
    print(f"  ✅ 数据加载器构建成功")
    
    # 5. 执行推理
    print("\n[5/5] 开始推理...")
    all_results = []
    
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            # 将数据移到 GPU
            if isinstance(data, dict):
                for key in data:
                    if isinstance(data[key], torch.Tensor):
                        data[key] = data[key].cuda()
            
            # 推理 - 使用测试模式
            try:
                # 移除训练相关的键
                test_data = {k: v for k, v in data.items() if k not in ['gt_segments', 'gt_labels']}
                output = model.forward_test(**test_data)
                
                # 处理输出
                if isinstance(output, (list, tuple)):
                    output = output[0]
                
                # 转换为可序列化的格式
                def convert_to_serializable(obj):
                    if isinstance(obj, torch.Tensor):
                        return obj.cpu().numpy().tolist()
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, dict):
                        return {k: convert_to_serializable(v) for k, v in obj.items()}
                    elif isinstance(obj, (list, tuple)):
                        return [convert_to_serializable(item) for item in obj]
                    else:
                        return obj
                
                result = convert_to_serializable(output)
                all_results.append(result)
                
                print(f"  ✅ 批次 {batch_idx + 1}/{len(dataloader)} 推理完成")
                
            except Exception as e:
                print(f"  ⚠️  批次 {batch_idx + 1} 推理失败: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # 保存结果
    print(f"\n💾 保存结果到: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "video": video_path,
            "results": all_results,
            "num_batches": len(all_results)
        }, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 70)
    print("✅ 推理完成！")
    print(f"📊 处理了 {len(all_results)} 个批次")
    print(f"📁 结果已保存到: {output_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()

