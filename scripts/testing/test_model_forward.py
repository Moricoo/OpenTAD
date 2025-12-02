#!/usr/bin/env python
"""
OpenTAD 模型前向传播测试
使用随机数据测试模型的前向传播功能
"""
import sys
import os
# 修复路径：从 scripts/testing/ 回到 OpenTAD 根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
from mmengine.config import Config
from opentad.models import build_detector

def test_model_forward():
    print("=" * 60)
    print("OpenTAD 模型前向传播测试")
    print("=" * 60)
    
    # 加载配置
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    config_path = os.path.join(base_dir, "configs/actionformer/thumos_i3d.py")
    print(f"\n[1/3] 加载配置文件: {config_path}")
    cfg = Config.fromfile(config_path)
    
    # 修改配置以使用更小的批次进行测试
    cfg.solver.train.batch_size = 1
    cfg.solver.train.num_workers = 0
    
    print(f"  ✓ 配置文件加载成功")
    print(f"  ✓ 模型类型: {cfg.model.type}")
    
    # 构建模型
    print(f"\n[2/3] 构建模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_detector(cfg.model)
    model = model.to(device)
    model.eval()
    print(f"  ✓ 模型构建成功")
    print(f"  ✓ 设备: {device}")
    
    # 创建随机输入数据
    print(f"\n[3/3] 测试前向传播...")
    batch_size = 1
    num_classes = 20  # THUMOS-14 有 20 个类别
    seq_len = 100  # 序列长度
    
    # 创建随机特征 (C, T) 格式
    feats = torch.randn(2048, seq_len).unsqueeze(0).to(device)  # (1, 2048, T)
    
    # 创建随机 ground truth segments 和 labels
    num_gt = 3
    gt_segments = torch.tensor([
        [10, 30],
        [50, 70],
        [80, 95]
    ]).float().unsqueeze(0).to(device)  # (1, num_gt, 2)
    
    gt_labels = torch.randint(0, num_classes, (1, num_gt)).to(device)  # (1, num_gt)
    
    # 创建 masks
    masks = torch.ones(1, seq_len).bool().to(device)  # (1, T)
    
    # 准备 metas（元数据）
    metas = [{'video_name': 'test_video_001', 'duration': seq_len * 4.0}]  # feature_stride=4
    
    print(f"  输入形状:")
    print(f"    inputs (feats): {feats.shape}")
    print(f"    masks: {masks.shape}")
    print(f"    gt_segments: {gt_segments.shape}")
    print(f"    gt_labels: {gt_labels.shape}")
    
    # 前向传播（训练模式）
    try:
        with torch.no_grad():
            outputs = model(
                inputs=feats,
                masks=masks,
                metas=metas,
                gt_segments=gt_segments,
                gt_labels=gt_labels,
                return_loss=True
            )
        
        print(f"\n  ✓ 前向传播（训练模式）成功！")
        print(f"  输出类型: {type(outputs)}")
        
        if isinstance(outputs, dict):
            print(f"  输出键: {list(outputs.keys())}")
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor):
                    print(f"    {key}: {value.shape} (值范围: [{value.min().item():.4f}, {value.max().item():.4f}])")
                else:
                    print(f"    {key}: {value}")
        
        # 测试推理模式（仅前向传播，不进行后处理）
        print(f"\n  测试推理模式（forward_test）...")
        with torch.no_grad():
            inference_outputs = model.forward_test(
                inputs=feats,
                masks=masks,
                metas=metas
            )
            print(f"  ✓ 推理模式（forward_test）成功！")
            if isinstance(inference_outputs, tuple):
                proposals, scores = inference_outputs
                print(f"  检测框数量: {len(proposals)} 个视频")
                if len(proposals) > 0:
                    print(f"  第一个视频的检测框形状: {proposals[0].shape if hasattr(proposals[0], 'shape') else type(proposals[0])}")
                    print(f"  第一个视频的分数形状: {scores[0].shape if hasattr(scores[0], 'shape') else type(scores[0])}")
        
    except Exception as e:
        print(f"\n  ✗ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("模型前向传播测试完成！")
    print("=" * 60)
    print("\n模型可以正常进行前向传播和推理。")
    print("要运行完整的训练，需要准备真实的数据集。")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = test_model_forward()
    sys.exit(0 if success else 1)

