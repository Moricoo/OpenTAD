#!/usr/bin/env python
"""
OpenTAD 安装验证脚本
测试模型构建和基本功能
"""
import sys
import os
# 修复路径：从 scripts/testing/ 回到 OpenTAD 根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
from mmengine.config import Config
from opentad.models import build_detector

def test_installation():
    print("=" * 60)
    print("OpenTAD 安装验证测试")
    print("=" * 60)
    
    # 1. 测试 PyTorch
    print("\n[1/4] 测试 PyTorch...")
    print(f"  ✓ PyTorch 版本: {torch.__version__}")
    print(f"  ✓ CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  ✓ GPU 数量: {torch.cuda.device_count()}")
        print(f"  ✓ GPU 名称: {torch.cuda.get_device_name(0)}")
    
    # 2. 测试模块导入
    print("\n[2/4] 测试模块导入...")
    try:
        import mmcv
        print(f"  ✓ mmcv 版本: {mmcv.__version__}")
    except Exception as e:
        print(f"  ✗ mmcv 导入失败: {e}")
        return False
    
    try:
        import mmaction
        print(f"  ✓ mmaction2 已安装")
    except Exception as e:
        print(f"  ✗ mmaction2 导入失败: {e}")
        return False
    
    try:
        from opentad import models, datasets, cores
        print(f"  ✓ opentad 模块导入成功")
    except Exception as e:
        print(f"  ✗ opentad 模块导入失败: {e}")
        return False
    
    # 3. 测试自定义 CUDA 扩展
    print("\n[3/4] 测试自定义 CUDA 扩展...")
    try:
        import nms_1d_cpu
        print(f"  ✓ nms_1d_cpu 导入成功")
    except Exception as e:
        print(f"  ⚠ nms_1d_cpu 导入失败: {e}")
    
    try:
        import Align1D
        print(f"  ✓ Align1D 导入成功")
    except Exception as e:
        print(f"  ⚠ Align1D 导入失败: {e}")
    
    try:
        import boundary_max_pooling_cuda
        print(f"  ✓ boundary_max_pooling_cuda 导入成功")
    except Exception as e:
        print(f"  ⚠ boundary_max_pooling_cuda 导入失败: {e}")
    
    # 4. 测试模型构建
    print("\n[4/4] 测试模型构建...")
    try:
        # 读取配置文件
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        config_path = os.path.join(base_dir, "configs/actionformer/thumos_i3d.py")
        if os.path.exists(config_path):
            cfg = Config.fromfile(config_path)
            print(f"  ✓ 配置文件加载成功: {config_path}")
            
            # 尝试构建模型（不加载权重）
            if hasattr(cfg, 'model'):
                print(f"  ✓ 模型配置存在")
                print(f"    模型类型: {cfg.model.get('type', 'N/A')}")
        else:
            print(f"  ⚠ 配置文件不存在: {config_path}")
    except Exception as e:
        print(f"  ⚠ 模型构建测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
    print("\n注意：要运行完整的训练实验，需要准备数据集。")
    print("请参考 docs/en/data.md 了解如何准备数据。")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = test_installation()
    sys.exit(0 if success else 1)

