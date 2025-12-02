"""
时序动作检测模型推理示例
使用方法：
    python inference_example.py --video <video_path> --checkpoint checkpoint/latest.pth --config config/e2e_thumos_videomae_s_768x1_160_adapter.py
"""

import argparse
import torch
from mmengine import Config
from opentad.models import build_model
from opentad.datasets import build_dataset, build_dataloader

def main():
    parser = argparse.ArgumentParser(description='时序动作检测推理')
    parser.add_argument('--video', type=str, required=True, help='输入视频路径')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型checkpoint路径')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--device', type=str, default='cuda:0', help='推理设备')
    args = parser.parse_args()
    
    # 加载配置
    cfg = Config.fromfile(args.config)
    
    # 构建模型
    model = build_model(cfg.model)
    
    # 加载checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model.eval()
    model.to(args.device)
    
    print(f"✅ 模型已加载到 {args.device}")
    print(f"📹 开始处理视频: {args.video}")
    
    # TODO: 实现视频推理逻辑
    # 这里需要根据OpenTAD的实际推理接口来实现
    
    print("推理完成！")

if __name__ == '__main__':
    main()
