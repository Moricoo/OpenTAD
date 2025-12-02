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
    
    print("Loading config...")
    cfg = Config.fromfile(config_path)
    
    # 修改数据集配置以使用单个视频
    # 创建一个临时的ann_file，包含视频信息
    import tempfile
    temp_ann = tempfile.NamedTemporaryFile(mode=w, suffix=.json, delete=False)
    ann_data = {
        "database": {
            "03-西班牙逛街日常-2-28.mp4": {
                "annotations": [],
                "duration": 0,  # 将从视频中获取
                "subset": "test"
            }
        }
    }
    json.dump(ann_data, temp_ann)
    temp_ann.close()
    
    # 更新配置
    cfg.dataset.test.ann_file = temp_ann.name
    cfg.dataset.test.video_prefix = os.path.dirname(video_path)
    
    print("Building model...")
    model = build_detector(cfg.model)
    
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location="cuda:0")
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model.eval()
    model.cuda()
    print("Model loaded successfully!")
    
    print("Building dataset...")
    dataset = build_dataset(cfg.dataset.test)
    dataloader = build_dataloader(dataset, cfg.solver.test, seed=42, dist=False)
    
    print("Starting inference...")
    results = []
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            # 将数据移到GPU
            if isinstance(data, dict):
                for key in data:
                    if isinstance(data[key], torch.Tensor):
                        data[key] = data[key].cuda()
            
            # 推理
            output = model(**data)
            
            # 处理输出
            if isinstance(output, (list, tuple)):
                output = output[0]
            
            results.append(output)
            print(f"Processed batch {batch_idx + 1}")
    
    # 保存结果
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Inference completed! Results saved to {output_path}")
    os.unlink(temp_ann.name)

if __name__ == "__main__":
    main()
