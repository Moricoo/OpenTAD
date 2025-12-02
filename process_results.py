# -*- coding: utf-8 -*-
"""
处理推理结果，将原始输出转换为可读的动作检测结果
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
from opentad.models.utils.post_processing import batched_nms, convert_to_seconds

def process_inference_results():
    """处理推理结果，进行后处理"""
    
    # 加载配置
    config_path = "model_package_thumos_adapter_baseline/config/e2e_thumos_videomae_s_768x1_160_adapter.py"
    checkpoint_path = "model_package_thumos_adapter_baseline/checkpoint/latest.pth"
    
    print("=" * 70)
    print("处理推理结果 - 后处理和格式化")
    print("=" * 70)
    
    # 加载配置
    print("\n[1/4] 加载配置...")
    cfg = Config.fromfile(config_path)
    
    # 构建模型（仅用于后处理）
    print("[2/4] 构建模型...")
    model = build_detector(cfg.model)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.eval()
    print("  ✅ 模型加载完成")
    
    # 读取类别映射
    print("[3/4] 读取类别映射...")
    with open("inference_category_idx.txt", "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f.readlines()]
    print(f"  ✅ 共 {len(class_names)} 个类别")
    
    # 读取原始结果
    print("[4/4] 处理推理结果...")
    with open("inference_results.json", "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    video_name = raw_data["video"]
    raw_results = raw_data["results"]
    
    print(f"\n处理视频: {video_name}")
    print(f"原始结果数量: {len(raw_results)}")
    
    # 读取 annotation 获取视频元信息
    with open("inference_annotation.json", "r", encoding="utf-8") as f:
        ann_data = json.load(f)
    
    video_key = list(ann_data["database"].keys())[0]
    video_info = ann_data["database"][video_key]
    fps = video_info["frame"] / video_info["duration"]
    
    # 处理每个窗口的结果
    all_detections = []
    
    for window_idx, window_result in enumerate(raw_results):
        print(f"\n处理窗口 {window_idx + 1}/{len(raw_results)}...")
        
        # window_result 应该是模型的原始输出
        # 根据 ActionFormer 的输出格式，应该是 [segments, scores, labels] 或类似格式
        # 这里需要根据实际输出格式调整
        
        if isinstance(window_result, list) and len(window_result) > 0:
            # 尝试解析结果
            # 通常 ActionFormer 输出是预测的 segments 和 scores
            print(f"  结果类型: {type(window_result)}")
            print(f"  结果长度: {len(window_result)}")
            
            # 如果结果包含多个列表，可能是 [segments, scores, labels]
            if isinstance(window_result[0], list):
                if len(window_result) >= 2:
                    # 假设是 segments 和 scores
                    segments = torch.tensor(window_result[0]) if isinstance(window_result[0][0], (int, float)) else None
                    scores = torch.tensor(window_result[1]) if len(window_result) > 1 and isinstance(window_result[1][0], (int, float)) else None
                    
                    if segments is not None and scores is not None:
                        print(f"  检测到 {len(segments)} 个候选片段")
                        
                        # 应用 NMS
                        if len(segments.shape) == 2 and segments.shape[1] == 2:
                            # segments 格式: [N, 2] (start, end)
                            labels = torch.zeros(len(segments), dtype=torch.long)  # 临时标签
                            
                            # NMS 后处理
                            nms_config = cfg.post_processing.nms
                            segments_nms, scores_nms, labels_nms = batched_nms(
                                segments.unsqueeze(0),
                                scores.unsqueeze(0),
                                labels.unsqueeze(0),
                                **nms_config
                            )
                            
                            segments_nms = segments_nms[0]
                            scores_nms = scores_nms[0]
                            labels_nms = labels_nms[0]
                            
                            # 转换为秒数
                            meta = {
                                "video_name": video_key,
                                "fps": fps,
                                "duration": video_info["duration"],
                                "frame": video_info["frame"]
                            }
                            
                            segments_seconds = convert_to_seconds(segments_nms, meta)
                            
                            # 格式化结果
                            for seg, score, label_idx in zip(segments_seconds, scores_nms, labels_nms):
                                label_name = class_names[label_idx.item()] if label_idx.item() < len(class_names) else f"Class_{label_idx.item()}"
                                all_detections.append({
                                    "window": window_idx + 1,
                                    "segment": [round(seg[0].item(), 2), round(seg[1].item(), 2)],
                                    "label": label_name,
                                    "score": round(score.item(), 4),
                                    "start_time": f"{int(seg[0].item())//60:02d}:{int(seg[0].item())%60:02d}",
                                    "end_time": f"{int(seg[1].item())//60:02d}:{int(seg[1].item())%60:02d}",
                                })
    
    # 保存处理后的结果
    output_data = {
        "video": video_name,
        "duration": video_info["duration"],
        "fps": fps,
        "total_frames": video_info["frame"],
        "detections": all_detections,
        "summary": {
            "total_detections": len(all_detections),
            "windows_processed": len(raw_results)
        }
    }
    
    output_path = "inference_results_processed.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 70)
    print("✅ 结果处理完成！")
    print(f"📊 检测到 {len(all_detections)} 个动作片段")
    print(f"📁 处理后的结果已保存到: {output_path}")
    print("=" * 70)
    
    # 打印前几个检测结果
    if all_detections:
        print("\n前 10 个检测结果:")
        print("-" * 70)
        for i, det in enumerate(all_detections[:10]):
            print(f"{i+1}. {det['label']} | "
                  f"时间: {det['start_time']} - {det['end_time']} "
                  f"({det['segment'][0]:.1f}s - {det['segment'][1]:.1f}s) | "
                  f"置信度: {det['score']:.4f}")
    else:
        print("\n⚠️  未检测到任何动作片段")
        print("这可能是因为:")
        print("  1. 模型输出格式需要进一步解析")
        print("  2. 需要调整置信度阈值")
        print("  3. 原始结果需要不同的后处理方式")

if __name__ == "__main__":
    process_inference_results()

