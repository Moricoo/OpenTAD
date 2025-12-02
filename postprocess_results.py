# -*- coding: utf-8 -*-
"""
完整的后处理脚本 - 将原始推理结果转换为可读的动作检测结果
"""
import os
import sys
sys.path.insert(0, "/root/OpenTAD")
os.chdir("/root/OpenTAD")

import torch
import json
import numpy as np

# 导入 OpenTAD 的后处理工具
from opentad.models.utils.post_processing import batched_nms, convert_to_seconds

def postprocess_results():
    """后处理推理结果"""
    
    print("=" * 70)
    print("Post-processing Inference Results")
    print("=" * 70)
    
    # 1. 读取原始结果
    print("\n[1/5] Loading raw results...")
    with open("inference_results.json", "r") as f:
        raw_data = json.load(f)
    
    video_name = raw_data["video"]
    raw_results = raw_data["results"]
    print("  Video: {}".format(video_name))
    print("  Number of windows: {}".format(len(raw_results)))
    
    # 2. 读取视频元信息
    print("\n[2/5] Loading video metadata...")
    with open("inference_annotation.json", "r") as f:
        ann_data = json.load(f)
    
    video_key = list(ann_data["database"].keys())[0]
    video_info = ann_data["database"][video_key]
    duration = video_info["duration"]
    total_frames = video_info["frame"]
    fps = total_frames / duration
    
    print("  Duration: {:.2f}s ({}m {}s)".format(
        duration, int(duration//60), int(duration%60)))
    print("  Total frames: {}".format(total_frames))
    print("  FPS: {:.2f}".format(fps))
    
    # 3. 读取类别映射
    print("\n[3/5] Loading class mapping...")
    with open("inference_category_idx.txt", "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    print("  Number of classes: {}".format(len(class_names)))
    
    # 4. 加载配置获取后处理参数
    print("\n[4/5] Loading post-processing config...")
    from mmengine.config import Config
    config_path = "model_package_thumos_adapter_baseline/config/e2e_thumos_videomae_s_768x1_160_adapter.py"
    cfg = Config.fromfile(config_path)
    
    post_cfg = cfg.post_processing
    pre_nms_thresh = getattr(post_cfg, "pre_nms_thresh", 0.001)
    pre_nms_topk = getattr(post_cfg, "pre_nms_topk", 2000)
    
    print("  Pre-NMS threshold: {}".format(pre_nms_thresh))
    print("  Pre-NMS top-k: {}".format(pre_nms_topk))
    print("  NMS config: {}".format(post_cfg.nms))
    
    # 5. 处理每个窗口的结果
    print("\n[5/5] Processing results...")
    all_detections = []
    
    for window_idx, window_result in enumerate(raw_results):
        print("  Processing window {}/{}...".format(window_idx + 1, len(raw_results)))
        
        # window_result 应该是 [rpn_proposals, rpn_scores] 的列表形式
        if not isinstance(window_result, list) or len(window_result) < 2:
            print("    Warning: Unexpected result format, skipping...")
            continue
        
        # 转换为 tensor
        try:
            # rpn_proposals: [N, 2] - segments in frame indices
            proposals_list = window_result[0]
            if isinstance(proposals_list, list) and len(proposals_list) > 0:
                if isinstance(proposals_list[0], list):
                    rpn_proposals = torch.tensor(proposals_list, dtype=torch.float32)
                else:
                    print("    Warning: Unexpected proposals format")
                    continue
            else:
                continue
            
            # rpn_scores: [N, num_classes] - class scores (after sigmoid)
            scores_list = window_result[1]
            if isinstance(scores_list, list) and len(scores_list) > 0:
                if isinstance(scores_list[0], list):
                    rpn_scores = torch.tensor(scores_list, dtype=torch.float32)
                else:
                    # 可能是单类情况
                    rpn_scores = torch.tensor(scores_list, dtype=torch.float32).unsqueeze(-1)
            else:
                continue
            
            # 确保维度正确
            if len(rpn_proposals.shape) == 2 and rpn_proposals.shape[1] == 2:
                # 添加 batch 维度: [1, N, 2]
                rpn_proposals = rpn_proposals.unsqueeze(0)
            else:
                continue
            
            if len(rpn_scores.shape) == 2:
                # 添加 batch 维度: [1, N, num_classes]
                rpn_scores = rpn_scores.unsqueeze(0)
            elif len(rpn_scores.shape) == 1:
                # 单类情况: [1, N, 1]
                rpn_scores = rpn_scores.unsqueeze(0).unsqueeze(-1)
            else:
                continue
            
            # 后处理
            num_classes = rpn_scores.shape[-1]
            segments = rpn_proposals[0].detach().cpu()  # [N, 2]
            scores = rpn_scores[0].detach().cpu()  # [N, num_classes]
            
            if num_classes == 1:
                # 单类情况
                scores_flat = scores.squeeze(-1)
                labels = torch.zeros(scores_flat.shape[0], dtype=torch.long)
            else:
                # 多类情况
                pred_prob = scores.flatten()  # [N*num_classes]
                
                # 1. 过滤低置信度
                keep_idxs1 = pred_prob > pre_nms_thresh
                pred_prob = pred_prob[keep_idxs1]
                topk_idxs = keep_idxs1.nonzero(as_tuple=True)[0]
                
                if len(topk_idxs) == 0:
                    print("    No detections above threshold")
                    continue
                
                # 2. Top-k 选择
                num_topk = min(pre_nms_topk, topk_idxs.size(0))
                pred_prob, idxs = pred_prob.sort(descending=True)
                pred_prob = pred_prob[:num_topk].clone()
                topk_idxs = topk_idxs[idxs[:num_topk]].clone()
                
                # 3. 提取提议和类别
                pt_idxs = torch.div(topk_idxs, num_classes, rounding_mode="floor")
                cls_idxs = torch.fmod(topk_idxs, num_classes)
                
                segments = segments[pt_idxs]
                scores_flat = pred_prob
                labels = cls_idxs.long()
            
            # 4. NMS
            if post_cfg.nms is not None:
                segments_nms, scores_nms, labels_nms = batched_nms(
                    segments.unsqueeze(0),
                    scores_flat.unsqueeze(0),
                    labels.unsqueeze(0),
                    **post_cfg.nms
                )
                segments = segments_nms[0]
                scores_flat = scores_nms[0]
                labels = labels_nms[0]
            
            # 5. 转换为秒数
            # 从配置中获取参数
            snippet_stride = cfg.dataset.test.get("feature_stride", 4)
            sample_stride = cfg.dataset.test.get("sample_stride", 1)
            actual_stride = snippet_stride * sample_stride
            
            # 计算窗口起始帧（对于滑动窗口，需要知道每个窗口的起始位置）
            # 由于我们不知道确切的窗口位置，使用 0 作为默认值
            # 实际应用中，应该从数据集的 meta 信息中获取
            window_start_frame = 0  # 第一个窗口从第0帧开始
            offset_frames = 0  # 通常为0，除非有特殊偏移
            
            meta = {
                "video_name": video_key,
                "fps": fps,
                "duration": duration,
                "frame": total_frames,
                "snippet_stride": actual_stride,
                "feature_stride": snippet_stride,
                "sample_stride": sample_stride,
                "offset_frames": offset_frames,
                "window_start_frame": window_start_frame
            }
            
            segments_seconds = convert_to_seconds(segments, meta)
            
            # 确保维度正确 - segments_seconds 应该是 [N, 2]
            if segments_seconds.dim() == 1:
                if segments_seconds.shape[0] == 2:
                    # 单个片段，reshape 为 [1, 2]
                    segments_seconds = segments_seconds.unsqueeze(0)
                    if scores_flat.dim() == 0:
                        scores_flat = scores_flat.unsqueeze(0)
                    if labels.dim() == 0:
                        labels = labels.unsqueeze(0)
                else:
                    print("    Unexpected segments shape: {}".format(segments_seconds.shape))
                    continue
            
            if segments_seconds.shape[0] == 0:
                print("    No segments after conversion")
                continue
            
            print("    After processing - Segments: {}, Scores: {}, Labels: {}".format(
                segments_seconds.shape, scores_flat.shape, labels.shape))
            
            # 6. 格式化结果
            num_segments = segments_seconds.shape[0]
            for i in range(num_segments):
                seg = segments_seconds[i]
                
                # 安全地提取起始和结束时间
                try:
                    if isinstance(seg, torch.Tensor):
                        if seg.dim() == 0 or seg.numel() < 2:
                            print("      Skipping segment {}: invalid shape".format(i))
                            continue
                        start_sec = float(seg[0].item())
                        end_sec = float(seg[1].item())
                    else:
                        if len(seg) < 2:
                            print("      Skipping segment {}: insufficient length".format(i))
                            continue
                        start_sec = float(seg[0])
                        end_sec = float(seg[1])
                except Exception as e:
                    print("      Error extracting segment {}: {}".format(i, e))
                    continue
                
                # 安全地提取分数和标签
                if isinstance(scores_flat, torch.Tensor):
                    score_val = float(scores_flat[i].item() if scores_flat.dim() > 0 else scores_flat.item())
                else:
                    score_val = float(scores_flat[i] if isinstance(scores_flat, (list, tuple)) else scores_flat)
                
                if isinstance(labels, torch.Tensor):
                    label_val = int(labels[i].item() if labels.dim() > 0 else labels.item())
                else:
                    label_val = int(labels[i] if isinstance(labels, (list, tuple)) else labels)
                
                label_name = class_names[label_val] if label_val < len(class_names) else "Unknown"
                start_time = "{}:{:02d}".format(int(start_sec // 60), int(start_sec % 60))
                end_time = "{}:{:02d}".format(int(end_sec // 60), int(end_sec % 60))
                
                # 计算帧索引
                start_frame = int(start_sec * fps)
                end_frame = int(end_sec * fps)
                
                all_detections.append({
                    "window": window_idx + 1,
                    "segment_seconds": [round(start_sec, 2), round(end_sec, 2)],
                    "segment_frames": [start_frame, end_frame],
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": round(end_sec - start_sec, 2),
                    "label": label_name,
                    "class_id": label_val,
                    "confidence": round(score_val, 4)
                })
            
            print("    Found {} detections after post-processing".format(len(segments)))
            
        except Exception as e:
            print("    Error processing window: {}".format(e))
            import traceback
            traceback.print_exc()
            continue
    
    # 6. 保存结果
    output_data = {
        "video": video_name,
        "video_info": {
            "duration": duration,
            "total_frames": total_frames,
            "fps": fps
        },
        "detections": all_detections,
        "summary": {
            "total_detections": len(all_detections),
            "windows_processed": len(raw_results),
            "classes_detected": len(set([d["label"] for d in all_detections]))
        }
    }
    
    output_path = "inference_results_final.json"
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 70)
    print("Post-processing Complete!")
    print("=" * 70)
    print("Total detections: {}".format(len(all_detections)))
    print("Output saved to: {}".format(output_path))
    
    # 打印前10个检测结果
    if all_detections:
        print("\nTop 10 Detections:")
        print("-" * 70)
        for i, det in enumerate(sorted(all_detections, key=lambda x: x["confidence"], reverse=True)[:10]):
            print("{}. {} | Time: {} - {} ({:.1f}s - {:.1f}s) | "
                  "Confidence: {:.4f} | Duration: {:.1f}s".format(
                i+1,
                det["label"],
                det["start_time"],
                det["end_time"],
                det["segment_seconds"][0],
                det["segment_seconds"][1],
                det["confidence"],
                det["duration"]
            ))
    else:
        print("\nNo detections found. This could mean:")
        print("  1. The video doesn't contain actions from THUMOS-14 dataset")
        print("  2. Confidence threshold is too high")
        print("  3. Need to adjust post-processing parameters")
    
    print("=" * 70)

if __name__ == "__main__":
    postprocess_results()

