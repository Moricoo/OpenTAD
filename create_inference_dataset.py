# -*- coding: utf-8 -*-
"""
为单个视频创建推理所需的数据集格式
"""
import json
import cv2
import os

def get_video_info(video_path):
    """获取视频信息"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    return fps, frame_count, duration

def create_annotation_file(video_path, output_path):
    """创建 annotation 文件"""
    video_name = os.path.basename(video_path)
    fps, frame_count, duration = get_video_info(video_path)
    
    annotation = {
        "database": {
            video_name: {
                "duration": duration,
                "frame": frame_count,
                "subset": "test",
                "annotations": []  # 推理时不需要真实标注
            }
        }
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(annotation, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 创建 annotation 文件: {output_path}")
    print(f"   视频: {video_name}")
    print(f"   时长: {duration:.2f} 秒")
    print(f"   帧数: {frame_count}")
    print(f"   FPS: {fps:.2f}")
    
    return annotation

def create_class_map(output_path):
    """创建类别映射文件（使用 THUMOS-14 的类别）"""
    # THUMOS-14 有 20 个动作类别
    classes = [
        "BaseballPitch", "BasketballDunk", "Billiards", "CleanAndJerk",
        "CliffDiving", "CricketBowling", "CricketShot", "Diving",
        "FrisbeeCatch", "GolfSwing", "HammerThrow", "HighJump",
        "JavelinThrow", "LongJump", "PoleVault", "Shotput",
        "SoccerPenalty", "TennisSwing", "ThrowDiscus", "VolleyballSpiking"
    ]
    
    with open(output_path, "w", encoding="utf-8") as f:
        for idx, cls in enumerate(classes):
            f.write(f"{cls}\n")
    
    print(f"✅ 创建类别映射文件: {output_path}")
    print(f"   类别数: {len(classes)}")

if __name__ == "__main__":
    video_path = "03-西班牙逛街日常-2-28.mp4"
    ann_file = "inference_annotation.json"
    class_map = "inference_category_idx.txt"
    
    create_annotation_file(video_path, ann_file)
    create_class_map(class_map)
    print("\n✅ 数据集文件创建完成！")
