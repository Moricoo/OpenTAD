# -*- coding: utf-8 -*-
"""
解释推理结果的含义
"""
import json

def explain_results():
    """解释推理结果"""
    
    print("=" * 70)
    print("推理结果解释")
    print("=" * 70)
    
    # 读取结果
    with open("inference_results.json", "r") as f:
        data = json.load(f)
    
    video_name = data["video"]
    results = data["results"]
    
    print("\n视频文件: {}".format(video_name))
    print("处理窗口数: {}".format(len(results)))
    
    # 读取视频信息
    with open("inference_annotation.json", "r") as f:
        ann_data = json.load(f)
    
    video_key = list(ann_data["database"].keys())[0]
    video_info = ann_data["database"][video_key]
    duration = video_info["duration"]
    total_frames = video_info["frame"]
    fps = total_frames / duration
    
    minutes = int(duration // 60)
    seconds = int(duration % 60)
    print("视频时长: {:.2f} 秒 ({}分{}秒)".format(duration, minutes, seconds))
    print("总帧数: {}".format(total_frames))
    print("帧率: {:.2f} FPS".format(fps))
    
    print("\n" + "=" * 70)
    print("结果格式说明")
    print("=" * 70)
    print("""
当前保存的是模型的原始输出，包含：

1. **模型输出格式**:
   - ActionFormer 模型输出的是预测的动作片段提议（proposals）
   - 每个窗口的输出包含多个候选动作片段
   - 每个片段包含：起始位置、结束位置、置信度分数、类别标签

2. **当前结果的含义**:
   - results[0]: 第一个滑动窗口的预测结果
   - results[1]: 第二个滑动窗口的预测结果
   - 每个结果是一个列表，包含模型的原始预测张量

3. **需要后处理**:
   原始输出需要经过以下步骤才能得到最终结果：
   - NMS (Non-Maximum Suppression): 去除重复检测
   - 置信度过滤: 过滤低置信度的检测
   - 时间转换: 将帧索引转换为秒数
   - 类别映射: 将类别索引转换为类别名称

4. **THUMOS-14 数据集类别**:
   模型可以检测 20 种动作类别，包括：
   - BaseballPitch, BasketballDunk, Billiards, CleanAndJerk
   - CliffDiving, CricketBowling, CricketShot, Diving
   - FrisbeeCatch, GolfSwing, HammerThrow, HighJump
   - JavelinThrow, LongJump, PoleVault, Shotput
   - SoccerPenalty, TennisSwing, ThrowDiscus, VolleyballSpiking
    """)
    
    # 分析第一个结果
    if results:
        result = results[0]
        print("\n" + "=" * 70)
        print("第一个窗口的原始输出分析")
        print("=" * 70)
        
        if isinstance(result, list):
            print("输出包含 {} 个部分".format(len(result)))
            for i, part in enumerate(result[:5]):  # 只显示前5个
                if isinstance(part, list):
                    print("  部分 {}: 列表，长度 {}".format(i+1, len(part)))
                    if len(part) > 0 and isinstance(part[0], list):
                        preview = part[0][:3]
                        print("    第一个元素: {}... (共 {} 个值)".format(preview, len(part[0])))
                else:
                    print("  部分 {}: {}".format(i+1, type(part)))
    
    print("\n" + "=" * 70)
    print("建议")
    print("=" * 70)
    print("""
要得到可读的动作检测结果，需要：

1. 使用 OpenTAD 框架的 post_processing 功能
2. 或者使用 tools/test.py 脚本进行完整推理（包含后处理）
3. 或者手动解析原始输出并进行 NMS 和格式转换

当前结果文件保存的是原始模型输出，主要用于：
- 调试模型性能
- 分析模型预测分布
- 进行自定义后处理
    """)

if __name__ == "__main__":
    explain_results()

