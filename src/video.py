import os
import json
import cv2
from datetime import timedelta

def process_video(video_path, json_path, output_dir):
    """处理单个视频-字幕对（修正版）"""
    # 读取字幕文件
    with open(json_path, 'r', encoding='utf-8') as f:
        subs = json.load(f)  # 直接加载列表，无需访问'body'键
    
    # 初始化视频捕获
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 创建输出目录
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_subdir = os.path.join(output_dir, base_name)
    os.makedirs(output_subdir, exist_ok=True)
    
    # 预构建时间轴映射表（提升查询效率）
    timeline = []
    for sub in subs:
        start = sub["from"]
        end = sub["to"]
        text = sub["content"]
        timeline.append( (start, end, text) )
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        current_time = frame_count / fps
        caption = "无字幕"
        
        # 线性搜索当前字幕（优化后可改用二分查找）
        for start, end, text in timeline:
            if start <= current_time <= end:
                caption = text.replace(' ', '_').replace('/', '-')[:50]
                break
        
        # 生成文件名并保存
        timestamp = str(timedelta(seconds=current_time))[:-7].replace(':', '-')
        filename = f"{timestamp}_{caption}.jpg"
        cv2.imwrite(os.path.join(output_subdir, filename), frame)
        
        frame_count += 1
    
    cap.release()

if __name__ == "__main__":
    VIDEO_DIR = "data"    # 视频文件夹路径
    OUTPUT_DIR = "output"   # 输出目录
    
    # 遍历处理所有视频文件
    for filename in os.listdir(VIDEO_DIR):
        if filename.endswith(".flv"):
            base_name = os.path.splitext(filename)[0]
            video_path = os.path.join(VIDEO_DIR, filename)
            json_path = os.path.join(VIDEO_DIR, f"{base_name}.json")
            output_subdir = os.path.join(OUTPUT_DIR, base_name)  # 新增输出目录路径检查
            
            # 新增条件：当输出目录已存在时跳过处理
            if os.path.exists(json_path) and not os.path.exists(output_subdir):
                print(f"正在处理：{filename}")
                process_video(video_path, json_path, OUTPUT_DIR)
            elif os.path.exists(output_subdir):
                print(f"跳过已处理视频：{filename}")
            else:
                print(f"未找到对应的字幕文件：{filename}")