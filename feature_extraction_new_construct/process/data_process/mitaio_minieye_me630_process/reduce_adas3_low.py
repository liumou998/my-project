import re
import os
from datetime import datetime

def synchronize_frames(adas1_file, adas2_file, adas3_file, output_prefix):
    # Get the directory of the input files
    output_dir = os.path.dirname(adas1_file)
    
    class DataGroup:
        def __init__(self):
            self.timestamp = None
            self.frameid = None
            self.lines = []

    def parse_file(filename):
        groups = []
        current_group = None
        
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                timestamp_match = re.search(r'\[(.*?)\]', line)
                frame_match = re.search(r'frameid (\w+)', line)
                
                if timestamp_match and frame_match and 'prot_type' in line:
                    if current_group:
                        groups.append(current_group)
                    current_group = DataGroup()
                    current_group.timestamp = timestamp_match.group(1)
                    current_group.frameid = frame_match.group(1)
                    current_group.lines = [line]
                elif current_group:
                    current_group.lines.append(line)
                
            if current_group:
                groups.append(current_group)
        return groups

    adas1_groups = parse_file(adas1_file)
    adas2_groups = parse_file(adas2_file)
    adas3_groups = parse_file(adas3_file)

    # 获取 ADAS3 的帧数，用于 ADAS1 和 ADAS2 的对齐
    target_frame_count = len(adas3_groups)
    adas3_timestamps = [group.timestamp for group in adas3_groups]

    def find_closest_frame(adas_groups, target_timestamp):
        closest_frame = None
        closest_time_diff = float('inf')
        
        for adas_group in adas_groups:
            time1 = datetime.strptime(adas_group.timestamp, '%Y-%m-%d %H:%M:%S.%f')
            time2 = datetime.strptime(target_timestamp, '%Y-%m-%d %H:%M:%S.%f')
            
            time_diff = abs((time1 - time2).total_seconds())
            if time_diff < closest_time_diff:
                closest_time_diff = time_diff
                closest_frame = adas_group
        
        return closest_frame

    selected_adas1_frames = []
    selected_adas2_frames = []
    
    # 为每个 ADAS3 帧找到对应的 ADAS1 和 ADAS2 帧
    for adas3_group in adas3_groups:
        closest_adas1_frame = find_closest_frame(adas1_groups, adas3_group.timestamp)
        closest_adas2_frame = find_closest_frame(adas2_groups, adas3_group.timestamp)

        if closest_adas1_frame and closest_adas1_frame not in selected_adas1_frames:
            selected_adas1_frames.append(closest_adas1_frame)
        if closest_adas2_frame and closest_adas2_frame not in selected_adas2_frames:
            selected_adas2_frames.append(closest_adas2_frame)

    # 如果匹配到的帧数不够，则补充剩余的帧
    if len(selected_adas1_frames) < target_frame_count:
        remaining_adas1_frames = [frame for frame in adas1_groups if frame not in selected_adas1_frames]
        selected_adas1_frames.extend(remaining_adas1_frames[:target_frame_count - len(selected_adas1_frames)])
        
    if len(selected_adas2_frames) < target_frame_count:
        remaining_adas2_frames = [frame for frame in adas2_groups if frame not in selected_adas2_frames]
        selected_adas2_frames.extend(remaining_adas2_frames[:target_frame_count - len(selected_adas2_frames)])

    # 确保不超过 ADAS3 的帧数
    selected_adas1_frames = selected_adas1_frames[:target_frame_count]
    selected_adas2_frames = selected_adas2_frames[:target_frame_count]

    def write_output(groups, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            for group in groups:
                f.write(''.join(group.lines))
                f.write('\n')

    # Save output files in the same directory as input files
    write_output(selected_adas1_frames, os.path.join(output_dir, f"{output_prefix}1_sync.txt"))
    write_output(selected_adas2_frames, os.path.join(output_dir, f"{output_prefix}2_sync.txt"))

    return len(selected_adas1_frames), len(selected_adas2_frames), target_frame_count

# Example usage
adas1_file = "/home/linux/ShenGang/feature_extraction_new_construct/process/data_process/dataprocess/ADAS1.txt"
adas2_file = "/home/linux/ShenGang/feature_extraction_new_construct/process/data_process/dataprocess/ADAS2.txt"
adas3_file = "/home/linux/ShenGang/feature_extraction_new_construct/process/data_process/dataprocess/ADAS3.txt"
output_prefix = "/home/linux/ShenGang/feature_extraction_new_construct/process/data_process/dataprocess/ADAS"

frame_count_adas1, frame_count_adas2, frame_count_adas3 = synchronize_frames(adas1_file, adas2_file, adas3_file, output_prefix)
print(f"ADAS3 帧数: {frame_count_adas3}")
print(f"同步后 ADAS1 帧数: {frame_count_adas1}")
print(f"同步后 ADAS2 帧数: {frame_count_adas2}")