


import re
from datetime import datetime

def synchronize_frames(adas1_file, adas2_file, adas3_file, output_prefix):
    # 用于存储每一组数据的类，包含时间戳、帧ID和行数据
    class DataGroup:
        def __init__(self):
            self.timestamp = None  # 时间戳
            self.frameid = None    # 帧ID
            self.lines = []        # 行数据（文件中的内容）

    # 解析文件的函数，根据时间戳和帧ID来分组数据
    def parse_file(filename):
        groups = []  # 存储所有的数据组
        current_group = None  # 当前正在处理的数据组
        
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                # 查找时间戳和帧ID
                timestamp_match = re.search(r'\[(.*?)\]', line)
                frame_match = re.search(r'frameid (\w+)', line)
                
                # 如果找到了时间戳和帧ID，并且这一行包含'prot_type'（标识有效数据行）
                if timestamp_match and frame_match and 'prot_type' in line:
                    if current_group:
                        groups.append(current_group)  # 将当前组添加到列表中
                    current_group = DataGroup()  # 创建一个新的数据组
                    current_group.timestamp = timestamp_match.group(1)  # 设置时间戳
                    current_group.frameid = frame_match.group(1)  # 设置帧ID
                    current_group.lines = [line]  # 将当前行加入数据组
                elif current_group:
                    current_group.lines.append(line)  # 如果当前组存在，则继续添加行数据
                
            # 将最后一组数据添加到列表中
            if current_group:
                groups.append(current_group)
        return groups

    # 解析三个ADAS文件的数据
    adas1_groups = parse_file(adas1_file)
    adas2_groups = parse_file(adas2_file)
    adas3_groups = parse_file(adas3_file)

    # 找到与目标时间戳最接近的帧
    def find_closest_frame(adas_groups, target_timestamp):
        closest_frame = None
        closest_time_diff = float('inf')  # 初始化时间差为无限大
        
        # 遍历所有ADAS帧，计算时间差并找到最接近的帧
        for adas_group in adas_groups:
            time1 = datetime.strptime(adas_group.timestamp, '%Y-%m-%d %H:%M:%S.%f')
            time2 = datetime.strptime(target_timestamp, '%Y-%m-%d %H:%M:%S.%f')
            
            time_diff = abs((time1 - time2).total_seconds())  # 计算时间差
            if time_diff < closest_time_diff:
                closest_time_diff = time_diff
                closest_frame = adas_group
        
        return closest_frame

    # 找到与ADAS2时间戳接近的ADAS1和ADAS3帧，确保减少到ADAS2的帧数
    selected_adas1_frames = []
    selected_adas3_frames = []

    for adas2_group in adas2_groups:
        closest_adas1_frame = find_closest_frame(adas1_groups, adas2_group.timestamp)
        closest_adas3_frame = find_closest_frame(adas3_groups, adas2_group.timestamp)

        # 如果找到接近的帧，则添加到选中的帧列表中
        if closest_adas1_frame:
            selected_adas1_frames.append(closest_adas1_frame)
        if closest_adas3_frame:
            selected_adas3_frames.append(closest_adas3_frame)

        # 如果选中的帧数达到了ADAS2的帧数，则停止
        if len(selected_adas1_frames) >= len(adas2_groups) and len(selected_adas3_frames) >= len(adas2_groups):
            break

    # 如果选中的帧不足，尝试从其他帧中填充
    if len(selected_adas1_frames) < len(adas2_groups):
        remaining_adas1_frames = [frame for frame in adas1_groups if frame not in selected_adas1_frames]
        selected_adas1_frames.extend(remaining_adas1_frames[:len(adas2_groups) - len(selected_adas1_frames)])
        
    if len(selected_adas3_frames) < len(adas2_groups):
        remaining_adas3_frames = [frame for frame in adas3_groups if frame not in selected_adas3_frames]
        selected_adas3_frames.extend(remaining_adas3_frames[:len(adas2_groups) - len(selected_adas3_frames)])

    # 写入输出文件的函数
    def write_output(groups, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            for group in groups:
                f.write(''.join(group.lines))  # 将每组的行数据写入文件
                f.write('\n')  # 添加换行符

    # 保存ADAS1和ADAS3同步数据到文件
    write_output(selected_adas1_frames, f"{output_prefix}1_sync.txt")
    write_output(selected_adas3_frames, f"{output_prefix}3_sync.txt")

    return len(selected_adas1_frames), len(selected_adas3_frames)  # 返回选中的帧数量

# 使用示例
adas1_file = "/home/linux/ShenGang/feature_extraction/testdata/0-5/0-5静止电动车/ADAS1.txt"
adas2_file = "/home/linux/ShenGang/feature_extraction/testdata/0-5/0-5静止电动车/ADAS2.txt"
adas3_file = "/home/linux/ShenGang/feature_extraction/testdata/0-5/0-5静止电动车/ADAS3.txt"
output_prefix = "/home/linux/ShenGang/feature_extraction/testdata/0-5/0-5静止电动车/ADAS"



frame_count_adas1, frame_count_adas3 = synchronize_frames(adas1_file, adas2_file, adas3_file, output_prefix)
print(f"Selected frames for ADAS1: {frame_count_adas1}")
print(f"Selected frames for ADAS3: {frame_count_adas3}")
