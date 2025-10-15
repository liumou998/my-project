import re
import os
import datetime

def extract_sample_time(log_line):
    """
    从日志行中提取时间戳，支持多种格式
    """
    # 添加对标准日志时间格式的支持
    time_pattern = r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\]"
    match = re.search(time_pattern, log_line)
    if match:
        # 将日期时间转换为整数时间戳
        dt = datetime.datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S.%f")
        return int(dt.timestamp() * 1000)
    
    # 保留原有的提取逻辑作为备选
    time_pattern = r"Decode] sample time is (\d+)"
    match = re.search(time_pattern, log_line)
    if match:
        return int(match.group(1))
    return 0

def parse_state_codes(log_line):
    """
    从每一行日志中提取状态码信息，并进行0/1编码。
    """
    state_codes = {'FCW': 0, 'PCW': 0, 'LDW': 0}
    
    fcw_match = re.search(r'fcw-switch:\s*(\d+),\s*value:\s*(\d+)', log_line)
    if fcw_match:
        fcw_switch = int(fcw_match.group(1))
        state_codes['FCW'] = fcw_switch

    pcw_match = re.search(r'pcw-switch:\s*(\d+),\s*value:\s*(\d+)', log_line)
    if pcw_match:
        pcw_switch = int(pcw_match.group(1))
        state_codes['PCW'] = pcw_switch

    ldw_match = re.search(r'ldw-switch:\s*(\d+),\s*leftvalue:\s*(\d+),\s*rightvalue:\s*(\d+)', log_line)
    if ldw_match:
        ldw_switch = int(ldw_match.group(1))
        state_codes['LDW'] = ldw_switch

    return state_codes

def extract_frameid(log_line):
    """
    从日志行中提取frameid
    """
    frameid_match = re.search(r'frameid (\w+)', log_line)
    if frameid_match:
        return frameid_match.group(1)
    return None

def extract_obstacles(lines, start_idx, max_search_lines=1000):
    """
    从指定行开始，提取障碍物信息，返回障碍物列表
    
    Args:
        lines: 文件行的列表
        start_idx: 开始查找的行索引
        max_search_lines: 最大搜索行数，防止无限搜索
    
    Returns:
        障碍物列表，每个障碍物包含类别和坐标
    """
    obstacles = []
    
    # 搜索范围限制
    end_idx = min(start_idx + max_search_lines, len(lines))
    
    # 提取对象数量
    obj_num_line = lines[start_idx]
    obj_num_match = re.search(r'object-num (\d+)', obj_num_line)
    if not obj_num_match:
        return obstacles
    
    obj_num = int(obj_num_match.group(1))
    if obj_num == 0:
        return obstacles
    
    # 查找障碍物信息
    found_obstacles = 0
    i = start_idx + 1
    while i < end_idx and found_obstacles < obj_num:
        line = lines[i]
        if 'obs[' in line:
            try:
                # 提取类别信息
                class_match = re.search(r'class is (\d+)', line)
                class_num = int(class_match.group(1)) if class_match else 0
                
                # 提取x坐标
                x_match = re.search(r'x=([0-9.-]+)', line)
                x = float(x_match.group(1)) if x_match else 0.0
                
                # 提取y坐标
                y_match = re.search(r'y=([0-9.-]+)', line)
                y = float(y_match.group(1)) if y_match else 0.0
                
                obstacles.append((class_num, x, y))
                found_obstacles += 1
            except (IndexError, ValueError):
                pass
        i += 1
    
    return obstacles

def process_log_file(log_file_path, output_file_path, chunk_size=10000):
    """
    处理日志文件，提取时间戳、状态信息和障碍物数据，并保存到新的文件中。
    同一时间戳下的多个障碍物整合到同一行输出。
    
    Args:
        log_file_path: 输入日志文件路径
        output_file_path: 输出文件路径
        chunk_size: 每次处理的块大小
    """
    try:
        # 创建输出目录（如果不存在）
        output_dir = os.path.dirname(output_file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 打开输出文件
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            # 计算文件大小
            file_size = os.path.getsize(log_file_path)
            print(f"日志文件大小: {file_size / (1024 * 1024):.2f} MB")
            
            with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as file:
                # 读取整个文件内容到内存中
                lines = file.readlines()
            
            total_lines = len(lines)
            print(f"日志文件总行数: {total_lines}")
            
            # 追踪处理进度
            processed_lines = 0
            processed_frames = 0
            unique_frameids = set()
            
            # 存储当前时间戳下的数据
            current_timestamp = None
            current_data = {
                'FCW': 0,
                'PCW': 0,
                'LDW': 0,
                'obstacles': []
            }
            
            # 分块处理
            for chunk_start in range(0, total_lines, chunk_size):
                chunk_end = min(chunk_start + chunk_size, total_lines)
                print(f"正在处理行 {chunk_start+1} 到 {chunk_end} ({chunk_end/total_lines*100:.1f}%完成)")
                
                # 处理当前块中的每一行
                i = chunk_start
                while i < chunk_end:
                    line = lines[i]
                    
                    # 提取并转换时间戳
                    timestamp = extract_sample_time(line)
                    frameid = extract_frameid(line)
                    
                    # 如果同时找到时间戳和frameid
                    if timestamp > 0 and frameid and 'object-num' in line:
                        # 记录帧ID
                        if frameid not in unique_frameids:
                            unique_frameids.add(frameid)
                            processed_frames += 1
                        
                        # 如果时间戳改变，先写入之前的数据
                        if current_timestamp is not None and current_timestamp != timestamp:
                            # 写入当前数据到文件
                            write_data_to_file(output_file, current_timestamp, current_data)
                            # 重置当前数据
                            current_data = {
                                'FCW': 0,
                                'PCW': 0,
                                'LDW': 0,
                                'obstacles': []
                            }
                        
                        # 更新当前时间戳
                        current_timestamp = timestamp
                        
                        # 提取状态码并更新
                        state_codes = parse_state_codes(line)
                        current_data['FCW'] = max(current_data['FCW'], state_codes['FCW'])
                        current_data['PCW'] = max(current_data['PCW'], state_codes['PCW'])
                        current_data['LDW'] = max(current_data['LDW'], state_codes['LDW'])
                        
                        # 提取障碍物信息并添加到当前数据
                        obstacles = extract_obstacles(lines, i)
                        if obstacles:
                            current_data['obstacles'].extend(obstacles)
                    
                    i += 1
                
                processed_lines += (chunk_end - chunk_start)
                print(f"已处理 {processed_lines} 行, 找到 {processed_frames} 个唯一帧ID")
            
            # 处理最后一批数据
            if current_timestamp is not None:
                write_data_to_file(output_file, current_timestamp, current_data)
        
        print(f"处理完成！日志数据已保存到: {output_file_path}")
        print(f"总共处理了 {total_lines} 行，找到 {processed_frames} 个唯一帧ID")
        print(f"唯一帧ID数量: {len(unique_frameids)}")

    except FileNotFoundError as e:
        print(f"无法打开日志文件: {e}")
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        # 打印更详细的错误信息
        import traceback
        traceback.print_exc()

def write_data_to_file(output_file, timestamp, data):
    """
    将数据写入输出文件
    
    Args:
        output_file: 输出文件对象
        timestamp: 时间戳
        data: 包含状态和障碍物的数据
    """
    # 准备输出行
    fcw = data['FCW']
    pcw = data['PCW']
    ldw = data['LDW']
    obstacles = data['obstacles']
    
    # 排序障碍物（可选）
    # obstacles.sort(key=lambda x: (x[0], x[1]))  # 按类别和x坐标排序
    
    # 准备输出格式：时间戳 FCW PCW LDW 障碍物数量 障碍物1类别 障碍物1x 障碍物1y 障碍物2类别 障碍物2x 障碍物2y ...
    output_line = f"{timestamp}  {fcw}  {pcw}  {ldw}  {len(obstacles)}"
    
    # 添加障碍物信息
    for obs in obstacles:
        output_line += f"  {obs[0]}  {obs[1]}  {obs[2]}"
    
    # 确保有足够的列（20列）
    # 计算当前列数: 5个固定列 + 3列/每个障碍物
    current_cols = 5 + 3 * len(obstacles)
    padding_needed = max(0, 20 - current_cols)
    
    # 添加填充0
    for _ in range(padding_needed):
        output_line += "  0"
    
    # 写入文件
    output_file.write(output_line + "\n")

if __name__ == "__main__":
    log_file_path = '/home/linux/ShenGang/feature_extraction_new_construct/process/data_process/reduced_adas3.txt'  # 输入日志文件路径
    output_file_path = '/home/linux/ShenGang/feature_extraction_new_construct/process/data_process/reduced_adas3_process.txt'  # 输出日志文件路径
    
    # 设置块大小，可根据服务器内存调整
    chunk_size = 10000
    
    # 处理并将结果保存到新文件
    process_log_file(log_file_path, output_file_path, chunk_size)