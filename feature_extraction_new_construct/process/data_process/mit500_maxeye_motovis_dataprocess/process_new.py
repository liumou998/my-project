# import re
# import os

# def extract_sample_time(log_line):
#     """
#     从日志行中提取 sample time 字段，并返回对应的时间戳。
#     """
#     time_pattern = r"Decode] sample time is (\d+)"
#     match = re.search(time_pattern, log_line)
#     if match:
#         # 返回提取的 sample time
#         return int(match.group(1))
#     return 0  # 如果没有找到时间戳，返回 0

# def parse_state_codes(log_line):
#     """
#     从每一行日志中提取状态码信息，并进行0/1编码。
#     """
#     state_codes = {'FCW': 0, 'PCW': 0, 'LDW': 0}
    
#     # 使用正则表达式提取状态字段信息
#     fcw_match = re.search(r'fcw-switch:\s*(\d+),\s*value:\s*(\d+)', log_line)
#     if fcw_match:
#         fcw_switch = int(fcw_match.group(1))
#         state_codes['FCW'] = fcw_switch

#     pcw_match = re.search(r'pcw-switch:\s*(\d+),\s*value:\s*(\d+)', log_line)
#     if pcw_match:
#         pcw_switch = int(pcw_match.group(1))
#         state_codes['PCW'] = pcw_switch

#     ldw_match = re.search(r'ldw-switch:\s*(\d+),\s*value:\s*(\d+)', log_line)
#     if ldw_match:
#         ldw_switch = int(ldw_match.group(1))
#         state_codes['LDW'] = ldw_switch

#     return state_codes

# def process_log_file(log_file_path, output_file_path, batch_size=1000):
#     """
#     处理日志文件，提取 sample time、状态信息和障碍物数据，并保存到新的文件中。
#     使用批处理方式减少内存占用。
    
#     Args:
#         log_file_path: 输入日志文件路径
#         output_file_path: 输出文件路径
#         batch_size: 每次处理的批次大小
#     """
#     try:
#         # 创建输出文件夹（如果不存在）
#         output_dir = os.path.dirname(output_file_path)
#         if not os.path.exists(output_dir):
#             os.makedirs(output_dir)

#         # 打开输出文件
#         with open(output_file_path, 'w', encoding='utf-8') as output_file:
#             # 第一次扫描，找出所有有效的时间戳行和对应的状态码
#             timestamp_states = {}
#             print("第一次扫描: 提取时间戳和状态码...")
            
#             with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as file:
#                 for line_num, line in enumerate(file):
#                     if line_num % 10000 == 0:
#                         print(f"正在处理第 {line_num} 行...")
                    
#                     # 提取并转换 sample time
#                     unix_timestamp = extract_sample_time(line)
#                     if unix_timestamp == 0:
#                         continue  # 如果没有找到时间戳，则跳过该行
                    
#                     # 如果时间戳已存在，跳过（保留第一次出现的）
#                     if unix_timestamp in timestamp_states:
#                         continue
                    
#                     # 提取状态码信息
#                     state_codes = parse_state_codes(line)
#                     timestamp_states[unix_timestamp] = state_codes
            
#             print(f"找到 {len(timestamp_states)} 个唯一时间戳。")
            
#             # 第二次扫描，提取障碍物信息
#             print("第二次扫描: 提取障碍物信息...")
#             current_timestamp = None
#             current_state = None
#             obstacles_batch = []
            
#             with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as file:
#                 lines = file.readlines()
#                 i = 0
                
#                 while i < len(lines):
#                     if i % 10000 == 0:
#                         print(f"正在处理第 {i} 行...")
                    
#                     line = lines[i]
                    
#                     # 检查是否有新的时间戳
#                     unix_timestamp = extract_sample_time(line)
#                     if unix_timestamp > 0 and unix_timestamp in timestamp_states:
#                         current_timestamp = unix_timestamp
#                         current_state = timestamp_states[unix_timestamp]
                    
#                     # 如果当前行包含 object-num 且我们有有效的时间戳
#                     if current_timestamp and 'object-num' in line:
#                         try:
#                             # 提取 object-num 的值
#                             obj_num = int(line.split('object-num')[-1].strip())
                            
#                             # 初始化障碍物数据
#                             obstacles_data = [obj_num]  # 保留 object-num 的信息
                            
#                             if obj_num == 0:
#                                 # 如果没有障碍物，添加一行全0数据
#                                 obstacles_data.extend([0] * 15)  # 5组 [类别,x,y]
                                
#                                 # 保存到批处理
#                                 obstacles_batch.append({
#                                     'timestamp': current_timestamp,
#                                     'FCW': current_state['FCW'],
#                                     'PCW': current_state['PCW'],
#                                     'LDW': current_state['LDW'],
#                                     'object-num': obstacles_data[0],
#                                     'obstacles': obstacles_data[1:]
#                                 })
#                             else:
#                                 # 如果有障碍物，收集接下来的 obj_num 行障碍物数据
#                                 found_obstacles = 0
#                                 j = i + 1
                                
#                                 while j < len(lines) and found_obstacles < obj_num:
#                                     obs_line = lines[j]
#                                     if 'obs[' in obs_line:
#                                         try:
#                                             # 提取障碍物信息：类别、x、y
#                                             class_num = int(obs_line.split('class is ')[1][0])
#                                             x = float(obs_line.split('x=')[1].split(',')[0])
#                                             y = float(obs_line.split('y=')[1])
#                                             obstacles_data.extend([class_num, x, y])
#                                             found_obstacles += 1
#                                         except (IndexError, ValueError) as e:
#                                             # 优雅地处理错误，继续处理下一行
#                                             print(f"警告: 处理障碍物数据时出错: {e}, 行: {j+1}")
#                                     j += 1
                                
#                                 # 如果数据不足5组，补充0
#                                 while len(obstacles_data) < 16:  # 1个 object-num + 15个障碍物数据
#                                     obstacles_data.extend([0, 0, 0])  # 补充每个障碍物的 [类别,x,y]
                                
#                                 # 保存到批处理
#                                 obstacles_batch.append({
#                                     'timestamp': current_timestamp,
#                                     'FCW': current_state['FCW'],
#                                     'PCW': current_state['PCW'],
#                                     'LDW': current_state['LDW'],
#                                     'object-num': obstacles_data[0],
#                                     'obstacles': obstacles_data[1:]
#                                 })
                                
#                                 # 更新索引以跳过已处理的障碍物行
#                                 i = j - 1  # 因为循环结束后会 i += 1
#                         except Exception as e:
#                             print(f"处理 object-num 数据时出错: {e}, 行: {i+1}")
                    
#                     # 当批处理达到指定大小时写入文件
#                     if len(obstacles_batch) >= batch_size:
#                         for entry in obstacles_batch:
#                             obstacles_str = '   '.join(map(str, entry['obstacles']))
#                             output_file.write(f"{entry['timestamp']}  {entry['FCW']}  {entry['PCW']}  {entry['LDW']}  {entry['object-num']}  {obstacles_str}\n")
#                         obstacles_batch = []  # 清空批处理
                    
#                     i += 1
                
#                 # 写入剩余的批处理数据
#                 for entry in obstacles_batch:
#                     obstacles_str = '   '.join(map(str, entry['obstacles']))
#                     output_file.write(f"{entry['timestamp']}  {entry['FCW']}  {entry['PCW']}  {entry['LDW']}  {entry['object-num']}  {obstacles_str}\n")
        
#         print(f"日志数据处理完成，已保存到: {output_file_path}")

#     except FileNotFoundError as e:
#         print(f"打开日志文件时出错: {e}")
#     except Exception as e:
#         print(f"发生错误: {e}")

# if __name__ == "__main__":
#     # 参数设置
#     log_file_path = '/home/linux/ShenGang/feature_extraction_new_construct/process/data_process/dataprocess/ADAS1_sync.txt'  # 输入日志文件路径
#     output_file_path = '/home/linux/ShenGang/feature_extraction_new_construct/process/data_process/dataprocess/adas1_process.txt'  # 输出日志文件路径
#     batch_size = 1000  # 批处理大小
    
#     # 处理文件
#     process_log_file(log_file_path, output_file_path, batch_size)

import re
import os

def extract_sample_time(log_line):
    """
    从日志行中提取 sample time 字段，并返回对应的时间戳。
    """
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

    ldw_match = re.search(r'ldw-switch:\s*(\d+),\s*value:\s*(\d+)', log_line)
    if ldw_match:
        ldw_switch = int(ldw_match.group(1))
        state_codes['LDW'] = ldw_switch

    return state_codes

def extract_obstacles_for_timestamp(lines, start_idx, max_search_lines=1000):
    """
    从指定行开始，提取障碍物信息，并转换为矩阵形式
    
    Args:
        lines: 文件行的列表
        start_idx: 开始查找的行索引
        max_search_lines: 最大搜索行数，防止无限搜索
    
    Returns:
        包含障碍物数据的矩阵
    """
    result_matrix = []
    
    # 搜索范围限制
    end_idx = min(start_idx + max_search_lines, len(lines))
    
    for i in range(start_idx, end_idx):
        line = lines[i]
        if 'object-num' in line:
            try:
                obj_num = int(line.split('object-num')[-1].strip())
                obj_num_original = obj_num  # 保存原始值用于计数
                
                obstacles_data = [obj_num]  # 直接使用原始数值，不应用阈值
                
                if obj_num_original == 0:
                    obstacles_data.extend([0] * 15)
                else:
                    found_obstacles = 0
                    j = i + 1
                    # 限制搜索范围，防止搜索过多行
                    while j < end_idx and found_obstacles < obj_num_original:
                        obs_line = lines[j]
                        if 'obs[' in obs_line:
                            try:
                                class_num = int(obs_line.split('class is ')[1][0])
                                x = float(obs_line.split('x=')[1].split(',')[0])
                                y = float(obs_line.split('y=')[1])
                                
                                # 直接使用原始值，不应用阈值
                                obstacles_data.extend([class_num, x, y])
                                found_obstacles += 1
                            except (IndexError, ValueError) as e:
                                # 忽略错误，继续处理下一行
                                pass
                        j += 1
                    
                    # 如果数据不足5组，补充0
                    while len(obstacles_data) < 16:
                        obstacles_data.extend([0, 0, 0])
                
                result_matrix.append(obstacles_data)
                # 找到一个 object-num 后就返回，这符合原有逻辑
                break
            except (ValueError, IndexError) as e:
                # 忽略错误，继续搜索下一行
                continue
    
    return result_matrix

def process_log_file(log_file_path, output_file_path, chunk_size=10000):
    """
    处理日志文件，提取 sample time、状态信息和障碍物数据，并保存到新的文件中。
    使用分块处理来减少内存使用。
    
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
            # 计算文件大小和行数
            file_size = os.path.getsize(log_file_path)
            print(f"日志文件大小: {file_size / (1024*1024):.2f} MB")
            
            with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as file:
                # 读取整个文件内容到内存中
                lines = file.readlines()
            
            total_lines = len(lines)
            print(f"日志文件总行数: {total_lines}")
            
            # 追踪处理进度
            processed_lines = 0
            processed_timestamps = 0
            last_timestamp = None
            
            # 分块处理
            for chunk_start in range(0, total_lines, chunk_size):
                chunk_end = min(chunk_start + chunk_size, total_lines)
                print(f"正在处理行 {chunk_start+1} 到 {chunk_end} ({chunk_end/total_lines*100:.1f}%完成)")
                
                # 处理当前块中的每一行
                i = chunk_start
                while i < chunk_end:
                    line = lines[i]
                    
                    # 提取并转换 sample time
                    unix_timestamp = extract_sample_time(line)
                    if unix_timestamp == 0 or unix_timestamp == last_timestamp:
                        i += 1
                        continue  # 如果没有找到时间戳或时间戳重复，则跳过该行
                    
                    # 更新上次处理的时间戳
                    last_timestamp = unix_timestamp
                    processed_timestamps += 1
                    
                    # 提取状态码信息
                    state_codes = parse_state_codes(line)
                    
                    # 提取障碍物数据（只在时间戳行后面查找）
                    obstacles_data = extract_obstacles_for_timestamp(lines, i)
                    
                    # 将处理后的数据直接写入文件
                    for obstacles in obstacles_data:
                        obstacles_str = '   '.join(map(str, obstacles[1:]))
                        output_file.write(f"{unix_timestamp}  {state_codes['FCW']}  {state_codes['PCW']}  {state_codes['LDW']}  {obstacles[0]}  {obstacles_str}\n")
                    
                    i += 1
                
                processed_lines += (chunk_end - chunk_start)
                print(f"已处理 {processed_lines} 行, 找到 {processed_timestamps} 个唯一时间戳")
        
        print(f"处理完成！日志数据已保存到: {output_file_path}")
        print(f"总共处理了 {total_lines} 行，找到 {processed_timestamps} 个唯一时间戳")

    except FileNotFoundError as e:
        print(f"无法打开日志文件: {e}")
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        # 打印更详细的错误信息
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    log_file_path = '/home/linux/ShenGang/feature_extraction_new_construct/process/data_process/dataprocess/ADAS3.txt'  # 输入日志文件路径
    output_file_path = '/home/linux/ShenGang/feature_extraction_new_construct/process/data_process/dataprocess/adas3_process.txt'  # 输出日志文件路径
    
    # 设置块大小，可根据服务器内存调整
    chunk_size = 10000
    
    # 处理并将结果保存到新文件
    process_log_file(log_file_path, output_file_path, chunk_size)