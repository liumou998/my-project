# import os
# import numpy as np
# from collections import defaultdict

# def get_file_dimension(filename):
#     """获取文件的数据维度（列数）"""
#     with open(filename, 'r') as f:
#         first_line = f.readline().strip()
#         return len(first_line.split())

# def read_file_with_timestamps(filename):
#     """读取文件，返回时间戳到数据的映射"""
#     time_data_map = defaultdict(list)
#     with open(filename, 'r') as f:
#         for line in f:
#             values = line.strip().split()
#             timestamp = int(values[0])
#             data = [float(x) for x in values[1:]]
#             time_data_map[timestamp].append(data)
#     return time_data_map

# def create_output_dirs():
#     """创建输出目录"""
#     os.makedirs('aligned_data1', exist_ok=True)
#     os.makedirs('aligned_data2', exist_ok=True)

# def align_and_save_data(file1_path, file2_path):
#     """对齐数据并保存"""
#     # 获取两个文件的维度
#     dim1 = get_file_dimension(file1_path)
#     dim2 = get_file_dimension(file2_path)
    
#     print(f"文件1维度: {dim1}")
#     print(f"文件2维度: {dim2}")
    
#     # 确定基准文件（维度较小的文件）
#     if dim1 <= dim2:
#         reference_file = file1_path
#         target_file = file2_path
#         reference_dim = dim1
#         is_file1_reference = True
#     else:
#         reference_file = file2_path
#         target_file = file1_path
#         reference_dim = dim2
#         is_file1_reference = False
    
#     # 读取文件数据
#     reference_data = read_file_with_timestamps(reference_file)
#     target_data = read_file_with_timestamps(target_file)
    
#     # 获取基准文件的时间戳
#     reference_timestamps = sorted(reference_data.keys())
    
#     # 创建输出目录
#     create_output_dirs()
    
#     # 准备输出文件
#     output_file1 = open('/home/linux/ShenGang/feature_extraction_new_construct/process/data_process/dataprocess/adas1_aligned.txt', 'w')
#     output_file2 = open('/home/linux/ShenGang/feature_extraction_new_construct/process/data_process/dataprocess/adas2_aligned.txt', 'w')
    
#     try:
#         # 按基准文件的时间戳进行对齐
#         for timestamp in reference_timestamps:
#             # 获取基准数据（取平均，如果有多条）
#             ref_data = np.mean(reference_data[timestamp], axis=0)
            
#             # 检查目标文件是否有对应时间戳的数据
#             if timestamp in target_data:
#                 target_data_avg = np.mean(target_data[timestamp], axis=0)
                
#                 # 根据哪个是基准文件来决定写入顺序
#                 if is_file1_reference:
#                     output_file1.write(f"{timestamp} {' '.join(map(str, ref_data))}\n")
#                     output_file2.write(f"{timestamp} {' '.join(map(str, target_data_avg[:reference_dim-1]))}\n")
#                 else:
#                     output_file1.write(f"{timestamp} {' '.join(map(str, target_data_avg[:reference_dim-1]))}\n")
#                     output_file2.write(f"{timestamp} {' '.join(map(str, ref_data))}\n")
    
#     finally:
#         # 关闭文件
#         output_file1.close()
#         output_file2.close()
    
#     # 统计处理结果
#     return len(reference_timestamps)

# def main():
#     # 文件路径配置
#     file1_path = '/home/linux/ShenGang/feature_extraction_new_construct/process/data_process/dataprocess/adas1_process.txt'
#     file2_path = '/home/linux/ShenGang/feature_extraction_new_construct/process/data_process/dataprocess/adas2_process.txt'
    
#     try:
#         # 对齐数据并保存
#         processed_lines = align_and_save_data(file1_path, file2_path)
        
#         print("\n处理完成！")
#         print(f"处理的数据行数: {processed_lines}")
#         print("结果已保存至 aligned_data1/aligned.txt 和 aligned_data2/aligned.txt")
        
#     except Exception as e:
#         print(f"处理过程中发生错误: {str(e)}")

# if __name__ == "__main__":
#     main()




# import os
# import numpy as np
# from collections import defaultdict

# def get_file_dimension(filename):
#     """获取文件的数据维度（列数）"""
#     with open(filename, 'r') as f:
#         first_line = f.readline().strip()
#         return len(first_line.split())

# def read_file_with_timestamps(filename):
#     """读取文件，返回时间戳到数据的映射"""
#     time_data_map = defaultdict(list)
#     line_count = 0
    
#     with open(filename, 'r') as f:
#         for line_num, line in enumerate(f, 1):
#             try:
#                 values = line.strip().split()
#                 timestamp = int(values[0])
                
#                 # 处理数据，将类似 '50+' 的值转换为浮点数
#                 data = []
#                 for val in values[1:]:
#                     try:
#                         # 如果值包含加号，只保留数字部分
#                         if '+' in val:
#                             val = val.replace('+', '')
#                         data.append(float(val))
#                     except ValueError:
#                         print(f"警告：在文件 {os.path.basename(filename)} 第 {line_num} 行中的值 '{val}' 无法转换为浮点数，将使用0替代")
#                         data.append(0.0)  # 使用0替代无法转换的值
                
#                 time_data_map[timestamp].append(data)
#                 line_count += 1
#             except Exception as e:
#                 print(f"警告：跳过文件 {os.path.basename(filename)} 第 {line_num} 行，原因: {str(e)}")
#                 continue
    
#     print(f"文件 {os.path.basename(filename)} 共读取 {line_count} 行数据")
#     return time_data_map, line_count

# def find_closest_timestamp(target_time, available_times, max_diff=100):
#     """找到最接近的时间戳，允许一定的误差范围"""
#     closest_time = None
#     min_diff = float('inf')
    
#     for time in available_times:
#         diff = abs(time - target_time)
#         if diff < min_diff and diff <= max_diff:
#             min_diff = diff
#             closest_time = time
    
#     return closest_time

# def align_with_flexible_timestamps(file_paths, output_paths, time_tolerance=100):
#     """使用灵活的时间戳匹配对齐多个文件"""
#     # 读取所有文件数据
#     all_data = []
#     line_counts = []
    
#     for file_path in file_paths:
#         data, count = read_file_with_timestamps(file_path)
#         all_data.append(data)
#         line_counts.append(count)
    
#     # 确定基准文件（行数最少的文件）
#     reference_idx = line_counts.index(min(line_counts))
#     reference_data = all_data[reference_idx]
#     reference_timestamps = sorted(reference_data.keys())
    
#     print(f"使用文件 {os.path.basename(file_paths[reference_idx])} 作为基准（{len(reference_timestamps)} 行）")
#     print(f"时间戳容差设置为 {time_tolerance}")
    
#     # 打开输出文件
#     output_files = []
#     try:
#         for output_path in output_paths:
#             output_files.append(open(output_path, 'w'))
        
#         # 处理计数
#         processed_count = 0
#         skipped_count = 0
        
#         # 对每个基准时间戳，在其他文件中找最接近的时间戳
#         for timestamp in reference_timestamps:
#             match_found = True
#             matched_data = [None] * len(all_data)
            
#             # 对基准文件，直接使用当前时间戳的数据
#             matched_data[reference_idx] = np.mean(reference_data[timestamp], axis=0)
            
#             # 对其他文件，查找最近的时间戳
#             for i, data in enumerate(all_data):
#                 if i == reference_idx:
#                     continue  # 跳过基准文件
                
#                 # 在当前文件中查找最接近的时间戳
#                 closest_time = find_closest_timestamp(timestamp, data.keys(), time_tolerance)
                
#                 if closest_time is not None:
#                     matched_data[i] = np.mean(data[closest_time], axis=0)
#                 else:
#                     match_found = False
#                     break
            
#             # 如果所有文件都找到了匹配的时间戳，写入数据
#             if match_found:
#                 for i, data in enumerate(matched_data):
#                     output_files[i].write(f"{timestamp} {' '.join(map(str, data))}\n")
#                 processed_count += 1
#             else:
#                 skipped_count += 1
    
#     finally:
#         # 关闭所有输出文件
#         for f in output_files:
#             f.close()
    
#     return processed_count, skipped_count

# def main():
#     # 文件路径配置
#     file_paths = [
#         '/home/linux/ShenGang/feature_extraction_new_construct/process/data_process/dataprocess/adas1_process.txt',
#         '/home/linux/ShenGang/feature_extraction_new_construct/process/data_process/dataprocess/adas2_process.txt',
#         '/home/linux/ShenGang/feature_extraction_new_construct/process/data_process/dataprocess/adas3_process.txt'
#     ]
    
#     output_paths = [
#         '/home/linux/ShenGang/feature_extraction_new_construct/process/data_process/dataprocess/adas1_aligned.txt',
#         '/home/linux/ShenGang/feature_extraction_new_construct/process/data_process/dataprocess/adas2_aligned.txt',
#         '/home/linux/ShenGang/feature_extraction_new_construct/process/data_process/dataprocess/adas3_aligned.txt'
#     ]
    
#     try:
#         # 设置时间戳容差（单位可能是毫秒或其他，根据你的数据调整）
#         time_tolerance = 100  # 可根据你的数据特点调整这个值
        
#         # 对齐数据并保存
#         processed_lines, skipped_lines = align_with_flexible_timestamps(
#             file_paths, output_paths, time_tolerance)
        
#         print("\n处理完成！")
#         print(f"成功对齐并处理的行数: {processed_lines}")
#         print(f"由于无法找到匹配的时间戳而跳过的行数: {skipped_lines}")
#         print("结果已保存至:")
#         for path in output_paths:
#             print(f"- {path}")
        
#     except Exception as e:
#         print(f"处理过程中发生错误: {str(e)}")
#         import traceback
#         traceback.print_exc()

# if __name__ == "__main__":
#     main()


# import os
# import numpy as np
# from collections import defaultdict

# def get_file_dimension(filename):
#     """获取文件的数据维度（列数）"""
#     with open(filename, 'r') as f:
#         first_line = f.readline().strip()
#         return len(first_line.split())

# def read_file_with_timestamps(filename):
#     """读取文件，返回时间戳到数据的映射"""
#     time_data_map = defaultdict(list)
#     line_count = 0
    
#     with open(filename, 'r') as f:
#         for line_num, line in enumerate(f, 1):
#             try:
#                 values = line.strip().split()
#                 timestamp = int(values[0])
                
#                 # 处理数据，将类似 '50+' 的值转换为浮点数
#                 data = []
#                 for val in values[1:]:
#                     try:
#                         # 如果值包含加号，只保留数字部分
#                         if '+' in val:
#                             val = val.replace('+', '')
#                         data.append(float(val))
#                     except ValueError:
#                         print(f"警告：在文件 {os.path.basename(filename)} 第 {line_num} 行中的值 '{val}' 无法转换为浮点数，将使用0替代")
#                         data.append(0.0)  # 使用0替代无法转换的值
                
#                 time_data_map[timestamp].append(data)
#                 line_count += 1
#             except Exception as e:
#                 print(f"警告：跳过文件 {os.path.basename(filename)} 第 {line_num} 行，原因: {str(e)}")
#                 continue
    
#     print(f"文件 {os.path.basename(filename)} 共读取 {line_count} 行数据")
#     return time_data_map, line_count

# def find_closest_timestamp(target_time, available_times, max_diff=100):
#     """找到最接近的时间戳，允许一定的误差范围"""
#     closest_time = None
#     min_diff = float('inf')
    
#     for time in available_times:
#         diff = abs(time - target_time)
#         if diff < min_diff and diff <= max_diff:
#             min_diff = diff
#             closest_time = time
    
#     return closest_time

# def align_with_flexible_timestamps(file_paths, output_paths, time_tolerance=100):
#     """使用灵活的时间戳匹配对齐多个文件"""
#     # 读取所有文件数据
#     all_data = []
#     line_counts = []
    
#     for file_path in file_paths:
#         data, count = read_file_with_timestamps(file_path)
#         all_data.append(data)
#         line_counts.append(count)
    
#     # 确定基准文件（行数最少的文件）
#     reference_idx = line_counts.index(min(line_counts))
#     reference_data = all_data[reference_idx]
#     reference_timestamps = sorted(reference_data.keys())
    
#     print(f"使用文件 {os.path.basename(file_paths[reference_idx])} 作为基准（{len(reference_timestamps)} 行）")
#     print(f"时间戳容差设置为 {time_tolerance}")
    
#     # 打开输出文件
#     output_files = []
#     try:
#         for output_path in output_paths:
#             output_files.append(open(output_path, 'w'))
        
#         # 处理计数
#         processed_count = 0
#         skipped_count = 0
        
#         # 对每个基准时间戳，在其他文件中找最接近的时间戳
#         for timestamp in reference_timestamps:
#             match_found = True
#             matched_data = [None] * len(all_data)
            
#             # 对基准文件，直接使用当前时间戳的数据
#             matched_data[reference_idx] = np.mean(reference_data[timestamp], axis=0)
            
#             # 对其他文件，查找最近的时间戳
#             for i, data in enumerate(all_data):
#                 if i == reference_idx:
#                     continue  # 跳过基准文件
                
#                 # 在当前文件中查找最接近的时间戳
#                 closest_time = find_closest_timestamp(timestamp, data.keys(), time_tolerance)
                
#                 if closest_time is not None:
#                     matched_data[i] = np.mean(data[closest_time], axis=0)
#                 else:
#                     match_found = False
#                     break
            
#                             # 如果所有文件都找到了匹配的时间戳，写入数据
#             if match_found:
#                 for i, data in enumerate(matched_data):
#                     # 保持原始数据格式，包括小数点
#                     data_str = [str(x) for x in data]
#                     output_files[i].write(f"{timestamp} {' '.join(data_str)}\n")
#                 processed_count += 1
#             else:
#                 skipped_count += 1
    
#     finally:
#         # 关闭所有输出文件
#         for f in output_files:
#             f.close()
    
#     return processed_count, skipped_count

# def main():
#     # 文件路径配置
#     file_paths = [
#         '/home/linux/ShenGang/feature_extraction_new_construct/process/data_process/dataprocess/adas1_process.txt',
#         '/home/linux/ShenGang/feature_extraction_new_construct/process/data_process/dataprocess/adas2_process.txt',
#         '/home/linux/ShenGang/feature_extraction_new_construct/process/data_process/dataprocess/adas3_process.txt'
#     ]
    
#     output_paths = [
#         '/home/linux/ShenGang/feature_extraction_new_construct/process/data_process/dataprocess/adas1_aligned.txt',
#         '/home/linux/ShenGang/feature_extraction_new_construct/process/data_process/dataprocess/adas2_aligned.txt',
#         '/home/linux/ShenGang/feature_extraction_new_construct/process/data_process/dataprocess/adas3_aligned.txt'
#     ]
    
#     try:
#         # 设置时间戳容差（单位可能是毫秒或其他，根据你的数据调整）
#         time_tolerance = 100  # 可根据你的数据特点调整这个值
        
#         # 对齐数据并保存
#         processed_lines, skipped_lines = align_with_flexible_timestamps(
#             file_paths, output_paths, time_tolerance)
        
#         print("\n处理完成！")
#         print(f"成功对齐并处理的行数: {processed_lines}")
#         print(f"由于无法找到匹配的时间戳而跳过的行数: {skipped_lines}")
#         print("结果已保存至:")
#         for path in output_paths:
#             print(f"- {path}")
        
#     except Exception as e:
#         print(f"处理过程中发生错误: {str(e)}")
#         import traceback
#         traceback.print_exc()

# if __name__ == "__main__":
#     main()

import os
import numpy as np
from collections import defaultdict

def get_file_dimension(filename):
    """获取文件的数据维度（列数）"""
    with open(filename, 'r') as f:
        first_line = f.readline().strip()
        return len(first_line.split())

def read_file_with_timestamps(filename):
    """读取文件，返回时间戳到数据的映射"""
    time_data_map = defaultdict(list)
    line_count = 0
    
    with open(filename, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                values = line.strip().split()
                timestamp = int(values[0])
                
                # 处理数据，将类似 '50+' 的值转换为浮点数
                data = []
                for val in values[1:]:
                    try:
                        # 如果值包含加号，只保留数字部分
                        if '+' in val:
                            val = val.replace('+', '')
                        data.append(float(val))
                    except ValueError:
                        print(f"警告：在文件 {os.path.basename(filename)} 第 {line_num} 行中的值 '{val}' 无法转换为浮点数，将使用0替代")
                        data.append(0.0)  # 使用0替代无法转换的值
                
                time_data_map[timestamp].append(data)
                line_count += 1
            except Exception as e:
                print(f"警告：跳过文件 {os.path.basename(filename)} 第 {line_num} 行，原因: {str(e)}")
                continue
    
    print(f"文件 {os.path.basename(filename)} 共读取 {line_count} 行数据")
    return time_data_map, line_count

def find_closest_timestamp(target_time, available_times, max_diff=100):
    """找到最接近的时间戳，允许一定的误差范围"""
    closest_time = None
    min_diff = float('inf')
    
    for time in available_times:
        diff = abs(time - target_time)
        if diff < min_diff and diff <= max_diff:
            min_diff = diff
            closest_time = time
    
    return closest_time

def format_value(val):
    """格式化数值，去除小数点后的零"""
    if val == int(val):  # 如果是整数
        return str(int(val))
    else:
        return str(val)

def align_to_target_file(file_paths, output_paths, target_file_idx=2, time_tolerance=100):
    """将所有文件对齐到目标文件（默认为adas3）的全部行数"""
    # 读取所有文件数据
    all_data = []
    line_counts = []
    
    for file_path in file_paths:
        data, count = read_file_with_timestamps(file_path)
        all_data.append(data)
        line_counts.append(count)
    
    # 使用指定的目标文件（默认是adas3，索引为2）作为基准
    target_data = all_data[target_file_idx]
    target_timestamps = sorted(target_data.keys())
    
    print(f"使用文件 {os.path.basename(file_paths[target_file_idx])} 作为基准（{len(target_timestamps)} 行）")
    print(f"时间戳容差设置为 {time_tolerance}")
    
    # 打开输出文件
    output_files = []
    try:
        for output_path in output_paths:
            output_files.append(open(output_path, 'w'))
        
        # 处理计数
        processed_count = 0
        filled_count = 0
        
        # 对每个目标文件的时间戳，在其他文件中找最接近的时间戳
        for timestamp in target_timestamps:
            matched_data = [None] * len(all_data)
            filled = False
            
            # 对目标文件，直接使用当前时间戳的数据
            matched_data[target_file_idx] = np.mean(target_data[timestamp], axis=0)
            
            # 对其他文件，查找最近的时间戳
            for i, data in enumerate(all_data):
                if i == target_file_idx:
                    continue  # 跳过目标文件
                
                # 在当前文件中查找最接近的时间戳
                closest_time = find_closest_timestamp(timestamp, data.keys(), time_tolerance)
                
                if closest_time is not None:
                    matched_data[i] = np.mean(data[closest_time], axis=0)
                else:
                    # 如果找不到匹配的时间戳，使用零值填充
                    matched_data[i] = np.zeros(len(matched_data[target_file_idx]))
                    filled = True
            
            # 写入所有数据（无论是否找到匹配）
            for i, data in enumerate(matched_data):
                # 格式化数据，去除小数点后的零
                formatted_data = [format_value(x) for x in data]
                output_files[i].write(f"{timestamp} {' '.join(formatted_data)}\n")
            
            processed_count += 1
            if filled:
                filled_count += 1
    
    finally:
        # 关闭所有输出文件
        for f in output_files:
            f.close()
    
    return processed_count, filled_count

def main():
    # 文件路径配置
    file_paths = [
        '/home/linux/ShenGang/feature_extraction_new_construct/process/data_process/dataprocess/adas1_process.txt',
        '/home/linux/ShenGang/feature_extraction_new_construct/process/data_process/dataprocess/adas2_process.txt',
        '/home/linux/ShenGang/feature_extraction_new_construct/process/data_process/dataprocess/adas3_process.txt'
    ]
    
    output_paths = [
        '/home/linux/ShenGang/feature_extraction_new_construct/process/data_process/dataprocess/adas1_aligned.txt',
        '/home/linux/ShenGang/feature_extraction_new_construct/process/data_process/dataprocess/adas2_aligned.txt',
        '/home/linux/ShenGang/feature_extraction_new_construct/process/data_process/dataprocess/adas3_aligned.txt'
    ]
    
    try:
        # 设置时间戳容差（单位可能是毫秒或其他，根据你的数据调整）
        time_tolerance = 100  # 可根据你的数据特点调整这个值
        
        # 对齐数据到adas3（索引为2）并保存，匹配原始数据格式
        processed_lines, filled_lines = align_to_target_file(
            file_paths, output_paths, target_file_idx=2, time_tolerance=time_tolerance)
        
        print("\n处理完成！")
        print(f"成功处理的总行数: {processed_lines}")
        print(f"其中需要零值填充的行数: {filled_lines}")
        print("结果已保存至:")
        for path in output_paths:
            print(f"- {path}")
        
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
