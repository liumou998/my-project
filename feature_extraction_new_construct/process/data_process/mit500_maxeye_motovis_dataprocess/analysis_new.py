# import re

# logfile = '/home/linux/ShenGang/feature_extraction_new_construct/data_process/new_log_sys/analysis/log_process/combined_sys.log'  # 日志文件路径
# output_file_1 = '/home/linux/ShenGang/feature_extraction_new_construct/data_process/new_log_sys/analysis/log_process/ADAS1.txt'  # 输出文件1
# output_file_2 = '/home/linux/ShenGang/feature_extraction_new_construct/data_process/new_log_sys/analysis/log_process/ADAS2.txt'  # 输出文件2
# output_file_3 = '/home/linux/ShenGang/feature_extraction_new_construct/data_process/new_log_sys/analysis/log_process/ADAS3.txt'  # 输出文件3

# # 处理并保存数据
# def retain_lines_by_port_type(file1, output_file_1, output_file_2, output_file_3):
#     with open(file1, 'r', encoding='utf-8') as f:
#         lines = f.readlines()

#     # 打开三个输出文件
#     out1 = open(output_file_1, 'w', encoding='utf-8')
#     out2 = open(output_file_2, 'w', encoding='utf-8')
#     out3 = open(output_file_3, 'w', encoding='utf-8')

#     current_type = None  # 当前正在处理的port_type
#     i = 0
#     while i < len(lines):
#         match = re.search(r'prot_type (\d)', lines[i])
#         if match:
#             port_type = f'prot_type {match.group(1)}'

#             # 判断当前的port_type，并写入相应的文件
#             if port_type == 'prot_type 3':
#                 out1.write(lines[i])  # 写入ADAS1
#                 # 处理后续行直到遇到下一个port_type
#                 j = i + 1
#                 while j < len(lines) and not re.search(r'prot_type (\d)', lines[j]):
#                     out1.write(lines[j])  # 写入ADAS1
#                     j += 1
#                 i = j  # 跳过已处理的行

#             elif port_type == 'prot_type 7':
#                 out2.write(lines[i])  # 写入ADAS2
#                 # 处理后续行直到遇到下一个port_type
#                 j = i + 1
#                 while j < len(lines) and not re.search(r'prot_type (\d)', lines[j]):
#                     out2.write(lines[j])  # 写入ADAS2
#                     j += 1
#                 i = j  # 跳过已处理的行

#             elif port_type == 'prot_type 2':
#                 out3.write(lines[i])  # 写入ADAS3
#                 # 处理后续行直到遇到下一个port_type
#                 j = i + 1
#                 while j < len(lines) and not re.search(r'prot_type (\d)', lines[j]):
#                     out3.write(lines[j])  # 写入ADAS3
#                     j += 1
#                 i = j  # 跳过已处理的行

#         else:
#             i += 1

#     # 关闭输出文件
#     out1.close()
#     out2.close()
#     out3.close()

# # 执行分类并保存到相应的文件
# retain_lines_by_port_type(logfile, output_file_1, output_file_2, output_file_3)





import re
import os

# 日志文件路径 - 修改为合并后的日志文件路径
logfile = '/home/linux/ShenGang/feature_extraction_new_construct/process/data_process2/analysis/log_process/combined_sys1.log'  # 更新为合并后的日志路径

# 输出文件路径
output_dir = '/home/linux/ShenGang/feature_extraction_new_construct/process/data_process2/analysis/log_process'
output_file_1 = os.path.join(output_dir, 'ADAS1.txt')  # 输出文件1
output_file_2 = os.path.join(output_dir, 'ADAS2.txt')  # 输出文件2
output_file_3 = os.path.join(output_dir, 'ADAS3.txt')  # 输出文件3

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

def retain_lines_by_port_type(input_file, output_file_1, output_file_2, output_file_3):
    """
    按照 prot_type 将日志分类并写入不同文件
    
    Args:
        input_file (str): 输入日志文件路径
        output_file_1 (str): prot_type 3 的输出文件路径
        output_file_2 (str): prot_type 5 的输出文件路径
        output_file_3 (str): prot_type 7 的输出文件路径
    """
    # 统计计数器
    counters = {'3': 0, '5': 0, '7': 0, 'other': 0, 'total': 0}
    
    try:
        # 打开三个输出文件
        with open(output_file_1, 'w', encoding='utf-8') as out1, \
             open(output_file_2, 'w', encoding='utf-8') as out2, \
             open(output_file_3, 'w', encoding='utf-8') as out3:
            
            # 逐行读取输入文件
            current_file = None
            current_type = None
            
            print(f"开始处理文件: {input_file}")
            
            with open(input_file, 'r', encoding='utf-8', errors='replace') as f:
                for line in f:
                    counters['total'] += 1
                    
                    # 每处理10万行输出一次进度
                    if counters['total'] % 100000 == 0:
                        print(f"已处理 {counters['total']} 行...")
                    
                    match = re.search(r'prot_type (\d)', line)
                    if match:
                        port_type = match.group(1)
                        
                        if port_type == '3':
                            current_file = out1
                            current_type = '3'
                        elif port_type == '5':
                            current_file = out2
                            current_type = '5'
                        elif port_type == '7':
                            current_file = out3
                            current_type = '7'
                        else:
                            current_file = None
                            current_type = 'other'
                        
                        if current_type != 'other':
                            counters[current_type] += 1
                        else:
                            counters['other'] += 1
                    
                    # 如果有有效的当前文件，写入该行
                    if current_file is not None:
                        current_file.write(line)
        
        # 打印处理结果
        print("\n处理完成!")
        print(f"总行数: {counters['total']}")
        print(f"prot_type 3 (ADAS1): {counters['3']} 条记录，保存到 {output_file_1}")
        print(f"prot_type 7 (ADAS2): {counters['5']} 条记录，保存到 {output_file_2}")
        print(f"prot_type 2 (ADAS3): {counters['7']} 条记录，保存到 {output_file_3}")
        print(f"其他 prot_type: {counters['other']} 条记录")
        
    except Exception as e:
        print(f"处理文件时出错: {e}")

if __name__ == "__main__":
    print("开始分析日志文件...")
    
    # 检查输入文件是否存在
    if not os.path.isfile(logfile):
        print(f"错误: 输入文件不存在 - {logfile}")
        print("请检查文件路径是否正确，或确认日志合并是否成功完成。")
        exit(1)
    
    # 输出文件信息
    print(f"输入文件: {logfile}")
    print(f"输出文件:")
    print(f"  - ADAS1 (prot_type 3): {output_file_1}")
    print(f"  - ADAS2 (prot_type 7): {output_file_2}")
    print(f"  - ADAS3 (prot_type 2): {output_file_3}")
    
    # 执行分类并保存到相应的文件
    retain_lines_by_port_type(logfile, output_file_1, output_file_2, output_file_3)