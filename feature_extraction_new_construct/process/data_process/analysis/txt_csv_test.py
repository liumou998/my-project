import csv
import sys

def txt_to_csv(txt_file, csv_file, expected_dims=20, delimiter='\t'):
    try:
        with open(txt_file, 'r', encoding='utf-8') as txt_input:
            lines = txt_input.readlines()
            data = []
            has_dimension_error = False
            
            # 检查每行的维度
            for line_num, line in enumerate(lines, 1):
                # 去除空白字符并分割
                row = line.strip().split(delimiter)
                
                # 严格检查维度
                if len(row) != expected_dims:
                    print(f"错误：第 {line_num} 行维度异常")
                    print(f"预期维度：{expected_dims}")
                    print(f"实际维度：{len(row)}")
                    print(f"问题行内容：{line.strip()}")
                    has_dimension_error = True
                    continue
                    
                data.append(row)
            
            if has_dimension_error:
                print("\n发现维度错误，转换已终止。请检查原始数据确保所有行都是20维。")
                sys.exit(1)
                
            # 只有所有行都符合要求才写入CSV文件
            with open(csv_file, 'w', encoding='utf-8', newline='') as csv_output:
                writer = csv.writer(csv_output)
                writer.writerows(data)
                
            print(f"转换完成！文件已保存为: {csv_file}")
            
    except Exception as e:
        print(f"转换过程中出现错误: {str(e)}")
        sys.exit(1)

def analyze_file_dimensions(txt_file, delimiter='\t'):
    """分析文件中每行的维度情况"""
    dimensions = {}
    try:
        with open(txt_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                dims = len(line.strip().split(delimiter))
                dimensions[dims] = dimensions.get(dims, []) + [line_num]
                
        print("\n维度分析结果：")
        for dims, lines in sorted(dimensions.items()):
            print(f"{dims}维的行数：{len(lines)}")
            print(f"示例行号：{lines[:5]}{'...' if len(lines) > 5 else ''}")
            
    except Exception as e:
        print(f"分析文件时出现错误: {str(e)}")

# 使用示例
if __name__ == "__main__":
    input_file = "/home/linux/ShenGang/demo/analysis/timep_ADAS2_with_states_and_obstacles.txt"
    output_file = "/home/linux/ShenGang/demo/analysis/timep_ADAS2_with_states_and_obstacles.csv"
    
    # 首先分析文件维度情况
    print("开始分析文件维度...")
    analyze_file_dimensions(input_file)
    
    # 确认是否继续转换
    response = input("\n是否继续转换？(y/n): ")
    if response.lower() == 'y':
        print("\n开始转换文件...")
        txt_to_csv(input_file, output_file, expected_dims=20)
    else:
        print("转换已取消")