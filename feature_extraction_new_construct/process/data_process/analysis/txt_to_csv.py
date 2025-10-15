import csv
import os

def txt_to_csv(txt_file, csv_file, delimiter=' ', encoding='utf-8'):
    try:
        # 检查输出文件所在目录是否存在，如果不存在则创建
        output_dir = os.path.dirname(csv_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 打开txt文件和csv文件
        with open(txt_file, 'r', encoding=encoding) as txt_f, open(csv_file, 'w', newline='', encoding=encoding) as csv_f:
            # 创建csv写入器
            csv_writer = csv.writer(csv_f)

            # 逐行读取txt文件
            for line in txt_f:
                # 如果行不为空
                if line.strip():
                    # 假设数据是用空格分隔的，可以根据需要调整分隔符
                    line_data = line.strip().split(delimiter)
                    # 将每行数据写入csv文件
                    csv_writer.writerow(line_data)

        print(f"文件已成功转换为 {csv_file}")
    except Exception as e:
        print(f"转换过程中发生错误: {e}")

# 使用示例
txt_file = '/home/linux/ShenGang/demo/analysis/timep_ADAS2_with_states_and_obstacles.txt'  # 输入txt文件路径
csv_file = '/home/linux/ShenGang/demo/analysis/timep_ADAS2_with_states_and_obstacles.csv'  # 输出csv文件路径

txt_to_csv(txt_file, csv_file)
