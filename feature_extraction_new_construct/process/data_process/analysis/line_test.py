from collections import defaultdict

def count_timestamps():
    # 指定文件路径
    file_path = '/home/linux/ShenGang/feature_extraction_new_construct/process/data_process/dataprocess/adas1_aligned.txt'  # 请将此处改为你的实际文件路径
    
    # 使用 defaultdict 来存储每个时间戳的计数
    timestamp_counts = defaultdict(int)
    
    try:
        with open(file_path, 'r') as file:
            for line in file:
                # 分割行并获取第一列（时间戳）
                parts = line.strip().split()
                if parts:  # 确保行不为空
                    timestamp = parts[0]
                    timestamp_counts[timestamp] += 1
    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}")
        return
    except Exception as e:
        print(f"发生错误：{str(e)}")
        return
    
    # 按时间戳排序并打印结果
    for timestamp in sorted(timestamp_counts.keys()):
        print(f"时间戳 {timestamp}: {timestamp_counts[timestamp]} 行")

if __name__ == "__main__":
    count_timestamps()