import csv

# CSV 文件路径
file_path = '/home/linux/ShenGang/feature_extraction/data/adas1.csv'

# 打开 CSV 文件并读取
with open(file_path, newline='', encoding='utf-8') as csvfile:
    csvreader = csv.reader(csvfile)
    
    # 遍历每一行
    for row_index, row in enumerate(csvreader):
        # 计算当前行的维度（即列数）
        row_dimension = len(row)
        
        # 如果维度不是 21，输出该行
        if row_dimension != 21:
            print(f"第{row_index + 1}行的维度是: {row_dimension}, 内容是: {row}")
