import pandas as pd
import numpy as np

# --- 文件路径定义 ---
# 输入的 .npy 文件路径
npy_file_path = "/home/linux/ShenGang/feature_extraction_new_construct/results/new_log_result/extract/all_adas/new_adas_mit200_feature/train_features.npy"

# 输出的 .csv 文件路径 (建议使用新文件名以避免覆盖原始文件)
csv_output_path = "/home/linux/ShenGang/feature_extraction_new_construct/results/new_log_result/extract/all_adas/new_adas_mit200_feature/train_features.csv"


# --- 转换过程 ---
# 1. 使用 numpy 加载 .npy 文件
print(f"正在从 {npy_file_path} 加载数据...")
data_array = np.load(npy_file_path)
print("数据加载完成。")

# 2. 将 numpy 数组转换为 pandas DataFrame
# 这是将数据保存为 CSV 的标准方式
df_from_npy = pd.DataFrame(data_array)
print("已将Numpy数组转换为Pandas DataFrame。")

# 3. 将 DataFrame 保存为 .csv 文件
# 设置 index=False 是一个好习惯，这样可以避免在 CSV 文件中写入一列额外的行号
df_from_npy.to_csv(csv_output_path, index=False)
print(f"文件已成功保存到: {csv_output_path}")