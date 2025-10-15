import pandas as pd
import numpy as np

# 读取 CSV 文件
df = pd.read_csv("/home/linux/ShenGang/feature_extraction_new_construct/e2e_fusion_result/maxeye_JMS3_fusion/JMS2-15.78.csv")

# 转换为 numpy 数组
data_array = df.values

# 保存为 .npy 文件
np.save("/home/linux/ShenGang/feature_extraction_new_construct/e2e_fusion_result/maxeye_JMS3_fusion/JMS2-15.78.npy", data_array)
