import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# 定义文件路径
npy_file = '/home/linux/ShenGang/feature_extraction_new_construct/e2e_self_supervised_results/fused_feature.npy'  # 替换为你的 .npy 文件路径
# csv_file = '/home/linux/ShenGang/feature_extraction/results/fusion_result/mmtm_features/fused_feature2.csv'  # 输出 CSV 文件路径

def load_and_inspect_npy(file_path):
    """
    加载并检查 .npy 文件内容
    """
    if not os.path.exists(file_path):
        print(f"文件 {file_path} 不存在！")
        return None
    
    # 加载 .npy 文件
    data = np.load(file_path)
    
    # 显示数据信息
    print("========= 数据信息 =========")
    print(f"数据类型：{type(data)}")
    print(f"数据形状：{data.shape}")
    print(f"数据内容（部分显示）：\n{data[:5] if data.ndim > 1 else data[:20]}")
    print("==========================")
    return data

def save_as_csv(data, output_path):
    """
    将数据保存为 CSV 文件
    """
    if isinstance(data, np.ndarray):
        try:
            np.savetxt(output_path, data, delimiter=',', fmt='%s')
            print(f"数据已保存为 CSV 文件：{output_path}")
        except Exception as e:
            print(f"保存 CSV 文件时出错：{e}")
    else:
        print("数据格式不支持保存为 CSV！")

def visualize_data(data):
    """
    可视化二维数组
    """
    if data.ndim == 2:  # 检查是否为二维数组
        plt.imshow(data, cmap='gray', aspect='auto')
        plt.colorbar()
        plt.title("二维数组可视化")
        plt.show()
    else:
        print("数据不是二维数组，无法可视化！")

# 主流程
if __name__ == "__main__":
    # 加载并检查数据
    data = load_and_inspect_npy(npy_file)
    
    if data is not None:
        # # 保存为 CSV 文件
        # save_as_csv(data, csv_file)
        
        # 可视化二维数组
        visualize_data(data)
