import numpy as np

def reduce_dimension(input_path, output_path, target_dim):
    """
    降低特征的维度到指定的大小。

    参数：
    - input_path: str，输入的npy文件路径。
    - output_path: str，输出的npy文件路径。
    - target_dim: int，目标维度（行数）。
    """
    try:
        # 加载npy文件
        feature_data = np.load(input_path)
        print(f"原始特征维度: {feature_data.shape}")

        # 检查是否为二维数组
        if feature_data.ndim != 2:
            raise ValueError("特征数据必须是二维数组，每行代表一个特征向量！")

        # 检查目标维度是否合理
        current_dim = feature_data.shape[0]
        if target_dim > current_dim:
            raise ValueError("目标维度不能大于原始维度！")

        # 随机选择目标数量的行
        indices = np.random.choice(current_dim, target_dim, replace=False)
        reduced_feature_data = feature_data[indices, :]
        print(f"降维后特征维度: {reduced_feature_data.shape}")

        # 保存降维后的特征数据
        np.save(output_path, reduced_feature_data)
        print(f"降维后的特征已保存至: {output_path}")

    except FileNotFoundError:
        print(f"文件未找到: {input_path}")
    except ValueError as e:
        print(f"数据处理错误: {e}")
    except Exception as e:
        print(f"发生错误: {e}")

# 示例使用
if __name__ == "__main__":
    input_path = "/home/linux/ShenGang/feature_extraction_new_construct/e2e_fusion_result/maxeye_JMS3_fusion/JMS2-15.88.npy"  # 请根据实际路径修改
    output_path = "/home/linux/ShenGang/feature_extraction_new_construct/e2e_fusion_result/maxeye_JMS3_fusion/JMS2-15.88_5161.npy"  # 保存降维后的文件
    target_dim = 5161  # 设置目标维度

    reduce_dimension(input_path, output_path, target_dim)
