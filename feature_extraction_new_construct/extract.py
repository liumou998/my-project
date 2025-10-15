# import torch
# import numpy as np
# import pandas as pd

# def extract_features(model, data, device=None):
#     """
#     使用训练好的模型提取特征
#     Args:
#         model: 训练好的模型
#         data: 输入数据
#         device: 计算设备（GPU/CPU）
#     Returns:
#         pd.DataFrame: 包含时间戳和特征的DataFrame
#     """
#     if device is None:
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     model.eval()
#     with torch.no_grad():
#         features = model.extract_features(
#             torch.FloatTensor(data).to(device)
#         ).cpu().numpy()
    
#     # 创建DataFrame并添加列名
#     df = pd.DataFrame(features, columns=['timestamp'] + [f'feature{i+1}' for i in range(features.shape[1]-1)])
#     return df

import torch
import pandas as pd

def extract_features(model, data, device=None):
    """
    使用训练好的模型提取特征
    Args:
        model: 训练好的模型
        data: 输入数据
        device: 计算设备（GPU/CPU）
    Returns:
        pd.DataFrame: 包含特征的DataFrame
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    with torch.no_grad():
        features = model.extract_features(
            torch.FloatTensor(data).to(device)
        ).cpu().numpy()
    
    # 创建DataFrame并添加列名
    df = pd.DataFrame(features, columns=[f'feature{i+1}' for i in range(features.shape[1])])
    return df