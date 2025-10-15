

# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# import os

# class DataLoader:
#     def __init__(self, config):
#         self.config = config
#         self.scaler = StandardScaler()
#         self.gt_error_bound = 3.0  # GT误差边界
#         self.gt_values = np.array([ 0, 15, -10, 25])  # 设定的GT值
    
#     def load_txt(self, file_path):
#         """加载单个TXT文件并过滤数据"""
#         if not os.path.exists(file_path):
#             print(f"错误：文件不存在: {file_path}")
#             print(f"当前工作目录: {os.getcwd()}")
#             return None
            
#         try:
#             df = pd.read_csv(file_path, 
#                            sep=' ',
#                            header=None,
#                            skipinitialspace=True)
            
#             print(f"\n成功加载文件: {file_path}")
#             print(f"原始数据维度: {df.shape}")
            
#             # 验证和处理数据维度
#             if df.shape[1] != self.config['input_dim']:
#                 if df.shape[1] == 21:
#                     df = df.iloc[:, :20]
#                     print(f"已截取前20列，新维度: {df.shape}")
#                 else:
#                     raise ValueError(f"数据维度必须是20或21，当前维度: {df.shape[1]}")
            
#             return df.values
            
#         except Exception as e:
#             print(f"加载文件 {file_path} 时出错: {str(e)}")
#             return None
    
#     def check_gt_error(self, data):
#         """检查数据与GT值之间的误差"""
#         selected_features = data[:, self.config['selected_indices']]
        
#         # 计算每个样本与GT值的绝对误差
#         errors = np.abs(selected_features - self.gt_values)
        
#         # 找出所有维度误差都在范围内的样本
#         valid_samples = np.all(errors <= self.gt_error_bound, axis=1)
        
#         return valid_samples
    
#     def prepare_data(self):
#         """准备数据集：加载、过滤和预处理"""
#         all_data = []
        
#         print(f"正在加载并过滤数据...")
#         for file_path in self.config['file_paths']:
#             print(f"处理文件: {file_path}")
#             data = self.load_txt(file_path)
#             if data is not None:
#                 # 过滤数据
#                 valid_samples = self.check_gt_error(data)
#                 filtered_data = data[valid_samples]
#                 print(f"- 原始样本数: {len(data)}")
#                 print(f"- 过滤后样本数: {len(filtered_data)}")
#                 if len(filtered_data) > 0:
#                     all_data.append(filtered_data)
        
#         if not all_data:
#             raise ValueError("没有符合GT误差要求的数据")
        
#         # 合并所有有效数据
#         X = np.vstack(all_data)
        
#         # 创建GT标签
#         y = np.tile(self.gt_values, (len(X), 1))
        
#         # 只对非选定特征列进行标准化
#         selected_indices = self.config['selected_indices']
#         other_indices = [i for i in range(X.shape[1]) if i not in selected_indices]
        
#         if other_indices:
#             X[:, other_indices] = self.scaler.fit_transform(X[:, other_indices])
        
#         print(f"\n数据处理完成:")
#         print(f"- 总有效样本数: {X.shape[0]}")
#         print(f"- 特征维度: {X.shape[1]}")
#         print(f"- GT值设定为: {self.gt_values}")
#         print(f"- GT误差范围: ±{self.gt_error_bound}")
        
#         return X, y


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

class DataLoader:
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.gt_error_bound = config.get('gt_error_bound', 3.0)  # 从配置中获取误差边界
        self.gt_values = np.array(config['gt_values'])  # 从配置中获取GT值
        
        # 验证GT值的数量是否与选定的特征数量匹配
        if len(self.gt_values) != len(config['selected_indices']):
            raise ValueError(f"GT值的数量({len(self.gt_values)})与选定的特征数量({len(config['selected_indices'])})不匹配")
    
    def load_txt(self, file_path):
        """加载单个TXT文件"""
        if not os.path.exists(file_path):
            print(f"错误：文件不存在: {file_path}")
            print(f"当前工作目录: {os.getcwd()}")
            return None
            
        try:
            df = pd.read_csv(file_path, 
                           sep=' ',
                           header=None,
                           skipinitialspace=True)
            
            print(f"\n成功加载文件: {file_path}")
            print(f"原始数据维度: {df.shape}")
            
            if df.shape[1] != self.config['input_dim']:
                if df.shape[1] == 21:
                    df = df.iloc[:, :20]
                    print(f"已截取前20列，新维度: {df.shape}")
                else:
                    raise ValueError(f"数据维度必须是20或21，当前维度: {df.shape[1]}")
            
            return df.values
            
        except Exception as e:
            print(f"加载文件 {file_path} 时出错: {str(e)}")
            return None
    
    def check_gt_error(self, data):
        """检查数据与GT值之间的误差"""
        selected_features = data[:, self.config['selected_indices']]
        
        # 计算每个样本与对应GT值的绝对误差
        errors = np.abs(selected_features - self.gt_values)
        
        # 找出所有维度误差都在范围内的样本
        valid_samples = np.all(errors <= self.gt_error_bound, axis=1)
        
        # 打印每个特征的误差统计
        for i, feature_name in enumerate(self.config['feature_names']):
            mean_error = errors[:, i].mean()
            max_error = errors[:, i].max()
            print(f"{feature_name} - 平均误差: {mean_error:.2f}, 最大误差: {max_error:.2f}")
        
        return valid_samples
    
    def prepare_data(self):
        """准备数据集：加载、过滤和预处理"""
        all_data = []
        
        print(f"正在加载并过滤数据...")
        print(f"使用的GT值: {self.gt_values}")
        print(f"误差边界: ±{self.gt_error_bound}")
        
        for file_path in self.config['file_paths']:
            print(f"\n处理文件: {file_path}")
            data = self.load_txt(file_path)
            if data is not None:
                # 过滤数据
                valid_samples = self.check_gt_error(data)
                filtered_data = data[valid_samples]
                print(f"- 原始样本数: {len(data)}")
                print(f"- 符合GT要求的样本数: {len(filtered_data)}")
                if len(filtered_data) > 0:
                    all_data.append(filtered_data)
        
        if not all_data:
            raise ValueError("没有符合GT误差要求的数据")
        
        # 合并所有有效数据
        X = np.vstack(all_data)
        
        # 创建GT标签
        y = np.tile(self.gt_values, (len(X), 1))
        
        # 只对非选定特征列进行标准化
        selected_indices = self.config['selected_indices']
        other_indices = [i for i in range(X.shape[1]) if i not in selected_indices]
        
        if other_indices:
            X[:, other_indices] = self.scaler.fit_transform(X[:, other_indices])
        
        print(f"\n数据处理完成:")
        print(f"- 总有效样本数: {X.shape[0]}")
        print(f"- 特征维度: {X.shape[1]}")
        for i, (feature_name, gt_val) in enumerate(zip(self.config['feature_names'], self.gt_values)):
            print(f"- {feature_name} 的 GT 值: {gt_val}")
        
        return X, y