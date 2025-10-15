# # # models/supervised_extractor.py
# # import torch
# # import torch.nn as nn

# # class SupervisedFeatureExtractor(nn.Module):
# #     def __init__(self, input_dim=20, feature_dim=2, selected_indices=[0, 6, 7]):
# #         super().__init__()
        
# #         self.selected_indices = selected_indices
# #         # 分离时间戳列和特征列
# #         self.timestamp_index = selected_indices[0]  # 时间戳列
# #         self.feature_indices = selected_indices[1:]  # 特征列
# #         self.selected_dim = len(self.feature_indices)  # 只计算特征列的维度
        
# #         # 特征提取器只处理非时间戳的特征列
# #         self.feature_extractor = nn.Sequential(
# #             nn.Linear(self.selected_dim, 8),
# #             nn.ReLU(),
# #             nn.BatchNorm1d(8),
# #             nn.Dropout(0.2),
# #             nn.Linear(8, feature_dim)
# #         )
    
# #     def forward(self, x):
# #         # 分离时间戳和特征
# #         timestamps = x[:, self.timestamp_index].unsqueeze(1)  # 保持二维
# #         selected_features = x[:, self.feature_indices]
        
# #         # 只对特征部分进行处理
# #         features = self.feature_extractor(selected_features)
        
# #         # 连接时间戳和处理后的特征
# #         output = torch.cat([timestamps, features], dim=1)
# #         return output, features
    
# #     def extract_features(self, x):
# #         with torch.no_grad():
# #             timestamps = x[:, self.timestamp_index].unsqueeze(1)
# #             selected_features = x[:, self.feature_indices]
# #             features = self.feature_extractor(selected_features)
# #             output = torch.cat([timestamps, features], dim=1)
# #             return output

# # models/supervised_extractor.py
# import torch
# import torch.nn as nn

# class SupervisedFeatureExtractor(nn.Module):
#     def __init__(self, input_dim=20, feature_dim=2, selected_indices=[6, 7]):
#         super().__init__()
        
#         self.selected_indices = selected_indices
#         self.selected_dim = len(self.selected_indices)  # 只计算选定特征列的维度
        
#         # 特征提取器
#         self.feature_extractor = nn.Sequential(
#             nn.Linear(self.selected_dim, 8),
#             nn.ReLU(),
#             nn.BatchNorm1d(8),
#             nn.Dropout(0.2),
#             nn.Linear(8, feature_dim)
#         )
    
#     def forward(self, x):
#         # 只选择指定的特征列
#         selected_features = x[:, self.selected_indices]
#         # 特征提取
#         features = self.feature_extractor(selected_features)
#         return features, features  # 返回相同的特征用于训练和输出
    
#     def extract_features(self, x):
#         with torch.no_grad():
#             selected_features = x[:, self.selected_indices]
#             features = self.feature_extractor(selected_features)
#             return features


# import torch
# import torch.nn as nn

# class SupervisedFeatureExtractor(nn.Module):
#     def __init__(self, input_dim=20, feature_dim=2, selected_indices=[6, 7]):
#         super().__init__()
        
#         self.selected_indices = selected_indices
#         self.selected_dim = len(self.selected_indices)
        
#         # 更复杂的特征提取器架构
#         self.feature_extractor = nn.Sequential(
#             nn.Linear(self.selected_dim, 16),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.BatchNorm1d(16),
            
#             nn.Linear(16, 12),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.BatchNorm1d(12),
            
#             nn.Linear(12, 8),
#             nn.ReLU(),
#             nn.BatchNorm1d(8),
            
#             nn.Linear(8, feature_dim)
#         )
        
#         # 初始化权重
#         self._initialize_weights()
    
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm1d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
    
#     def forward(self, x):
#         selected_features = x[:, self.selected_indices]
#         features = self.feature_extractor(selected_features)
#         return features, features
    
#     def extract_features(self, x):
#         with torch.no_grad():
#             selected_features = x[:, self.selected_indices]
#             features = self.feature_extractor(selected_features)
#             return features


# import torch
# import torch.nn as nn

# class SupervisedFeatureExtractor(nn.Module):
#     def __init__(self, input_dim=20, feature_dim=2, selected_indices=[6, 7]):
#         super().__init__()
        
#         self.selected_indices = selected_indices
#         self.selected_dim = len(self.selected_indices)
        
#         # 简化的特征提取器
#         self.feature_extractor = nn.Sequential(
#             nn.Linear(self.selected_dim, self.selected_dim),
#             nn.ReLU(),
#             nn.Linear(self.selected_dim, self.selected_dim)
#         )
        
#         # 用于GT预测的网络
#         self.predictor = nn.Sequential(
#             nn.Linear(self.selected_dim, 8),
#             nn.ReLU(),
#             nn.Linear(8, 2)  # 输出维度与GT维度相同
#         )
    
#     def forward(self, x):
#         # 提取选定的特征
#         selected_features = x[:, self.selected_indices]
#         # 特征提取
#         features = self.feature_extractor(selected_features)
#         # 预测GT
#         predictions = self.predictor(features)
        
#         return features, predictions
    
#     def extract_features(self, x):
#         with torch.no_grad():
#             selected_features = x[:, self.selected_indices]
#             features = self.feature_extractor(selected_features)
#             return features


# import torch
# import torch.nn as nn

# class SupervisedFeatureExtractor(nn.Module):
#     def __init__(self, input_dim=20, feature_dim=4, selected_indices=[6, 7,9,10]):
#         super().__init__()
        
#         self.selected_indices = selected_indices
#         self.selected_dim = len(self.selected_indices)
#         self.error_bound = 3.0  # 设置误差边界为3
        
#         # # 特征提取器
#         # self.feature_extractor = nn.Sequential(
#         #     nn.Linear(self.selected_dim, self.selected_dim),
#         #     nn.ReLU(),
#         #     nn.Linear(self.selected_dim, self.selected_dim)
#         # )
      
#         # # 预测器
#         # self.predictor = nn.Sequential(
#         #     nn.Linear(self.selected_dim, 8),
#         #     nn.ReLU(),
#         #     nn.Linear(8, 2)
#         # )
#         # 修改特征提取器以适应4个特征
#         self.feature_extractor = nn.Sequential(
#             nn.Linear(self.selected_dim, 8),  # 输入维度变为4
#             nn.ReLU(),
#             nn.BatchNorm1d(8),
#             nn.Dropout(0.2),
#             nn.Linear(8, self.selected_dim)  # 输出维度也为4，保持特征维度
#         )
        
#         # 修改预测器以适应4个特征
#         self.predictor = nn.Sequential(
#             nn.Linear(self.selected_dim, 12),  # 增加中间层维度以处理更多特征
#             nn.ReLU(),
#             nn.BatchNorm1d(12),
#             nn.Linear(12, 8),
#             nn.ReLU(),
#             nn.Linear(8, 4)  # 输出维度为4，对应四个特征的预测
#         )
    
#     def forward(self, x):
#         selected_features = x[:, self.selected_indices]
#         extracted = self.feature_extractor(selected_features)
#         predictions = self.predictor(extracted)
        
#         # 计算特征误差
#         feature_error = torch.abs(extracted - selected_features)
#         # 创建误差掩码，标记超出范围的值
#         error_mask = feature_error > self.error_bound
        
#         return extracted, predictions, feature_error, error_mask
    
#     def extract_features(self, x):
#         with torch.no_grad():
#             selected_features = x[:, self.selected_indices]
#             extracted = self.feature_extractor(selected_features)
#             feature_error = torch.abs(extracted - selected_features)
#             # 只返回误差在范围内的特征
#             valid_mask = feature_error <= self.error_bound
#             extracted[~valid_mask] = selected_features[~valid_mask]  # 对于超出范围的值，使用原始值
#             return extracted



import torch
import torch.nn as nn

class SupervisedFeatureExtractor(nn.Module):
    def __init__(self, input_dim=20, feature_dim=None, selected_indices=None):
        super().__init__()
        
        # 动态获取选定的特征数量
        self.selected_indices = selected_indices if selected_indices is not None else [6, 7]
        self.selected_dim = len(self.selected_indices)
        self.error_bound = 3.0
        
        # 根据特征数量动态设置网络层的维度
        self.hidden_dim = max(8, self.selected_dim * 2)  # 确保隐藏层维度足够大
        
        # 特征提取器 - 使用多层结构，层数和维度根据特征数量动态调整
        extractor_layers = []
        
        # 第一层：从输入维度到隐藏维度
        extractor_layers.extend([
            nn.Linear(self.selected_dim, self.hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Dropout(0.2)
        ])
        
        # 中间层：如果特征数量较多，添加额外的处理层
        if self.selected_dim > 4:
            mid_dim = (self.hidden_dim + self.selected_dim) // 2
            extractor_layers.extend([
                nn.Linear(self.hidden_dim, mid_dim),
                nn.ReLU(),
                nn.BatchNorm1d(mid_dim),
                nn.Dropout(0.1),
                nn.Linear(mid_dim, self.selected_dim)
            ])
        else:
            # 特征数量较少时使用简单结构
            extractor_layers.append(nn.Linear(self.hidden_dim, self.selected_dim))
        
        self.feature_extractor = nn.Sequential(*extractor_layers)
        
        # 预测器 - 同样根据特征数量动态调整
        predictor_layers = []
        pred_hidden_dim = self.selected_dim * 3  # 预测器隐藏层维度
        
        predictor_layers.extend([
            nn.Linear(self.selected_dim, pred_hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(pred_hidden_dim)
        ])
        
        # 对于更多特征，添加更深的预测层
        if self.selected_dim > 4:
            predictor_layers.extend([
                nn.Linear(pred_hidden_dim, pred_hidden_dim // 2),
                nn.ReLU(),
                nn.BatchNorm1d(pred_hidden_dim // 2),
                nn.Linear(pred_hidden_dim // 2, self.selected_dim)
            ])
        else:
            predictor_layers.append(nn.Linear(pred_hidden_dim, self.selected_dim))
            
        self.predictor = nn.Sequential(*predictor_layers)
    
    def forward(self, x):
        # 选择指定的特征列
        selected_features = x[:, self.selected_indices]
        
        # 特征提取
        extracted = self.feature_extractor(selected_features)
        
        # 预测
        predictions = self.predictor(extracted)
        
        # 计算特征误差
        feature_error = torch.abs(extracted - selected_features)
        
        # 创建误差掩码
        error_mask = feature_error > self.error_bound
        
        return extracted, predictions, feature_error, error_mask
    
    def extract_features(self, x):
        with torch.no_grad():
            selected_features = x[:, self.selected_indices]
            extracted = self.feature_extractor(selected_features)
            feature_error = torch.abs(extracted - selected_features)
            valid_mask = feature_error <= self.error_bound
            extracted[~valid_mask] = selected_features[~valid_mask]
            return extracted