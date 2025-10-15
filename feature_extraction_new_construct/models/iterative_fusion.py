
# import torch
# import torch.nn as nn

# class IterativeFusionLayer(nn.Module):
#     def __init__(self, dim_1, dim_2, hidden_dim=64, feature1_weight=0.5, feature2_weight=0.5):
#         super(IterativeFusionLayer, self).__init__()
        
#         self.feature1_weight = feature1_weight
#         self.feature2_weight = feature2_weight
        
#         # 特征转换层
#         self.transform1 = nn.Sequential(
#             nn.Linear(dim_1, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, dim_1)
#         )
        
#         self.transform2 = nn.Sequential(
#             nn.Linear(dim_2, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, dim_2)
#         )
        
#         # 交互层
#         self.interaction = nn.Sequential(
#             nn.Linear(dim_1 + dim_2, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, dim_1 + dim_2)
#         )
        
#         self.norm1 = nn.LayerNorm(dim_1)
#         self.norm2 = nn.LayerNorm(dim_2)

#     def forward(self, x1, x2):
#         # 特征转换
#         trans1 = self.transform1(x1)
#         trans2 = self.transform2(x2)
        
#         # 特征交互，加入权重
#         combined = torch.cat([trans1, trans2], dim=1)
#         interaction = self.interaction(combined)
        
#         # 分离交互后的特征
#         inter1, inter2 = torch.split(interaction, [x1.shape[1], x2.shape[1]], dim=1)
        
#         # 残差连接和归一化，考虑权重
#         x1 = self.norm1(x1 + self.feature1_weight * inter1)
#         x2 = self.norm2(x2 + self.feature2_weight * inter2)
        
#         return x1, x2

# # class IterativeFusionNetwork(nn.Module):
# #     def __init__(self, dim_1, dim_2, n_iterations=3, hidden_dim=64, feature1_weight=0.5, feature2_weight=0.5):
# #         super(IterativeFusionNetwork, self).__init__()
        
# #         # 保存权重
# #         self.feature1_weight = feature1_weight
# #         self.feature2_weight = feature2_weight
        
# #         # 创建多个迭代层，传入权重
# #         self.iterations = nn.ModuleList([
# #             IterativeFusionLayer(
# #                 dim_1, dim_2, hidden_dim,
# #                 feature1_weight=feature1_weight,
# #                 feature2_weight=feature2_weight
# #             )
# #             for _ in range(n_iterations)
# #         ])
        
# #         # 最终的特征调整层
# #         self.final_adjust1 = nn.Linear(dim_1, dim_1)
# #         self.final_adjust2 = nn.Linear(dim_2, dim_2)
        
# #     def forward(self, x1, x2):
# #         # 多次迭代
# #         for iteration in self.iterations:
# #             x1, x2 = iteration(x1, x2)
        
# #         # 修改最终调整：不要直接相乘
# #         x1 = self.final_adjust1(x1)
# #         x2 = self.final_adjust2(x2)
        

# #         return (x1 * self.feature1_weight), (x2 * self.feature2_weight)
    
# class IterativeFusionNetwork(nn.Module):
#     def __init__(self, dim_1, dim_2, n_iterations=3, hidden_dim=64, feature1_weight=0.5, feature2_weight=0.5):
#         super(IterativeFusionNetwork, self).__init__()
        
#         self.feature1_weight = feature1_weight
#         self.feature2_weight = feature2_weight
        
#         # 保持原有的迭代层
#         self.iterations = nn.ModuleList([
#             IterativeFusionLayer(dim_1, dim_2, hidden_dim)
#             for _ in range(n_iterations)
#         ])
        
#         # 修改最终层，将两个特征融合为一个
#         self.final_fusion = nn.Sequential(
#             nn.Linear(dim_1 + dim_2, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, dim_1)  # 输出维度可以是dim_1或dim_2
#         )
        
#     def forward(self, x1, x2):
#         # 多次迭代交互
#         for iteration in self.iterations:
#             x1, x2 = iteration(x1, x2)
        
#         # 加权融合为单一特征
#         weighted_x1 = x1 * self.feature1_weight
#         weighted_x2 = x2 * self.feature2_weight
        
#         # 连接后进行最终融合
#         combined = torch.cat([weighted_x1, weighted_x2], dim=1)
#         fused_feature = self.final_fusion(combined)
        
#         return fused_feature

import torch
import torch.nn as nn

class IterativeFusionLayer(nn.Module):
    def __init__(self, dim_1, dim_2, hidden_dim):
        super(IterativeFusionLayer, self).__init__()
        
        # 特征转换层
        self.transform1 = nn.Sequential(
            nn.Linear(dim_1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim_1)
        )
        
        self.transform2 = nn.Sequential(
            nn.Linear(dim_2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim_2)
        )
        
        # 交互层
        self.interaction = nn.Sequential(
            nn.Linear(dim_1 + dim_2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim_1 + dim_2)
        )
        
        self.norm1 = nn.LayerNorm(dim_1)
        self.norm2 = nn.LayerNorm(dim_2)

    def forward(self, x1, x2):
        # 特征转换
        trans1 = self.transform1(x1)
        trans2 = self.transform2(x2)
        
        # 特征交互
        combined = torch.cat([trans1, trans2], dim=1)
        interaction = self.interaction(combined)
        
        # 分离交互后的特征
        inter1, inter2 = torch.split(interaction, [x1.shape[1], x2.shape[1]], dim=1)
        
        # 残差连接和归一化 - 移除权重应用，让权重只在最终融合时应用一次
        x1 = self.norm1(x1 + inter1)
        x2 = self.norm2(x2 + inter2)
        
        return x1, x2

class IterativeFusionNetwork(nn.Module):
    def __init__(self, dim_1, dim_2, config):
        super(IterativeFusionNetwork, self).__init__()
        
        # 直接从config中获取参数
        self.feature1_weight = config['fusion']['feature1_weight']
        self.feature2_weight = config['fusion']['feature2_weight']
        n_iterations = config['fusion']['n_iterations']
        hidden_dim = config['fusion']['hidden_dim']
        
        # 确保权重归一化为总和为1
        total_weight = self.feature1_weight + self.feature2_weight
        self.feature1_weight = self.feature1_weight / total_weight
        self.feature2_weight = self.feature2_weight / total_weight
        
        # 创建多个迭代层，不再传入权重
        self.iterations = nn.ModuleList([
            IterativeFusionLayer(dim_1, dim_2, hidden_dim)
            for _ in range(n_iterations)
        ])
        
        # 修改最终层，适应不同维度的特征
        self.output_dim = dim_1  # 或者可以根据需要选择不同的输出维度
        
        # 添加额外层来处理维度变化和权重融合
        # 将feature1映射到输出维度
        self.adjust_dim1 = nn.Sequential(
            nn.Linear(dim_1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.output_dim)
        )
        
        # 将feature2映射到输出维度
        self.adjust_dim2 = nn.Sequential(
            nn.Linear(dim_2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.output_dim)
        )
        
    def forward(self, x1, x2):
        # 多次迭代交互
        for iteration in self.iterations:
            x1, x2 = iteration(x1, x2)
        
        # 调整特征到相同维度
        adjusted_x1 = self.adjust_dim1(x1)
        adjusted_x2 = self.adjust_dim2(x2)
        
        # 根据权重融合特征 - 这里是权重唯一应用的地方
        fused_feature = (
            self.feature1_weight * adjusted_x1 + 
            self.feature2_weight * adjusted_x2
        )
        
        return fused_feature