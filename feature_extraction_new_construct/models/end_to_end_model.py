# file: models/end_to_end_model.py

import torch
import torch.nn as nn

# 为了避免循环导入或依赖其他文件，我把所有需要的类都放在了这个文件里

# ===================================================================
# =================== 依赖的子模块定义 ============================
# ===================================================================

class SupervisedFeatureExtractor(nn.Module):
    def __init__(self, selected_indices=None):
        super().__init__()
        self.selected_indices = selected_indices if selected_indices is not None else [6, 7]
        self.selected_dim = len(self.selected_indices)
        
        self.hidden_dim = max(8, self.selected_dim * 2)
        
        extractor_layers = []
        extractor_layers.extend([
            nn.Linear(self.selected_dim, self.hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Dropout(0.2)
        ])
        
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
            extractor_layers.append(nn.Linear(self.hidden_dim, self.selected_dim))
        
        self.feature_extractor = nn.Sequential(*extractor_layers)
    
    def forward(self, x):
        selected_features = x[:, self.selected_indices]
        extracted = self.feature_extractor(selected_features)
        return extracted


class IterativeFusionLayer(nn.Module):
    def __init__(self, dim_1, dim_2, hidden_dim):
        super(IterativeFusionLayer, self).__init__()
        self.transform1 = nn.Sequential(nn.Linear(dim_1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, dim_1))
        self.transform2 = nn.Sequential(nn.Linear(dim_2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, dim_2))
        self.interaction = nn.Sequential(nn.Linear(dim_1 + dim_2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, dim_1 + dim_2))
        self.norm1 = nn.LayerNorm(dim_1)
        self.norm2 = nn.LayerNorm(dim_2)

    def forward(self, x1, x2):
        trans1 = self.transform1(x1)
        trans2 = self.transform2(x2)
        combined = torch.cat([trans1, trans2], dim=1)
        interaction = self.interaction(combined)
        inter1, inter2 = torch.split(interaction, [x1.shape[1], x2.shape[1]], dim=1)
        x1 = self.norm1(x1 + inter1)
        x2 = self.norm2(x2 + inter2)
        return x1, x2


class IterativeFusionNetwork(nn.Module):
    def __init__(self, dim_1, dim_2, config):
        super(IterativeFusionNetwork, self).__init__()
        
        self.feature1_weight = config['fusion']['feature1_weight']
        self.feature2_weight = config['fusion']['feature2_weight']
        n_iterations = config['fusion']['n_iterations']
        hidden_dim = config['fusion']['hidden_dim']
        
        total_weight = self.feature1_weight + self.feature2_weight
        # 避免除以零
        if total_weight > 0:
            self.feature1_weight = self.feature1_weight / total_weight
            self.feature2_weight = self.feature2_weight / total_weight
        
        self.iterations = nn.ModuleList([
            IterativeFusionLayer(dim_1, dim_2, hidden_dim)
            for _ in range(n_iterations)
        ])
        
        self.output_dim = dim_1
        
        self.adjust_dim1 = nn.Sequential(
            nn.Linear(dim_1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.output_dim)
        )
        
        self.adjust_dim2 = nn.Sequential(
            nn.Linear(dim_2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.output_dim)
        )
        
    def forward(self, x1, x2):
        for iteration in self.iterations:
            x1, x2 = iteration(x1, x2)
        
        adjusted_x1 = self.adjust_dim1(x1)
        adjusted_x2 = self.adjust_dim2(x2)
        
        fused_feature = (
            self.feature1_weight * adjusted_x1 + 
            self.feature2_weight * adjusted_x2
        )
        
        return fused_feature

class EndToEndADASModel(nn.Module):
    def __init__(self, config):
        super(EndToEndADASModel, self).__init__()
        self.config = config

        # 1. 初始化两个独立的特征提取器
        adas1_config = config['end_to_end']['adas1']
        adas2_config = config['end_to_end']['adas2']

        self.extractor_adas1 = SupervisedFeatureExtractor(
            selected_indices=adas1_config['selected_indices']
        )
        self.extractor_adas2 = SupervisedFeatureExtractor(
            selected_indices=adas2_config['selected_indices']
        )

        # 2. 初始化特征融合模块
        dim1 = len(adas1_config['selected_indices'])
        dim2 = len(adas2_config['selected_indices'])
        
        self.fusion_module = IterativeFusionNetwork(dim1, dim2, config)

        # # 3. 定义最终的预测头
        # fusion_output_dim = self.fusion_module.output_dim
        # pred_head_config = config['end_to_end']['prediction_head']

        # self.prediction_head = nn.Sequential(
        #     nn.Linear(fusion_output_dim, pred_head_config['hidden_dim']),
        #     nn.ReLU(),
        #     nn.Dropout(pred_head_config['dropout']),
        #     nn.Linear(pred_head_config['hidden_dim'], 1) # 输出为1维的距离
        # )

    def forward(self, input_adas1, input_adas2):
        """定义整个模型的前向传播路径"""
        
        xe1 = self.extractor_adas1(input_adas1)
        xe2 = self.extractor_adas2(input_adas2)
        
        fused_feature = self.fusion_module(xe1, xe2)
        
        # predicted_distance = self.prediction_head(fused_feature)
        
        return fused_feature, xe1, xe2