
# import torch
# import torch.nn as nn
# import numpy as np
# from torch.utils.data import DataLoader, TensorDataset
# from torch.optim import Adam
# from models.iterative_fusion import IterativeFusionNetwork
# from config import CONFIG
# from utils.output_manager import OutputManager

# class FusionTrainer:
#     def __init__(self, config, output_manager):
#         self.config = config
#         self.output_manager = output_manager
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
#         # 初始化权重参数
#         self.feature1_weight = config['fusion'].get('feature1_weight', 0.5)
#         self.feature2_weight = config['fusion'].get('feature2_weight', 0.5)
        
#         # 初始化模型
#         self.model = None
#         self.criterion = nn.MSELoss()
#         self.learning_rate = config['fusion']['learning_rate']
#         self.batch_size = config['fusion']['batch_size']
#         self.epochs = config['fusion']['epochs']
        
#         # 保存数据统计信息用于还原
#         self.feature1_stats = None
#         self.feature2_stats = None
    
#     def prepare_data(self, feature1, feature2):
#         """准备数据集，包括标准化"""
#         # 计算并保存统计信息
#         feature1_mean = feature1.mean(axis=0)
#         feature1_std = feature1.std(axis=0)
#         feature2_mean = feature2.mean(axis=0)
#         feature2_std = feature2.std(axis=0)
        
#         self.feature1_stats = (feature1_mean, feature1_std)
#         self.feature2_stats = (feature2_mean, feature2_std)
        
#         # 标准化特征
#         feature1_norm = (feature1 - feature1_mean) / (feature1_std + 1e-8)
#         feature2_norm = (feature2 - feature2_mean) / (feature2_std + 1e-8)
        
#         # 创建数据集
#         dataset = TensorDataset(
#             torch.FloatTensor(feature1_norm),
#             torch.FloatTensor(feature2_norm)
#         )
        
#         return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
    



#     def correlation_loss(self, x1, x2):
#         """计算特征间的相关性损失"""
#         x1_mean = x1.mean(dim=1, keepdim=True)
#         x2_mean = x2.mean(dim=1, keepdim=True)
        
#         x1_norm = x1 - x1_mean
#         x2_norm = x2 - x2_mean
        
#         corr = (x1_norm * x2_norm).sum(dim=1) / (
#             torch.sqrt((x1_norm ** 2).sum(dim=1)) * 
#             torch.sqrt((x2_norm ** 2).sum(dim=1)) + 1e-8
#         )
        
#         return (1 - corr.abs()).mean()

#     def train(self, feature1, feature2):
#         """训练融合模型"""
#         self.model = IterativeFusionNetwork(
#             dim_1=feature1.shape[1],
#             dim_2=feature2.shape[1],
#             n_iterations=self.config['fusion']['n_iterations'],
#             hidden_dim=self.config['fusion']['hidden_dim'],
#             feature1_weight=self.feature1_weight,
#             feature2_weight=self.feature2_weight
#         ).to(self.device)
        
#         optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
#         train_loader = self.prepare_data(feature1, feature2)
        
#         self.output_manager.log_info("开始特征融合模型训练...")
#         self.output_manager.log_info(f"特征1权重: {self.feature1_weight}, 特征2权重: {self.feature2_weight}")
#         best_loss = float('inf')
        
#         for epoch in range(self.epochs):
#             self.model.train()
#             total_loss = 0
#             total_std = 0  # 添加标准差监控
            
#             for batch_x1, batch_x2 in train_loader:
#                 batch_x1 = batch_x1.to(self.device)
#                 batch_x2 = batch_x2.to(self.device)
                
#                 # 前向传播得到融合特征
#                 fused_feature = self.model(batch_x1, batch_x2)
                
#                 # 计算加权的目标特征
#                 weighted_reconstruction = (
#                     self.feature1_weight * batch_x1 + 
#                     self.feature2_weight * batch_x2
#                 )
                
#                 # 计算损失
#                 loss = self.criterion(fused_feature, weighted_reconstruction)
                
#                 # 监控融合特征的标准差
#                 batch_std = fused_feature.std().item()
#                 total_std += batch_std
                
#                 # 反向传播
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
                
#                 total_loss += loss.item()
            
#             avg_loss = total_loss / len(train_loader)
#             avg_std = total_std / len(train_loader)
            
#             self.output_manager.log_info(
#                 f"Epoch [{epoch+1}/{self.epochs}] "
#                 f"Loss: {avg_loss:.6f} "
#                 f"Std: {avg_std:.6f}"
#             )
            
#             if avg_loss < best_loss:
#                 best_loss = avg_loss
#                 torch.save(self.model.state_dict(), self.output_manager.get_fusion_model_path())
        
#         self.output_manager.log_info("特征融合模型训练完成！")

#     def fusion_features(self):
#         """执行特征融合"""
#         self.output_manager.log_info("开始特征融合")
        
#         try:
#             # 获取特征文件路径
#             feature1_path = self.config['fusion']['feature_paths']['feature1']
#             feature2_path = self.config['fusion']['feature_paths']['feature2']
            
#             # 加载特征
#             self.output_manager.log_info(f"加载特征1: {feature1_path}")
#             feature1 = np.load(feature1_path)
#             self.output_manager.log_info(f"加载特征2: {feature2_path}")
#             feature2 = np.load(feature2_path)
            
#             self.output_manager.log_info(f"特征1形状: {feature1.shape}")
#             self.output_manager.log_info(f"特征2形状: {feature2.shape}")
            
#             # 训练模型
#             self.train(feature1, feature2)
            
#             # 加载最佳模型
#             self.model.load_state_dict(torch.load(self.output_manager.get_fusion_model_path()))
#             self.model.eval()
            
#             # 特征融合
#             with torch.no_grad():
#                 feature1_norm = (feature1 - self.feature1_stats[0]) / (self.feature1_stats[1] + 1e-8)
#                 feature2_norm = (feature2 - self.feature2_stats[0]) / (self.feature2_stats[1] + 1e-8)
                
#                 feature1_tensor = torch.FloatTensor(feature1_norm).to(self.device)
#                 feature2_tensor = torch.FloatTensor(feature2_norm).to(self.device)
                
#                 fused_feature = self.model(feature1_tensor, feature2_tensor)
            
#             # 还原到原始尺度（使用feature1的统计信息）
#             fused_feature_real = fused_feature.cpu().numpy() * self.feature1_stats[1] + self.feature1_stats[0]
            
#             # 监控最终融合特征的统计信息
#             self.output_manager.log_info(f"融合特征形状: {fused_feature_real.shape}")
#             self.output_manager.log_info(f"融合特征均值: {fused_feature_real.mean():.6f}")
#             self.output_manager.log_info(f"融合特征标准差: {fused_feature_real.std():.6f}")
            
#             # 保存融合后的特征
#             fused_path = self.output_manager.get_fusion_feature_path('fused_feature_real.npy')
#             np.save(fused_path, fused_feature_real)
            
#             self.output_manager.log_info("特征融合完成！")
#             self.output_manager.log_info(f"融合后特征保存至: {fused_path}")
            
#         except Exception as e:
#             self.output_manager.log_error(f"特征融合过程出错: {str(e)}")
#             raise

# def fusion_features():
#     output_manager = OutputManager()
#     trainer = FusionTrainer(CONFIG, output_manager)
#     trainer.fusion_features()

# if __name__ == "__main__":
#     fusion_features()


# import torch
# import torch.nn as nn
# import numpy as np
# from torch.utils.data import DataLoader, TensorDataset
# from torch.optim import Adam
# from models.iterative_fusion import IterativeFusionNetwork
# from config import CONFIG
# from utils.output_manager import OutputManager

# class FusionTrainer:
#     def __init__(self, config, output_manager):
#         self.config = config
#         self.output_manager = output_manager
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
#         # 初始化权重参数
#         self.feature1_weight = config['fusion'].get('feature1_weight', 0.)
#         self.feature2_weight = config['fusion'].get('feature2_weight', 0.5)
        
#         # 初始化模型
#         self.model = None
#         self.criterion = nn.MSELoss()
#         self.learning_rate = config['fusion']['learning_rate']
#         self.batch_size = config['fusion']['batch_size']
#         self.epochs = config['fusion']['epochs']
        
#         # 保存数据统计信息用于还原
#         self.feature1_stats = None
#         self.feature2_stats = None
    
#     def prepare_data(self, feature1, feature2):
#         """准备数据集，包括标准化"""
#         # 计算并保存统计信息
#         feature1_mean = feature1.mean(axis=0)
#         feature1_std = feature1.std(axis=0)
#         feature2_mean = feature2.mean(axis=0)
#         feature2_std = feature2.std(axis=0)
        
#         self.feature1_stats = (feature1_mean, feature1_std)
#         self.feature2_stats = (feature2_mean, feature2_std)
        
#         # 标准化特征
#         feature1_norm = (feature1 - feature1_mean) / (feature1_std + 1e-8)
#         feature2_norm = (feature2 - feature2_mean) / (feature2_std + 1e-8)
        
#         # 创建数据集
#         dataset = TensorDataset(
#             torch.FloatTensor(feature1_norm),
#             torch.FloatTensor(feature2_norm)
#         )
        
#         return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
    
#     def correlation_loss(self, x1, x2):
#         """计算特征间的相关性损失"""
#         x1_mean = x1.mean(dim=1, keepdim=True)
#         x2_mean = x2.mean(dim=1, keepdim=True)
        
#         x1_norm = x1 - x1_mean
#         x2_norm = x2 - x2_mean
        
#         corr = (x1_norm * x2_norm).sum(dim=1) / (
#             torch.sqrt((x1_norm ** 2).sum(dim=1)) * 
#             torch.sqrt((x2_norm ** 2).sum(dim=1)) + 1e-8
#         )
        
#         return (1 - corr.abs()).mean()

#     def train(self, feature1, feature2):
#         """训练融合模型"""
#         self.model = IterativeFusionNetwork(
#             dim_1=feature1.shape[1],
#             dim_2=feature2.shape[1],
#             n_iterations=self.config['fusion']['n_iterations'],
#             hidden_dim=self.config['fusion']['hidden_dim'],
#             feature1_weight=self.feature1_weight,
#             feature2_weight=self.feature2_weight
#         ).to(self.device)
        
#         optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
#         train_loader = self.prepare_data(feature1, feature2)
        
#         self.output_manager.log_info("开始特征融合模型训练...")
#         self.output_manager.log_info(f"特征1权重: {self.feature1_weight}, 特征2权重: {self.feature2_weight}")
#         best_loss = float('inf')
        
#         for epoch in range(self.epochs):
#             self.model.train()
#             total_loss = 0
#             total_std = 0  # 添加标准差监控
            
#             for batch_x1, batch_x2 in train_loader:
#                 batch_x1 = batch_x1.to(self.device)
#                 batch_x2 = batch_x2.to(self.device)
                
#                 # 前向传播得到融合特征
#                 fused_feature = self.model(batch_x1, batch_x2)
                
#                 # 计算加权的目标特征 - 这样可以明确体现权重的影响
#                 weighted_reconstruction = (
#                     self.feature1_weight * batch_x1[:, :fused_feature.shape[1]] + 
#                     self.feature2_weight * batch_x2[:, :fused_feature.shape[1]]
#                 )
                
#                 # 计算损失
#                 loss = self.criterion(fused_feature, weighted_reconstruction)
                
#                 # 监控融合特征的标准差
#                 batch_std = fused_feature.std().item()
#                 total_std += batch_std
                
#                 # 反向传播
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
                
#                 total_loss += loss.item()
            
#             avg_loss = total_loss / len(train_loader)
#             avg_std = total_std / len(train_loader)
            
#             self.output_manager.log_info(
#                 f"Epoch [{epoch+1}/{self.epochs}] "
#                 f"Loss: {avg_loss:.6f} "
#                 f"Std: {avg_std:.6f}"
#             )
            
#             if avg_loss < best_loss:
#                 best_loss = avg_loss
#                 torch.save(self.model.state_dict(), self.output_manager.get_fusion_model_path())
        
#         self.output_manager.log_info("特征融合模型训练完成！")

#     def fusion_features(self):
#         """执行特征融合"""
#         self.output_manager.log_info("开始特征融合")
#         self.output_manager.log_info(f"使用特征权重 - 特征1: {self.feature1_weight}, 特征2: {self.feature2_weight}")
        
#         try:
#             # 获取特征文件路径
#             feature1_path = self.config['fusion']['feature_paths']['feature1']
#             feature2_path = self.config['fusion']['feature_paths']['feature2']
            
#             # 加载特征
#             self.output_manager.log_info(f"加载特征1: {feature1_path}")
#             feature1 = np.load(feature1_path)
#             self.output_manager.log_info(f"加载特征2: {feature2_path}")
#             feature2 = np.load(feature2_path)
            
#             self.output_manager.log_info(f"特征1形状: {feature1.shape}")
#             self.output_manager.log_info(f"特征2形状: {feature2.shape}")
            
#             # 训练模型
#             self.train(feature1, feature2)
            
#             # 加载最佳模型
#             self.model.load_state_dict(torch.load(self.output_manager.get_fusion_model_path()))
#             self.model.eval()
            
#             # 特征融合
#             with torch.no_grad():
#                 feature1_norm = (feature1 - self.feature1_stats[0]) / (self.feature1_stats[1] + 1e-8)
#                 feature2_norm = (feature2 - self.feature2_stats[0]) / (self.feature2_stats[1] + 1e-8)
                
#                 feature1_tensor = torch.FloatTensor(feature1_norm).to(self.device)
#                 feature2_tensor = torch.FloatTensor(feature2_norm).to(self.device)
                
#                 # 添加这些打印语句来验证权重的影响
#                 self.output_manager.log_info(f"检查模型权重 - 特征1: {self.model.feature1_weight}, 特征2: {self.model.feature2_weight}")
                
#                 # 融合特征
#                 fused_feature = self.model(feature1_tensor, feature2_tensor)
                
#                 # 检查各特征的贡献
#                 with_only_feature1 = self.model(feature1_tensor, torch.zeros_like(feature2_tensor))
#                 with_only_feature2 = self.model(torch.zeros_like(feature1_tensor), feature2_tensor)
                
#                 # 计算相似度以评估贡献
#                 feature1_contribution = torch.nn.functional.cosine_similarity(
#                     fused_feature.flatten().unsqueeze(0), 
#                     with_only_feature1.flatten().unsqueeze(0)
#                 ).item()
                
#                 feature2_contribution = torch.nn.functional.cosine_similarity(
#                     fused_feature.flatten().unsqueeze(0), 
#                     with_only_feature2.flatten().unsqueeze(0)
#                 ).item()
                
#                 # 记录贡献信息
#                 self.output_manager.log_info(f"特征1对融合特征的贡献: {feature1_contribution:.4f}")
#                 self.output_manager.log_info(f"特征2对融合特征的贡献: {feature2_contribution:.4f}")
                
#                 # 计算相对贡献比例
#                 total_contribution = abs(feature1_contribution) + abs(feature2_contribution)
#                 if total_contribution > 0:
#                     rel_feature1 = abs(feature1_contribution) / total_contribution
#                     rel_feature2 = abs(feature2_contribution) / total_contribution
#                     self.output_manager.log_info(f"特征1相对贡献: {rel_feature1:.4f} (目标: {self.feature1_weight:.4f})")
#                     self.output_manager.log_info(f"特征2相对贡献: {rel_feature2:.4f} (目标: {self.feature2_weight:.4f})")
            
#             # 还原到原始尺度（使用feature1的统计信息）
#             fused_feature_real = fused_feature.cpu().numpy() * self.feature1_stats[1] + self.feature1_stats[0]
            
#             # 监控最终融合特征的统计信息
#             self.output_manager.log_info(f"融合特征形状: {fused_feature_real.shape}")
#             self.output_manager.log_info(f"融合特征均值: {fused_feature_real.mean():.6f}")
#             self.output_manager.log_info(f"融合特征标准差: {fused_feature_real.std():.6f}")
            
#             # 保存融合后的特征
#             fused_path = self.output_manager.get_fusion_feature_path('fused_feature_real.npy')
#             np.save(fused_path, fused_feature_real)
            
#             self.output_manager.log_info("特征融合完成！")
#             self.output_manager.log_info(f"融合后特征保存至: {fused_path}")
            
#         except Exception as e:
#             self.output_manager.log_error(f"特征融合过程出错: {str(e)}")
#             raise

# def fusion_features():
#     output_manager = OutputManager()
#     trainer = FusionTrainer(CONFIG, output_manager)
#     trainer.fusion_features()

# if __name__ == "__main__":
#     fusion_features()

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from models.iterative_fusion import IterativeFusionNetwork
from config import CONFIG
from utils.output_manager import OutputManager

class FusionTrainer:
    def __init__(self, config, output_manager):
        self.config = config
        self.output_manager = output_manager
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 从config中直接获取权重参数
        self.feature1_weight = config['fusion']['feature1_weight']
        self.feature2_weight = config['fusion']['feature2_weight']
        
        # 确保权重归一化
        total_weight = self.feature1_weight + self.feature2_weight
        self.feature1_weight = self.feature1_weight / total_weight
        self.feature2_weight = self.feature2_weight / total_weight
        
        # 记录原始权重用于日志
        self.original_feature1_weight = config['fusion']['feature1_weight']
        self.original_feature2_weight = config['fusion']['feature2_weight']
        
        # 从config中获取其他参数
        self.criterion = nn.MSELoss()
        self.learning_rate = config['fusion']['learning_rate']
        self.batch_size = config['fusion']['batch_size']
        self.epochs = config['fusion']['epochs']
        
        # 保存两个特征的统计信息
        self.feature1_stats = None
        self.feature2_stats = None
        # 保存融合特征的统计信息
        self.fusion_stats = None
    
    def prepare_data(self, feature1, feature2):
        """准备数据集，包括标准化"""
        # 计算并保存统计信息
        feature1_mean = feature1.mean(axis=0)
        feature1_std = feature1.std(axis=0)
        feature2_mean = feature2.mean(axis=0)
        feature2_std = feature2.std(axis=0)
        
        self.feature1_stats = (feature1_mean, feature1_std)
        self.feature2_stats = (feature2_mean, feature2_std)
        
        # 标准化特征
        feature1_norm = (feature1 - feature1_mean) / (feature1_std + 1e-8)
        feature2_norm = (feature2 - feature2_mean) / (feature2_std + 1e-8)
        
        # 创建数据集
        dataset = TensorDataset(
            torch.FloatTensor(feature1_norm),
            torch.FloatTensor(feature2_norm)
        )
        
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def get_weighted_target(self, x1, x2):
        """为训练创建加权目标"""
        # 权重归一化确保总和为1
        total_weight = self.feature1_weight + self.feature2_weight
        norm_w1 = self.feature1_weight / total_weight
        norm_w2 = self.feature2_weight / total_weight
        
        # 这里我们直接将两个特征按权重线性组合
        # 注意这里假设x1和x2已经映射到相同的维度
        return norm_w1 * x1 + norm_w2 * x2

    def train(self, feature1, feature2):
        """训练融合模型"""
        # 直接传入config，不再单独传递参数
        self.model = IterativeFusionNetwork(
            dim_1=feature1.shape[1],
            dim_2=feature2.shape[1],
            config=self.config
        ).to(self.device)
        
        optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        train_loader = self.prepare_data(feature1, feature2)
        
        self.output_manager.log_info("开始特征融合模型训练...")
        self.output_manager.log_info(f"原始特征1权重: {self.original_feature1_weight}, 特征2权重: {self.original_feature2_weight}")
        self.output_manager.log_info(f"归一化后特征1权重: {self.feature1_weight:.4f}, 特征2权重: {self.feature2_weight:.4f}")
        
        best_loss = float('inf')
        
        # 收集用于计算融合特征统计信息的数组
        all_fused_features = []
        
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            epoch_fused_features = []
            
            for batch_x1, batch_x2 in train_loader:
                batch_x1 = batch_x1.to(self.device)
                batch_x2 = batch_x2.to(self.device)
                
                # 前向传播得到融合特征
                fused_feature = self.model(batch_x1, batch_x2)
                
                # 计算损失 - 这里使用自定义的损失函数
                # 在IterativeFusionNetwork中，特征已经按权重融合，
                # 所以这里我们可以直接和加权目标比较
                weighted_target = self.get_weighted_target(
                    self.model.adjust_dim1(batch_x1), 
                    self.model.adjust_dim2(batch_x2)
                )
                
                loss = self.criterion(fused_feature, weighted_target)
                
                # 收集融合特征用于统计信息
                if epoch == self.epochs - 1:  # 只在最后一个epoch收集
                    epoch_fused_features.append(fused_feature.detach().cpu().numpy())
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            
            self.output_manager.log_info(
                f"Epoch [{epoch+1}/{self.epochs}] "
                f"Loss: {avg_loss:.6f}"
            )
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.model.state_dict(), self.output_manager.get_fusion_model_path())
                
                # 在最佳模型保存时，也收集融合特征
                if epoch_fused_features:
                    all_fused_features = np.vstack(epoch_fused_features)
        
        # 计算融合特征的统计信息
        if len(all_fused_features) > 0:
            fusion_mean = np.mean(all_fused_features, axis=0)
            fusion_std = np.std(all_fused_features, axis=0)
            self.fusion_stats = (fusion_mean, fusion_std)
        
            self.output_manager.log_info(f"融合特征训练集均值: {np.mean(fusion_mean):.6f}")
            self.output_manager.log_info(f"融合特征训练集标准差: {np.mean(fusion_std):.6f}")
        # if all_fused_features:
        #     fusion_mean = np.mean(all_fused_features, axis=0)
        #     fusion_std = np.std(all_fused_features, axis=0)
        #     self.fusion_stats = (fusion_mean, fusion_std)
            
        #     self.output_manager.log_info(f"融合特征训练集均值: {np.mean(fusion_mean):.6f}")
        #     self.output_manager.log_info(f"融合特征训练集标准差: {np.mean(fusion_std):.6f}")
        
        self.output_manager.log_info("特征融合模型训练完成！")

    def fusion_features(self):
        """执行特征融合"""
        self.output_manager.log_info("开始特征融合")
        
        try:
            # 获取特征文件路径
            feature1_path = self.config['fusion']['feature_paths']['feature1']
            feature2_path = self.config['fusion']['feature_paths']['feature2']
            
            # 加载特征
            self.output_manager.log_info(f"加载特征1: {feature1_path}")
            feature1 = np.load(feature1_path)
            self.output_manager.log_info(f"加载特征2: {feature2_path}")
            feature2 = np.load(feature2_path)
            
            self.output_manager.log_info(f"特征1形状: {feature1.shape}")
            self.output_manager.log_info(f"特征2形状: {feature2.shape}")
            self.output_manager.log_info(f"特征1均值: {np.mean(feature1):.6f}, 标准差: {np.std(feature1):.6f}")
            self.output_manager.log_info(f"特征2均值: {np.mean(feature2):.6f}, 标准差: {np.std(feature2):.6f}")
            
            # 训练模型
            self.train(feature1, feature2)
            
            # 加载最佳模型
            self.model.load_state_dict(torch.load(self.output_manager.get_fusion_model_path()))
            self.model.eval()
            
            # 特征融合
            with torch.no_grad():
                feature1_norm = (feature1 - self.feature1_stats[0]) / (self.feature1_stats[1] + 1e-8)
                feature2_norm = (feature2 - self.feature2_stats[0]) / (self.feature2_stats[1] + 1e-8)
                
                feature1_tensor = torch.FloatTensor(feature1_norm).to(self.device)
                feature2_tensor = torch.FloatTensor(feature2_norm).to(self.device)
                
                fused_feature = self.model(feature1_tensor, feature2_tensor)
                fused_feature_np = fused_feature.cpu().numpy()
            
            # 创建基于权重的混合统计信息进行反标准化
            # 这样结果会反映两个特征的真实影响
            if self.fusion_stats:
                # 使用训练中计算的融合特征统计信息
                fusion_mean, fusion_std = self.fusion_stats
                self.output_manager.log_info("使用融合特征的统计信息进行反标准化")
                fused_feature_real = fused_feature_np * fusion_std + fusion_mean
            else:
                # 如果没有融合统计信息，创建基于权重的混合统计信息
                mixed_mean = (
                    self.feature1_weight * self.feature1_stats[0] + 
                    self.feature2_weight * self.feature2_stats[0]
                )
                mixed_std = (
                    self.feature1_weight * self.feature1_stats[1] + 
                    self.feature2_weight * self.feature2_stats[1]
                )
                self.output_manager.log_info("使用加权混合统计信息进行反标准化")
                fused_feature_real = fused_feature_np * mixed_std + mixed_mean
            
            # 监控最终融合特征的统计信息
            self.output_manager.log_info(f"融合特征形状: {fused_feature_real.shape}")
            self.output_manager.log_info(f"融合特征均值: {fused_feature_real.mean():.6f}")
            self.output_manager.log_info(f"融合特征标准差: {fused_feature_real.std():.6f}")
            
            # 保存融合后的特征
            fused_path = self.output_manager.get_fusion_feature_path('fused_feature_real.npy')
            np.save(fused_path, fused_feature_real)
            
            # 同时保存原始特征1和特征2的结果以便比较
            np.save(self.output_manager.get_fusion_feature_path('feature1_original.npy'), feature1)
            np.save(self.output_manager.get_fusion_feature_path('feature2_original.npy'), feature2)
            
            self.output_manager.log_info("特征融合完成！")
            self.output_manager.log_info(f"融合后特征保存至: {fused_path}")
            
        except Exception as e:
            self.output_manager.log_error(f"特征融合过程出错: {str(e)}")
            raise

def fusion_features():
    output_manager = OutputManager()
    trainer = FusionTrainer(CONFIG, output_manager)
    trainer.fusion_features()

if __name__ == "__main__":
    fusion_features()