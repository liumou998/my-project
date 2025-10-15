# # import numpy as np
# # import matplotlib.pyplot as plt
# # from sklearn.metrics import mean_squared_error, mean_absolute_error
# # import seaborn as sns

# # class FeatureEvaluator:
# #     def __init__(self):
# #         self.metrics = {}
    
# #     def calculate_metrics(self, original_features, fusion_features, gt_values):
# #         """
# #         计算各种评估指标
        
# #         Args:
# #             original_features: tuple (feature1, feature2) 原始特征
# #             fusion_features: tuple (fusion1, fusion2) 融合后特征
# #             gt_values: tuple (gt1, gt2) 目标值
# #         """
# #         feature1, feature2 = original_features
# #         fusion1, fusion2 = fusion_features
# #         gt1, gt2 = gt_values
        
# #         # 计算MSE
# #         self.metrics['orig_mse_1'] = mean_squared_error(feature1, gt1)
# #         self.metrics['orig_mse_2'] = mean_squared_error(feature2, gt2)
# #         self.metrics['fusion_mse_1'] = mean_squared_error(fusion1, gt1)
# #         self.metrics['fusion_mse_2'] = mean_squared_error(fusion2, gt2)
        
# #         # 计算MAE
# #         self.metrics['orig_mae_1'] = mean_absolute_error(feature1, gt1)
# #         self.metrics['orig_mae_2'] = mean_absolute_error(feature2, gt2)
# #         self.metrics['fusion_mae_1'] = mean_absolute_error(fusion1, gt1)
# #         self.metrics['fusion_mae_2'] = mean_absolute_error(fusion2, gt2)
        
# #         # 计算相关系数
# #         self.metrics['orig_corr_1'] = np.corrcoef(feature1, gt1)[0,1]
# #         self.metrics['orig_corr_2'] = np.corrcoef(feature2, gt2)[0,1]
# #         self.metrics['fusion_corr_1'] = np.corrcoef(fusion1, gt1)[0,1]
# #         self.metrics['fusion_corr_2'] = np.corrcoef(fusion2, gt2)[0,1]
    
# #     def plot_feature_comparison(self, original_features, fusion_features, gt_values, 
# #                               save_path=None):
# #         """
# #         绘制特征对比图
# #         """
# #         feature1, feature2 = original_features
# #         fusion1, fusion2 = fusion_features
# #         gt1, gt2 = gt_values
        
# #         # 创建图形
# #         fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
# #         plt.style.use('seaborn')
        
# #         # 绘制Feature 1的对比
# #         time = np.arange(len(feature1))
# #         ax1.plot(time, feature1, 'b-', label='Original Feature 1', alpha=0.6)
# #         ax1.plot(time, fusion1, 'r-', label='Fusion Feature 1', alpha=0.6)
# #         ax1.plot(time, gt1, 'g--', label='GT 1', alpha=0.6)
# #         ax1.set_title('Feature 1 Comparison')
# #         ax1.set_xlabel('Time Steps')
# #         ax1.set_ylabel('Value')
# #         ax1.legend()
# #         ax1.grid(True, alpha=0.3)
        
# #         # 添加MSE和相关系数注释
# #         ax1.text(0.02, 0.98, 
# #                 f'Original MSE: {self.metrics["orig_mse_1"]:.4f}\n'
# #                 f'Fusion MSE: {self.metrics["fusion_mse_1"]:.4f}\n'
# #                 f'Original Corr: {self.metrics["orig_corr_1"]:.4f}\n'
# #                 f'Fusion Corr: {self.metrics["fusion_corr_1"]:.4f}',
# #                 transform=ax1.transAxes, 
# #                 verticalalignment='top',
# #                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
# #         # 绘制Feature 2的对比
# #         ax2.plot(time, feature2, 'b-', label='Original Feature 2', alpha=0.6)
# #         ax2.plot(time, fusion2, 'r-', label='Fusion Feature 2', alpha=0.6)
# #         ax2.plot(time, gt2, 'g--', label='GT 2', alpha=0.6)
# #         ax2.set_title('Feature 2 Comparison')
# #         ax2.set_xlabel('Time Steps')
# #         ax2.set_ylabel('Value')
# #         ax2.legend()
# #         ax2.grid(True, alpha=0.3)
        
# #         # 添加MSE和相关系数注释
# #         ax2.text(0.02, 0.98, 
# #                 f'Original MSE: {self.metrics["orig_mse_2"]:.4f}\n'
# #                 f'Fusion MSE: {self.metrics["fusion_mse_2"]:.4f}\n'
# #                 f'Original Corr: {self.metrics["orig_corr_2"]:.4f}\n'
# #                 f'Fusion Corr: {self.metrics["fusion_corr_2"]:.4f}',
# #                 transform=ax2.transAxes, 
# #                 verticalalignment='top',
# #                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
# #         plt.tight_layout()
        
# #         if save_path:
# #             plt.savefig(save_path, dpi=300, bbox_inches='tight')
# #             plt.close()
# #         else:
# #             plt.show()
    
# #     def plot_error_distribution(self, original_features, fusion_features, gt_values, 
# #                               save_path=None):
# #         """
# #         绘制误差分布图
# #         """
# #         feature1, feature2 = original_features
# #         fusion1, fusion2 = fusion_features
# #         gt1, gt2 = gt_values
        
# #         # 计算误差
# #         orig_error1 = np.abs(feature1 - gt1)
# #         orig_error2 = np.abs(feature2 - gt2)
# #         fusion_error1 = np.abs(fusion1 - gt1)
# #         fusion_error2 = np.abs(fusion2 - gt2)
        
# #         # 创建图形
# #         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
# #         # Feature 1的误差分布
# #         sns.kdeplot(data=orig_error1, ax=ax1, label='Original Error 1', alpha=0.6)
# #         sns.kdeplot(data=fusion_error1, ax=ax1, label='Fusion Error 1', alpha=0.6)
# #         ax1.set_title('Feature 1 Error Distribution')
# #         ax1.set_xlabel('Absolute Error')
# #         ax1.set_ylabel('Density')
# #         ax1.legend()
        
# #         # Feature 2的误差分布
# #         sns.kdeplot(data=orig_error2, ax=ax2, label='Original Error 2', alpha=0.6)
# #         sns.kdeplot(data=fusion_error2, ax=ax2, label='Fusion Error 2', alpha=0.6)
# #         ax2.set_title('Feature 2 Error Distribution')
# #         ax2.set_xlabel('Absolute Error')
# #         ax2.set_ylabel('Density')
# #         ax2.legend()
        
# #         plt.tight_layout()
        
# #         if save_path:
# #             plt.savefig(save_path, dpi=300, bbox_inches='tight')
# #             plt.close()
# #         else:
# #             plt.show()

# # # 使用示例
# # def evaluate_features(original_features, fusion_features, gt_values, output_dir=None):
# #     """
# #     评估特征融合效果
    
# #     Args:
# #         original_features: tuple (feature1, feature2) 原始特征
# #         fusion_features: tuple (fusion1, fusion2) 融合后特征
# #         gt_values: tuple (gt1, gt2) 目标值
# #         output_dir: str, 可选，图片保存目录
# #     """
# #     evaluator = FeatureEvaluator()
    
# #     # 计算评估指标
# #     evaluator.calculate_metrics(original_features, fusion_features, gt_values)
    
# #     # 绘制特征对比图
# #     if output_dir:
# #         comparison_path = f"{output_dir}/feature_comparison.png"
# #         error_dist_path = f"{output_dir}/error_distribution.png"
# #     else:
# #         comparison_path = None
# #         error_dist_path = None
    
# #     evaluator.plot_feature_comparison(original_features, fusion_features, gt_values, 
# #                                     comparison_path)
# #     evaluator.plot_error_distribution(original_features, fusion_features, gt_values, 
# #                                     error_dist_path)
    
# #     return evaluator.metrics


# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# import seaborn as sns

# class FeatureEvaluator:
#     def __init__(self):
#         self.metrics = {}
    
#     def calculate_metrics(self, original_features, fusion_features, gt_values):
#         """
#         计算各种评估指标
        
#         Args:
#             original_features: tuple (feature1, feature2) 原始特征
#             fusion_features: tuple (fusion1, fusion2) 融合后特征
#             gt_values: tuple (gt1, gt2) 目标值
#         """
#         feature1, feature2 = original_features
#         fusion1, fusion2 = fusion_features
#         gt1, gt2 = gt_values
        
#         # 计算MSE
#         self.metrics['orig_mse_1'] = mean_squared_error(feature1, gt1)
#         self.metrics['orig_mse_2'] = mean_squared_error(feature2, gt2)
#         self.metrics['fusion_mse_1'] = mean_squared_error(fusion1, gt1)
#         self.metrics['fusion_mse_2'] = mean_squared_error(fusion2, gt2)
        
#         # 计算MAE
#         self.metrics['orig_mae_1'] = mean_absolute_error(feature1, gt1)
#         self.metrics['orig_mae_2'] = mean_absolute_error(feature2, gt2)
#         self.metrics['fusion_mae_1'] = mean_absolute_error(fusion1, gt1)
#         self.metrics['fusion_mae_2'] = mean_absolute_error(fusion2, gt2)
        
#         # 计算相关系数
#         self.metrics['orig_corr_1'] = np.corrcoef(feature1, gt1)[0,1]
#         self.metrics['orig_corr_2'] = np.corrcoef(feature2, gt2)[0,1]
#         self.metrics['fusion_corr_1'] = np.corrcoef(fusion1, gt1)[0,1]
#         self.metrics['fusion_corr_2'] = np.corrcoef(fusion2, gt2)[0,1]
    
#     def plot_feature_comparison(self, original_features, fusion_features, gt_values, save_path=None):
#         """
#         绘制特征对比图 - 改进版
#         """
#         feature1, feature2 = original_features
#         fusion1, fusion2 = fusion_features
#         gt1, gt2 = gt_values
        
#         # 创建图形
#         fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
#         plt.style.use('seaborn')
        
#         # 设置统一的时间轴
#         time = np.arange(len(feature1))
        
#         # 可以选择只显示部分数据点，比如1000个点
#         window_size = 1000
#         start_idx = 0
#         end_idx = start_idx + window_size if len(time) > window_size else len(time)
        
#         # Feature 1的原始特征与GT对比
#         ax1.plot(time[start_idx:end_idx], feature1[start_idx:end_idx], 'b-', 
#                 label='Original Feature 1', alpha=0.6)
#         ax1.plot(time[start_idx:end_idx], gt1[start_idx:end_idx], 'g--', 
#                 label='GT 1', alpha=0.6)
#         ax1.set_title('Original Feature 1 vs GT')
#         ax1.set_xlabel('Time Steps')
#         ax1.set_ylabel('Value')
#         ax1.legend()
#         ax1.grid(True, alpha=0.3)
        
#         # Feature 1的融合特征与GT对比
#         ax2.plot(time[start_idx:end_idx], fusion1[start_idx:end_idx], 'r-', 
#                 label='Fusion Feature 1', alpha=0.6)
#         ax2.plot(time[start_idx:end_idx], gt1[start_idx:end_idx], 'g--', 
#                 label='GT 1', alpha=0.6)
#         ax2.set_title('Fusion Feature 1 vs GT')
#         ax2.set_xlabel('Time Steps')
#         ax2.set_ylabel('Value')
#         ax2.legend()
#         ax2.grid(True, alpha=0.3)
        
#         # Feature 2的原始特征与GT对比
#         ax3.plot(time[start_idx:end_idx], feature2[start_idx:end_idx], 'b-', 
#                 label='Original Feature 2', alpha=0.6)
#         ax3.plot(time[start_idx:end_idx], gt2[start_idx:end_idx], 'g--', 
#                 label='GT 2', alpha=0.6)
#         ax3.set_title('Original Feature 2 vs GT')
#         ax3.set_xlabel('Time Steps')
#         ax3.set_ylabel('Value')
#         ax3.legend()
#         ax3.grid(True, alpha=0.3)
        
#         # Feature 2的融合特征与GT对比
#         ax4.plot(time[start_idx:end_idx], fusion2[start_idx:end_idx], 'r-', 
#                 label='Fusion Feature 2', alpha=0.6)
#         ax4.plot(time[start_idx:end_idx], gt2[start_idx:end_idx], 'g--', 
#                 label='GT 2', alpha=0.6)
#         ax4.set_title('Fusion Feature 2 vs GT')
#         ax4.set_xlabel('Time Steps')
#         ax4.set_ylabel('Value')
#         ax4.legend()
#         ax4.grid(True, alpha=0.3)
        
#         # 添加MSE和相关系数注释到每个子图
#         ax1.text(0.02, 0.98, 
#                 f'MSE: {self.metrics["orig_mse_1"]:.4f}\n'
#                 f'Corr: {self.metrics["orig_corr_1"]:.4f}',
#                 transform=ax1.transAxes, 
#                 verticalalignment='top',
#                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
#         ax2.text(0.02, 0.98, 
#                 f'MSE: {self.metrics["fusion_mse_1"]:.4f}\n'
#                 f'Corr: {self.metrics["fusion_corr_1"]:.4f}',
#                 transform=ax2.transAxes, 
#                 verticalalignment='top',
#                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
#         ax3.text(0.02, 0.98, 
#                 f'MSE: {self.metrics["orig_mse_2"]:.4f}\n'
#                 f'Corr: {self.metrics["orig_corr_2"]:.4f}',
#                 transform=ax3.transAxes, 
#                 verticalalignment='top',
#                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
#         ax4.text(0.02, 0.98, 
#                 f'MSE: {self.metrics["fusion_mse_2"]:.4f}\n'
#                 f'Corr: {self.metrics["fusion_corr_2"]:.4f}',
#                 transform=ax4.transAxes, 
#                 verticalalignment='top',
#                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
#         plt.tight_layout()
        
#         if save_path:
#             plt.savefig(save_path, dpi=300, bbox_inches='tight')
#             plt.close()
#         else:
#             plt.show()
            
    
#     def plot_error_distribution(self, original_features, fusion_features, gt_values, save_path=None):
#         """
#         绘制误差分布图
#         Plot error distribution with modified colors
#         """
#         feature1, feature2 = original_features
#         fusion1, fusion2 = fusion_features
#         gt1, gt2 = gt_values
        
#         # 计算误差
#         orig_error1 = np.abs(feature1 - gt1)
#         orig_error2 = np.abs(feature2 - gt2)
#         fusion_error1 = np.abs(fusion1 - gt1)
#         fusion_error2 = np.abs(fusion2 - gt2)
        
#         # 创建图形
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
#         # 新的配色方案
#         # # Feature 1 colors
#         # orig_color1 = '#2E86C1'  # 深蓝色
#         # fusion_color1 = '#E74C3C'  # 红色
        
#         # # Feature 2 colors
#         # orig_color2 = '#27AE60'  # 绿色
#         # fusion_color2 = '#8E44AD'  # 紫色
        
#         # Feature 1的误差分布
#         sns.kdeplot(data=orig_error1, ax=ax1, label='Original Error 1', 
#                     alpha=0.6, color='blue', linestyle='-', linewidth=2)
#         sns.kdeplot(data=fusion_error1, ax=ax1, label='Fusion Error 1', 
#                     alpha=0.6, color=f'orange', linestyle='--', linewidth=2.5)
#         ax1.set_title('Feature 1 Error Distribution', fontsize=12, pad=15)
#         ax1.set_xlabel('Absolute Error', fontsize=10)
#         ax1.set_ylabel('Density', fontsize=10)
#         ax1.legend(fontsize=10)
        
#         # Feature 2的误差分布
#         sns.kdeplot(data=orig_error2, ax=ax2, label='Original Error 2', 
#                     alpha=0.6, color='purple', linestyle='-', linewidth=2)
#         sns.kdeplot(data=fusion_error2, ax=ax2, label='Fusion Error 2', 
#                     alpha=0.6, color='red', linestyle='--', linewidth=2.5)
#         ax2.set_title('Feature 2 Error Distribution', fontsize=12, pad=15)
#         ax2.set_xlabel('Absolute Error', fontsize=10)
#         ax2.set_ylabel('Density', fontsize=10)
#         ax2.legend(fontsize=10)
        
#         plt.tight_layout()
        
#         if save_path:
#             plt.savefig(save_path, dpi=300, bbox_inches='tight')
#             plt.close()
#         else:
#             plt.show()


#     # def plot_error_distribution(self, original_features, fusion_features, gt_values, save_path=None):
#     #     """
#     #     绘制误差分布图
#     #     """
#     #     feature1, feature2 = original_features
#     #     fusion1, fusion2 = fusion_features
#     #     gt1, gt2 = gt_values
        
#     #     # 计算误差
#     #     orig_error1 = np.abs(feature1 - gt1)
#     #     orig_error2 = np.abs(feature2 - gt2)
#     #     fusion_error1 = np.abs(fusion1 - gt1)
#     #     fusion_error2 = np.abs(fusion2 - gt2)
        
#     #     # 创建图形
#     #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
#     #     # Feature 1的误差分布
#     #     sns.kdeplot(data=orig_error1, ax=ax1, label='Original Error 1', alpha=0.6, color='orange', linestyle='-', linewidth='2')
#     #     sns.kdeplot(data=fusion_error1, ax=ax1, label='Fusion Error 1', alpha=0.6, color='purple', linestyle='--', linewidth='3')
#     #     ax1.set_title('Feature 1 Error Distribution')
#     #     ax1.set_xlabel('Absolute Error')
#     #     ax1.set_ylabel('Density')
#     #     ax1.legend()
        
#     #     # Feature 2的误差分布
#     #     sns.kdeplot(data=orig_error2, ax=ax2, label='Original Error 2', alpha=0.6, color='blue', linestyle='-', linewidth='2')
#     #     sns.kdeplot(data=fusion_error2, ax=ax2, label='Fusion Error 2', alpha=0.6, color='red', linestyle='--', linewidth='3')
#     #     ax2.set_title('Feature 2 Error Distribution')
#     #     ax2.set_xlabel('Absolute Error')
#     #     ax2.set_ylabel('Density')
#     #     ax2.legend()
        
#     #     plt.tight_layout()
        
#     #     if save_path:
#     #         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     #         plt.close()
#     #     else:
#     #         plt.show()
            
    
# def evaluate_features(original_features, fusion_features, gt_values, output_dir=None):
#     """
#     评估特征融合效果
    
#     Args:
#         original_features: tuple (feature1, feature2) 原始特征
#         fusion_features: tuple (fusion1, fusion2) 融合后特征
#         gt_values: tuple (gt1, gt2) 目标值
#         output_dir: str, 可选，图片保存目录
#     """
#     evaluator = FeatureEvaluator()
    
#     # 计算评估指标
#     evaluator.calculate_metrics(original_features, fusion_features, gt_values)
    
#     # 绘制特征对比图
#     if output_dir:
#         comparison_path = f"{output_dir}/feature_comparison.png"
#         error_dist_path = f"{output_dir}/error_distribution.png"
#     else:
#         comparison_path = None
#         error_dist_path = None
    
#     evaluator.plot_feature_comparison(original_features, fusion_features, gt_values, 
#                                     comparison_path)
#     evaluator.plot_error_distribution(original_features, fusion_features, gt_values, 
#                                     error_dist_path)

    
#     return evaluator.metrics


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns

class FeatureEvaluator:
    def __init__(self):
        self.metrics = {}
    
    def calculate_metrics(self, original_features, fusion_features, gt_values):
        """
        计算各种评估指标
        
        Args:
            original_features: tuple (feature1, feature2) 原始特征
            fusion_features: tuple (fusion1, fusion2) 融合后特征
            gt_values: tuple (gt1, gt2) 目标值
        """
        feature1, feature2 = original_features
        fusion1, fusion2 = fusion_features
        gt1, gt2 = gt_values
        
        # 计算MSE
        self.metrics['orig_mse_1'] = mean_squared_error(feature1, gt1)
        self.metrics['orig_mse_2'] = mean_squared_error(feature2, gt2)
        self.metrics['fusion_mse_1'] = mean_squared_error(fusion1, gt1)
        self.metrics['fusion_mse_2'] = mean_squared_error(fusion2, gt2)
        
        # 计算MAE
        self.metrics['orig_mae_1'] = mean_absolute_error(feature1, gt1)
        self.metrics['orig_mae_2'] = mean_absolute_error(feature2, gt2)
        self.metrics['fusion_mae_1'] = mean_absolute_error(fusion1, gt1)
        self.metrics['fusion_mae_2'] = mean_absolute_error(fusion2, gt2)
        
        # 计算相关系数
        self.metrics['orig_corr_1'] = np.corrcoef(feature1, gt1)[0,1]
        self.metrics['orig_corr_2'] = np.corrcoef(feature2, gt2)[0,1]
        self.metrics['fusion_corr_1'] = np.corrcoef(fusion1, gt1)[0,1]
        self.metrics['fusion_corr_2'] = np.corrcoef(fusion2, gt2)[0,1]
    
    def plot_feature_comparison(self, original_features, fusion_features, gt_values, save_path=None):
        """
        绘制特征对比图 - 针对常数GT值(15)的改进版
        """
        feature1, feature2 = original_features
        fusion1, fusion2 = fusion_features
        gt1, gt2 = gt_values
        
        # 创建图形
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        plt.style.use('seaborn')
        
        # 设置统一的时间轴
        time = np.arange(len(feature1))
        
        # 可以选择只显示部分数据点，比如1000个点
        window_size = 1000
        start_idx = 0
        end_idx = start_idx + window_size if len(time) > window_size else len(time)
        
        # 计算Y轴范围以便更好地展示数据
        # 获取所有特征的最大最小值
        all_features = np.concatenate([
            feature1[start_idx:end_idx], 
            feature2[start_idx:end_idx],
            fusion1[start_idx:end_idx], 
            fusion2[start_idx:end_idx]
        ])
        
        # 确定Y轴的上下限 - 留出一定空间使GT值(15)在图中位置合适
        min_val = min(np.min(all_features), 15) - 2  # GT值(15)的下方留出空间
        max_val = max(np.max(all_features), 15) + 2  # GT值(15)的上方留出空间
        
        # Feature 1的原始特征与GT对比
        ax1.plot(time[start_idx:end_idx], feature1[start_idx:end_idx], 'b-', 
                label='Original Feature 1', alpha=0.6)
        ax1.axhline(y=15, color='g', linestyle='--', 
                label='GT (15)', alpha=0.6, linewidth=2)
        ax1.set_title('Original Feature 1 vs GT')
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Value')
        ax1.set_ylim(min_val, max_val)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Feature 1的融合特征与GT对比
        ax2.plot(time[start_idx:end_idx], fusion1[start_idx:end_idx], 'r-', 
                label='Fusion Feature 1', alpha=0.6)
        ax2.axhline(y=15, color='g', linestyle='--', 
                label='GT (15)', alpha=0.6, linewidth=2)
        ax2.set_title('Fusion Feature 1 vs GT')
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Value')
        ax2.set_ylim(min_val, max_val)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Feature 2的原始特征与GT对比
        ax3.plot(time[start_idx:end_idx], feature2[start_idx:end_idx], 'b-', 
                label='Original Feature 2', alpha=0.6)
        ax3.axhline(y=15, color='g', linestyle='--', 
                label='GT (15)', alpha=0.6, linewidth=2)
        ax3.set_title('Original Feature 2 vs GT')
        ax3.set_xlabel('Time Steps')
        ax3.set_ylabel('Value')
        ax3.set_ylim(min_val, max_val)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Feature 2的融合特征与GT对比
        ax4.plot(time[start_idx:end_idx], fusion2[start_idx:end_idx], 'r-', 
                label='Fusion Feature 2', alpha=0.6)
        ax4.axhline(y=15, color='g', linestyle='--', 
                label='GT (15)', alpha=0.6, linewidth=2)
        ax4.set_title('Fusion Feature 2 vs GT')
        ax4.set_xlabel('Time Steps')
        ax4.set_ylabel('Value')
        ax4.set_ylim(min_val, max_val)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 添加MSE和相关系数注释到每个子图
        ax1.text(0.02, 0.98, 
                f'MSE: {self.metrics["orig_mse_1"]:.4f}\n'
                f'MAE: {self.metrics["orig_mae_1"]:.4f}\n'
                f'Corr: {self.metrics["orig_corr_1"]:.4f}',
                transform=ax1.transAxes, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax2.text(0.02, 0.98, 
                f'MSE: {self.metrics["fusion_mse_1"]:.4f}\n'
                f'MAE: {self.metrics["fusion_mae_1"]:.4f}\n'
                f'Corr: {self.metrics["fusion_corr_1"]:.4f}',
                transform=ax2.transAxes, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax3.text(0.02, 0.98, 
                f'MSE: {self.metrics["orig_mse_2"]:.4f}\n'
                f'MAE: {self.metrics["orig_mae_2"]:.4f}\n'
                f'Corr: {self.metrics["orig_corr_2"]:.4f}',
                transform=ax3.transAxes, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax4.text(0.02, 0.98, 
                f'MSE: {self.metrics["fusion_mse_2"]:.4f}\n'
                f'MAE: {self.metrics["fusion_mae_2"]:.4f}\n'
                f'Corr: {self.metrics["fusion_corr_2"]:.4f}',
                transform=ax4.transAxes, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 添加标题，显示这是针对恒定GT值的评估
        fig.suptitle(f'Feature Evaluation Against Constant GT Value (15)', fontsize=16)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)  # 为suptitle腾出空间
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
            
    
    def plot_error_distribution(self, original_features, fusion_features, gt_values, save_path=None):
        """
        绘制误差分布图 - 针对常数GT值的改进版
        """
        feature1, feature2 = original_features
        fusion1, fusion2 = fusion_features
        gt1, gt2 = gt_values
        
        # 计算误差
        orig_error1 = np.abs(feature1 - gt1)
        orig_error2 = np.abs(feature2 - gt2)
        fusion_error1 = np.abs(fusion1 - gt1)
        fusion_error2 = np.abs(fusion2 - gt2)
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Feature 1的误差分布
        sns.kdeplot(data=orig_error1, ax=ax1, label='Original Error 1', 
                    alpha=0.7, color='blue', linestyle='-', linewidth=2)
        sns.kdeplot(data=fusion_error1, ax=ax1, label='Fusion Error 1', 
                    alpha=0.7, color='red', linestyle='--', linewidth=2.5)
        ax1.set_title('Feature 1 Error Distribution (Deviation from GT=15)', fontsize=12, pad=15)
        ax1.set_xlabel('Absolute Error', fontsize=10)
        ax1.set_ylabel('Density', fontsize=10)
        
        # 添加平均误差标记
        ax1.axvline(x=np.mean(orig_error1), color='blue', linestyle=':', alpha=0.8,
                   label=f'Orig Avg: {np.mean(orig_error1):.4f}')
        ax1.axvline(x=np.mean(fusion_error1), color='red', linestyle=':', alpha=0.8,
                   label=f'Fusion Avg: {np.mean(fusion_error1):.4f}')
        ax1.legend(fontsize=10)
        
        # Feature 2的误差分布
        sns.kdeplot(data=orig_error2, ax=ax2, label='Original Error 2', 
                    alpha=0.7, color='blue', linestyle='-', linewidth=2)
        sns.kdeplot(data=fusion_error2, ax=ax2, label='Fusion Error 2', 
                    alpha=0.7, color='red', linestyle='--', linewidth=2.5)
        ax2.set_title('Feature 2 Error Distribution (Deviation from GT=15)', fontsize=12, pad=15)
        ax2.set_xlabel('Absolute Error', fontsize=10)
        ax2.set_ylabel('Density', fontsize=10)
        
        # 添加平均误差标记
        ax2.axvline(x=np.mean(orig_error2), color='blue', linestyle=':', alpha=0.8,
                   label=f'Orig Avg: {np.mean(orig_error2):.4f}')
        ax2.axvline(x=np.mean(fusion_error2), color='red', linestyle=':', alpha=0.8,
                   label=f'Fusion Avg: {np.mean(fusion_error2):.4f}')
        ax2.legend(fontsize=10)
        
        # 添加总体评估注释
        improvement1 = (1 - (np.mean(fusion_error1) / np.mean(orig_error1))) * 100
        improvement2 = (1 - (np.mean(fusion_error2) / np.mean(orig_error2))) * 100
        
        fig.suptitle(f'Error Distribution Analysis Against Constant GT Value (15)', fontsize=14)
        
        # 添加改进百分比注释
        fig.text(0.5, 0.01, 
                f'Feature 1 Improvement: {improvement1:.2f}% | Feature 2 Improvement: {improvement2:.2f}%',
                ha='center', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='lightgrey', alpha=0.5))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)  # 为底部注释腾出空间
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def evaluate_features(original_features, fusion_features, gt_values, output_dir=None):
    """
    评估特征融合效果
    
    Args:
        original_features: tuple (feature1, feature2) 原始特征
        fusion_features: tuple (fusion1, fusion2) 融合后特征
        gt_values: tuple (gt1, gt2) 目标值
        output_dir: str, 可选，图片保存目录
    """
    evaluator = FeatureEvaluator()
    
    # 计算评估指标
    evaluator.calculate_metrics(original_features, fusion_features, gt_values)
    
    # 绘制特征对比图
    if output_dir:
        comparison_path = f"{output_dir}/feature_comparison.png"
        error_dist_path = f"{output_dir}/error_distribution.png"
    else:
        comparison_path = None
        error_dist_path = None
    
    evaluator.plot_feature_comparison(original_features, fusion_features, gt_values, 
                                    comparison_path)
    evaluator.plot_error_distribution(original_features, fusion_features, gt_values, 
                                    error_dist_path)
    
    return evaluator.metrics