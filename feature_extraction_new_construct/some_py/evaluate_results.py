# # import numpy as np
# # import os
# # from config import CONFIG
# # from feature_evaluator import FeatureEvaluator, evaluate_features  # 假设上面的评估代码保存在feature_evaluator.py中

# # def load_and_evaluate_features():
# #     # 创建输出目录
# #     output_dir = "results/evaluation"
# #     os.makedirs(output_dir, exist_ok=True)
    
# #     # 加载原始特征
# #     feature1_path = CONFIG['mmtm']['feature_paths']['feature1']
# #     feature2_path = CONFIG['mmtm']['feature_paths']['feature2']
    
# #     original_features1 = np.load(feature1_path)
# #     original_features2 = np.load(feature2_path)
    
# #     # 加载融合特征
# #     fusion_feature1_path = CONFIG['mmtm']['fusion_feature_paths']['fusion_feature1']
# #     fusion_feature2_path = CONFIG['mmtm']['fusion_feature_paths']['fusion_feature2']
    
# #     fusion_features1 = np.load(fusion_feature1_path)
# #     fusion_features2 = np.load(fusion_feature2_path)
    
# #     # 创建GT值（根据您的说明，使用10作为GT）
# #     gt1 = np.full_like(original_features1, 15)
# #     gt2 = np.full_like(original_features2, 15)
    
# #     # 运行评估
# #     metrics = evaluate_features(
# #         original_features=(original_features1, original_features2),
# #         fusion_features=(fusion_features1, fusion_features2),
# #         gt_values=(gt1, gt2),
# #         output_dir=output_dir
# #     )
    
# #     # 打印评估结果
# #     print("\n=== 特征1评估结果 ===")
# #     print(f"原始特征MSE: {metrics['orig_mse_1']:.4f}")
# #     print(f"融合特征MSE: {metrics['fusion_mse_1']:.4f}")
# #     print(f"原始特征相关系数: {metrics['orig_corr_1']:.4f}")
# #     print(f"融合特征相关系数: {metrics['fusion_corr_1']:.4f}")
    
# #     print("\n=== 特征2评估结果 ===")
# #     print(f"原始特征MSE: {metrics['orig_mse_2']:.4f}")
# #     print(f"融合特征MSE: {metrics['fusion_mse_2']:.4f}")
# #     print(f"原始特征相关系数: {metrics['orig_corr_2']:.4f}")
# #     print(f"融合特征相关系数: {metrics['fusion_corr_2']:.4f}")
    
# #     print(f"\n评估图表已保存至: {output_dir}")

# # if __name__ == "__main__":
# #     load_and_evaluate_features()



# import numpy as np
# import os
# from config import CONFIG
# from feature_evaluator import FeatureEvaluator, evaluate_features

# def load_and_evaluate_features():
#     # 创建输出目录
#     output_dir = "results/evaluation"
#     os.makedirs(output_dir, exist_ok=True)
    
#     # 加载原始特征
#     feature1_path = CONFIG['fusion']['feature_paths']['feature1']
#     feature2_path = CONFIG['fusion']['feature_paths']['feature2']
    
#     original_features1 = np.load(feature1_path)
#     original_features2 = np.load(feature2_path)
    
#     # 加载融合特征
#     fusion_feature1_path = CONFIG['fusion']['fusion_feature_paths']['fusion_feature1']
#     fusion_feature2_path = CONFIG['fusion']['fusion_feature_paths']['fusion_feature2']
    
#     fusion_features1 = np.load(fusion_feature1_path)
#     fusion_features2 = np.load(fusion_feature2_path)
    
#     # 创建GT值（这里使用15作为GT）
#     gt1 = np.full_like(original_features1, 15)
#     gt2 = np.full_like(original_features2, 15)
    
#     # 运行评估
#     metrics = evaluate_features(
#         original_features=(original_features1, original_features2),
#         fusion_features=(fusion_features1, fusion_features2),
#         gt_values=(gt1, gt2),
#         output_dir=output_dir
#     )
    
#     # 打印评估结果
#     print("\n=== 特征1评估结果 ===")
#     print(f"原始特征MSE: {metrics['orig_mse_1']:.4f}")
#     print(f"融合特征MSE: {metrics['fusion_mse_1']:.4f}")
#     print(f"原始特征相关系数: {metrics['orig_corr_1']:.4f}")
#     print(f"融合特征相关系数: {metrics['fusion_corr_1']:.4f}")
    
#     print("\n=== 特征2评估结果 ===")
#     print(f"原始特征MSE: {metrics['orig_mse_2']:.4f}")
#     print(f"融合特征MSE: {metrics['fusion_mse_2']:.4f}")
#     print(f"原始特征相关系数: {metrics['orig_corr_2']:.4f}")
#     print(f"融合特征相关系数: {metrics['fusion_corr_2']:.4f}")
    
#     print(f"\n评估图表已保存至: {output_dir}")

# if __name__ == "__main__":
#     load_and_evaluate_features()


import numpy as np
import os
from config import CONFIG
from feature_evaluator import FeatureEvaluator, evaluate_features

def load_and_evaluate_features():
    # 创建输出目录
    output_dir = "results/evaluation"
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载原始特征
    feature1_path = CONFIG['fusion']['feature_paths']['feature1']
    feature2_path = CONFIG['fusion']['feature_paths']['feature2']
    
    original_features1 = np.load(feature1_path)
    original_features2 = np.load(feature2_path)
    
    # 加载融合特征
    fusion_feature1_path = CONFIG['fusion']['fusion_feature_paths']['fusion_feature1']
    fusion_feature2_path = CONFIG['fusion']['fusion_feature_paths']['fusion_feature2']
    
    fusion_features1 = np.load(fusion_feature1_path)
    fusion_features2 = np.load(fusion_feature2_path)
    
    # 创建GT值（这里使用15作为GT）
    gt1 = np.full_like(original_features1, 15)
    gt2 = np.full_like(original_features2, 15)
    
    # 运行评估
    metrics = evaluate_features(
        original_features=(original_features1, original_features2),
        fusion_features=(fusion_features1, fusion_features2),
        gt_values=(gt1, gt2),
        output_dir=output_dir
    )
    
    # 打印评估结果
    print("\n=== 特征1评估结果 ===")
    print(f"原始特征MSE: {metrics['orig_mse_1']:.4f}")
    print(f"融合特征MSE: {metrics['fusion_mse_1']:.4f}")
    print(f"原始特征相关系数: {metrics['orig_corr_1']:.4f}")
    print(f"融合特征相关系数: {metrics['fusion_corr_1']:.4f}")
    
    print("\n=== 特征2评估结果 ===")
    print(f"原始特征MSE: {metrics['orig_mse_2']:.4f}")
    print(f"融合特征MSE: {metrics['fusion_mse_2']:.4f}")
    print(f"原始特征相关系数: {metrics['orig_corr_2']:.4f}")
    print(f"融合特征相关系数: {metrics['fusion_corr_2']:.4f}")
    
    print(f"\n评估图表已保存至: {output_dir}")

if __name__ == "__main__":
    load_and_evaluate_features()