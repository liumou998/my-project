# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import mean_squared_error
# from scipy.stats import pearsonr
# import math
# import os

# print("=" * 60)
# print("ADAS评估分析工具 - 精细化版本")
# print("=" * 60)

# # 运行时输入文件路径
# adas1_path = "/home/linux/ShenGang/feature_extraction_new_construct/results/new_log_result/extract/adas1_feature_gt=15/features/jms2_test_reduced_feature.npy"
# adas2_path = "/home/linux/ShenGang/feature_extraction_new_construct/results/new_log_result/extract/adas2_feature_gt=15/features/jms2_test_reduced_feature.npy"
# adas3_path = "/home/linux/ShenGang/feature_extraction_new_construct/results/new_log_result/extract/jms2_adas3_gt=15/features/train_features.npy"
# fused_path = "/home/linux/ShenGang/feature_extraction_new_construct/results/new_log_result/fusion/fusion_test29/fusion_features/jms2_test_reduced_feature.npy"

# # 设置输出文件夹
# output_dir = "/home/linux/ShenGang/feature_extraction_new_construct/results/new_log_result/evaluation_results"
# os.makedirs(output_dir, exist_ok=True)

# # 加载数据
# try:
#     adas1_features = np.load(adas1_path)
#     print(f"成功加载ADAS1特征，形状为: {adas1_features.shape}")
# except Exception as e:
#     print(f"加载ADAS1特征时出错: {str(e)}")
#     exit(1)

# try:
#     adas2_features = np.load(adas2_path)
#     print(f"成功加载ADAS2特征，形状为: {adas2_features.shape}")
# except Exception as e:
#     print(f"加载ADAS2特征时出错: {str(e)}")
#     exit(1)

# try:
#     adas3_features = np.load(adas3_path)
#     print(f"成功加载ADAS3特征，形状为: {adas3_features.shape}")
# except Exception as e:
#     print(f"加载ADAS3特征时出错: {str(e)}")
#     exit(1)

# try:
#     fused_features = np.load(fused_path)
#     print(f"成功加载融合特征，形状为: {fused_features.shape}")
# except Exception as e:
#     print(f"加载融合特征时出错: {str(e)}")
#     exit(1)

# # 检查所有数据的行数是否一致
# shapes = [arr.shape[0] for arr in [adas1_features, adas2_features, adas3_features, fused_features]]
# if len(set(shapes)) > 1:
#     print(f"警告: 数据行数不一致! 行数分别为: {shapes}")
#     print("为确保比较的准确性，将截断至最小长度")
#     min_rows = min(shapes)
#     adas1_features = adas1_features[:min_rows]
#     adas2_features = adas2_features[:min_rows]
#     adas3_features = adas3_features[:min_rows]
#     fused_features = fused_features[:min_rows]
#     print(f"所有数据已截断至 {min_rows} 行")

# num_rows = adas1_features.shape[0]
# print(f"数据总行数: {num_rows}")

# # 计算逐行MSE和RMSE
# def calculate_row_metrics(target, prediction):
#     """计算每一行的MSE和RMSE"""
#     row_mse = []
#     row_rmse = []
    
#     for i in range(target.shape[0]):
#         # 计算每一行的MSE
#         mse = mean_squared_error(target[i], prediction[i])
#         # 计算每一行的RMSE
#         rmse = math.sqrt(mse)
        
#         row_mse.append(mse)
#         row_rmse.append(rmse)
    
#     return np.array(row_mse), np.array(row_rmse)

# # 计算每组实验的逐行指标
# print("\n开始计算逐行指标...")

# # 实验1: ADAS3 vs ADAS1
# mse1_by_row, rmse1_by_row = calculate_row_metrics(adas3_features, adas1_features)
# print(f"实验1 (ADAS3 vs ADAS1) - 平均MSE: {np.mean(mse1_by_row):.4f}, 平均RMSE: {np.mean(rmse1_by_row):.4f}")

# # 实验2: ADAS3 vs ADAS2
# mse2_by_row, rmse2_by_row = calculate_row_metrics(adas3_features, adas2_features)
# print(f"实验2 (ADAS3 vs ADAS2) - 平均MSE: {np.mean(mse2_by_row):.4f}, 平均RMSE: {np.mean(rmse2_by_row):.4f}")

# # 实验3: ADAS3 vs 融合结果
# mse3_by_row, rmse3_by_row = calculate_row_metrics(adas3_features, fused_features)
# print(f"实验3 (ADAS3 vs 融合结果) - 平均MSE: {np.mean(mse3_by_row):.4f}, 平均RMSE: {np.mean(rmse3_by_row):.4f}")

# # 绘制实验1和实验3的对比图
# print("\n绘制实验1和实验3的对比图...")
# plt.figure(figsize=(15, 10))

# # MSE对比
# plt.subplot(2, 1, 1)
# plt.plot(range(num_rows), mse1_by_row, 'b-', alpha=0.7, label='ADAS3 vs ADAS1')
# plt.plot(range(num_rows), mse3_by_row, 'r-', alpha=0.7, label='ADAS3 vs 融合结果')
# plt.title('实验1 vs 实验3: 逐行MSE对比 (越低越好)', fontsize=14)
# plt.xlabel('数据行索引', fontsize=12)
# plt.ylabel('MSE值', fontsize=12)
# plt.legend(fontsize=12)
# plt.grid(True, linestyle='--', alpha=0.7)

# # 添加水平平均线
# plt.axhline(y=np.mean(mse1_by_row), color='b', linestyle='--', 
#             label=f'ADAS1平均: {np.mean(mse1_by_row):.4f}')
# plt.axhline(y=np.mean(mse3_by_row), color='r', linestyle='--', 
#             label=f'融合平均: {np.mean(mse3_by_row):.4f}')
# plt.legend(fontsize=10)

# # RMSE对比
# plt.subplot(2, 1, 2)
# plt.plot(range(num_rows), rmse1_by_row, 'b-', alpha=0.7, label='ADAS3 vs ADAS1')
# plt.plot(range(num_rows), rmse3_by_row, 'r-', alpha=0.7, label='ADAS3 vs 融合结果')
# plt.title('实验1 vs 实验3: 逐行RMSE对比 (越低越好)', fontsize=14)
# plt.xlabel('数据行索引', fontsize=12)
# plt.ylabel('RMSE值', fontsize=12)
# plt.legend(fontsize=12)
# plt.grid(True, linestyle='--', alpha=0.7)

# # 添加水平平均线
# plt.axhline(y=np.mean(rmse1_by_row), color='b', linestyle='--', 
#             label=f'ADAS1平均: {np.mean(rmse1_by_row):.4f}')
# plt.axhline(y=np.mean(rmse3_by_row), color='r', linestyle='--', 
#             label=f'融合平均: {np.mean(rmse3_by_row):.4f}')
# plt.legend(fontsize=10)

# plt.tight_layout()
# comparison1_filename = os.path.join(output_dir, "exp1_vs_exp3_comparison.png")
# plt.savefig(comparison1_filename, dpi=300)
# print(f"图表已保存为 {comparison1_filename}")

# # 绘制实验2和实验3的对比图
# print("\n绘制实验2和实验3的对比图...")
# plt.figure(figsize=(15, 10))

# # MSE对比
# plt.subplot(2, 1, 1)
# plt.plot(range(num_rows), mse2_by_row, 'g-', alpha=0.7, label='ADAS3 vs ADAS2')
# plt.plot(range(num_rows), mse3_by_row, 'r-', alpha=0.7, label='ADAS3 vs 融合结果')
# plt.title('实验2 vs 实验3: 逐行MSE对比 (越低越好)', fontsize=14)
# plt.xlabel('数据行索引', fontsize=12)
# plt.ylabel('MSE值', fontsize=12)
# plt.legend(fontsize=12)
# plt.grid(True, linestyle='--', alpha=0.7)

# # 添加水平平均线
# plt.axhline(y=np.mean(mse2_by_row), color='g', linestyle='--', 
#             label=f'ADAS2平均: {np.mean(mse2_by_row):.4f}')
# plt.axhline(y=np.mean(mse3_by_row), color='r', linestyle='--', 
#             label=f'融合平均: {np.mean(mse3_by_row):.4f}')
# plt.legend(fontsize=10)

# # RMSE对比
# plt.subplot(2, 1, 2)
# plt.plot(range(num_rows), rmse2_by_row, 'g-', alpha=0.7, label='ADAS3 vs ADAS2')
# plt.plot(range(num_rows), rmse3_by_row, 'r-', alpha=0.7, label='ADAS3 vs 融合结果')
# plt.title('实验2 vs 实验3: 逐行RMSE对比 (越低越好)', fontsize=14)
# plt.xlabel('数据行索引', fontsize=12)
# plt.ylabel('RMSE值', fontsize=12)
# plt.legend(fontsize=12)
# plt.grid(True, linestyle='--', alpha=0.7)

# # 添加水平平均线
# plt.axhline(y=np.mean(rmse2_by_row), color='g', linestyle='--', 
#             label=f'ADAS2平均: {np.mean(rmse2_by_row):.4f}')
# plt.axhline(y=np.mean(rmse3_by_row), color='r', linestyle='--', 
#             label=f'融合平均: {np.mean(rmse3_by_row):.4f}')
# plt.legend(fontsize=10)

# plt.tight_layout()
# comparison2_filename = os.path.join(output_dir, "exp2_vs_exp3_comparison.png")
# plt.savefig(comparison2_filename, dpi=300)
# print(f"图表已保存为 {comparison2_filename}")

# # 生成统计摘要
# print("\n生成统计摘要...")
# summary_filename = os.path.join(output_dir, "detailed_metrics_summary.csv")

# import csv
# with open(summary_filename, 'w', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow(['指标', 'ADAS3 vs ADAS1', 'ADAS3 vs ADAS2', 'ADAS3 vs 融合结果'])
    
#     # MSE统计
#     writer.writerow(['平均MSE', f"{np.mean(mse1_by_row):.6f}", f"{np.mean(mse2_by_row):.6f}", f"{np.mean(mse3_by_row):.6f}"])
#     writer.writerow(['MSE标准差', f"{np.std(mse1_by_row):.6f}", f"{np.std(mse2_by_row):.6f}", f"{np.std(mse3_by_row):.6f}"])
#     writer.writerow(['MSE最小值', f"{np.min(mse1_by_row):.6f}", f"{np.min(mse2_by_row):.6f}", f"{np.min(mse3_by_row):.6f}"])
#     writer.writerow(['MSE最大值', f"{np.max(mse1_by_row):.6f}", f"{np.max(mse2_by_row):.6f}", f"{np.max(mse3_by_row):.6f}"])
#     writer.writerow(['MSE中位数', f"{np.median(mse1_by_row):.6f}", f"{np.median(mse2_by_row):.6f}", f"{np.median(mse3_by_row):.6f}"])
    
#     # RMSE统计
#     writer.writerow([])
#     writer.writerow(['平均RMSE', f"{np.mean(rmse1_by_row):.6f}", f"{np.mean(rmse2_by_row):.6f}", f"{np.mean(rmse3_by_row):.6f}"])
#     writer.writerow(['RMSE标准差', f"{np.std(rmse1_by_row):.6f}", f"{np.std(rmse2_by_row):.6f}", f"{np.std(rmse3_by_row):.6f}"])
#     writer.writerow(['RMSE最小值', f"{np.min(rmse1_by_row):.6f}", f"{np.min(rmse2_by_row):.6f}", f"{np.min(rmse3_by_row):.6f}"])
#     writer.writerow(['RMSE最大值', f"{np.max(rmse1_by_row):.6f}", f"{np.max(rmse2_by_row):.6f}", f"{np.max(rmse3_by_row):.6f}"])
#     writer.writerow(['RMSE中位数', f"{np.median(rmse1_by_row):.6f}", f"{np.median(rmse2_by_row):.6f}", f"{np.median(rmse3_by_row):.6f}"])
    
#     # 计算融合结果与最佳单一ADAS的改进百分比
#     best_single_mse_mean = min(np.mean(mse1_by_row), np.mean(mse2_by_row))
#     best_single_rmse_mean = min(np.mean(rmse1_by_row), np.mean(rmse2_by_row))
    
#     mse_improvement = (best_single_mse_mean - np.mean(mse3_by_row)) / best_single_mse_mean * 100
#     rmse_improvement = (best_single_rmse_mean - np.mean(rmse3_by_row)) / best_single_rmse_mean * 100
    
#     writer.writerow([])
#     writer.writerow(['融合改进分析', '', '', ''])
#     writer.writerow(['相比最佳单一ADAS的MSE改进率', '', '', f"{mse_improvement:.2f}% {'(更好)' if mse_improvement > 0 else '(更差)'}"])
#     writer.writerow(['相比最佳单一ADAS的RMSE改进率', '', '', f"{rmse_improvement:.2f}% {'(更好)' if rmse_improvement > 0 else '(更差)'}"])

# print(f"详细统计摘要已保存到 {summary_filename}")

# # 评估结论
# print("\n评估结论:")
# if np.mean(mse3_by_row) < min(np.mean(mse1_by_row), np.mean(mse2_by_row)):
#     print("✓ 融合特征的平均MSE低于单一ADAS，表明融合效果好")
# else:
#     print("✗ 融合特征的平均MSE未能低于单一ADAS")

# if np.mean(rmse3_by_row) < min(np.mean(rmse1_by_row), np.mean(rmse2_by_row)):
#     print("✓ 融合特征的平均RMSE低于单一ADAS，表明融合效果好")
# else:
#     print("✗ 融合特征的平均RMSE未能低于单一ADAS")

# print(f"\nMSE改进: {mse_improvement:.2f}% {'(更好)' if mse_improvement > 0 else '(更差)'}")
# print(f"RMSE改进: {rmse_improvement:.2f}% {'(更好)' if rmse_improvement > 0 else '(更差)'}")

# print("\n评估完成!")


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import math
import os

print("=" * 60)
print("ADAS评估分析工具 - 优化版本")
print("=" * 60)

# 运行时输入文件路径
adas1_path = "/home/linux/ShenGang/feature_extraction_new_construct/results/new_log_result/extract/all_adas/minieye_gt=15_5162/features/train_features.npy"
adas2_path = "/home/linux/ShenGang/feature_extraction_new_construct/results/new_log_result/extract/all_adas/motivis_gt=15_6168/features/reduce5161_features.npy"
adas3_path = "/home/linux/ShenGang/feature_extraction_new_construct/results/new_log_result/extract/all_adas/jms2_gt=15_6168/features/reduce5161_features.npy"
fused_path = "/home/linux/ShenGang/feature_extraction_new_construct/results/new_log_result/fusion/minieye_motovis_fusion/fusion_test11/fusion_features/fused_feature_real.npy"

# 设置输出文件夹
output_dir = "/home/linux/ShenGang/feature_extraction_new_construct/fusion_evaluate_results"
os.makedirs(output_dir, exist_ok=True)

# 加载数据
try:
    adas1_features = np.load(adas1_path)
    print(f"成功加载ADAS1特征，形状为: {adas1_features.shape}")
except Exception as e:
    print(f"加载ADAS1特征时出错: {str(e)}")
    exit(1)

try:
    adas2_features = np.load(adas2_path)
    print(f"成功加载ADAS2特征，形状为: {adas2_features.shape}")
except Exception as e:
    print(f"加载ADAS2特征时出错: {str(e)}")
    exit(1)

try:
    adas3_features = np.load(adas3_path)
    print(f"成功加载ADAS3特征，形状为: {adas3_features.shape}")
except Exception as e:
    print(f"加载ADAS3特征时出错: {str(e)}")
    exit(1)

try:
    fused_features = np.load(fused_path)
    print(f"成功加载融合特征，形状为: {fused_features.shape}")
except Exception as e:
    print(f"加载融合特征时出错: {str(e)}")
    exit(1)

# 检查所有数据的行数是否一致
shapes = [arr.shape[0] for arr in [adas1_features, adas2_features, adas3_features, fused_features]]
if len(set(shapes)) > 1:
    print(f"警告: 数据行数不一致! 行数分别为: {shapes}")
    print("为确保比较的准确性，将截断至最小长度")
    min_rows = min(shapes)
    adas1_features = adas1_features[:min_rows]
    adas2_features = adas2_features[:min_rows]
    adas3_features = adas3_features[:min_rows]
    fused_features = fused_features[:min_rows]
    print(f"所有数据已截断至 {min_rows} 行")

num_rows = adas1_features.shape[0]
print(f"数据总行数: {num_rows}")

# 计算逐行MSE和RMSE
def calculate_row_metrics(target, prediction):
    """计算每一行的MSE和RMSE"""
    row_mse = []
    row_rmse = []
    
    for i in range(target.shape[0]):
        # 计算每一行的MSE
        mse = mean_squared_error(target[i], prediction[i])
        # 计算每一行的RMSE
        rmse = math.sqrt(mse)
        
        row_mse.append(mse)
        row_rmse.append(rmse)
    
    return np.array(row_mse), np.array(row_rmse)

# 计算每组实验的逐行指标
print("\n开始计算逐行指标...")

# 实验1: ADAS3 vs ADAS1
mse1_by_row, rmse1_by_row = calculate_row_metrics(adas3_features, adas1_features)
print(f"实验1 (ADAS3 vs ADAS1) - 平均MSE: {np.mean(mse1_by_row):.4f}, 平均RMSE: {np.mean(rmse1_by_row):.4f}")

# 实验2: ADAS3 vs ADAS2
mse2_by_row, rmse2_by_row = calculate_row_metrics(adas3_features, adas2_features)
print(f"实验2 (ADAS3 vs ADAS2) - 平均MSE: {np.mean(mse2_by_row):.4f}, 平均RMSE: {np.mean(rmse2_by_row):.4f}")

# 实验3: ADAS3 vs 融合结果
mse3_by_row, rmse3_by_row = calculate_row_metrics(adas3_features, fused_features)
print(f"实验3 (ADAS3 vs Fusion_Feature) - 平均MSE: {np.mean(mse3_by_row):.4f}, 平均RMSE: {np.mean(rmse3_by_row):.4f}")

# 数据减采样 - 为了图形更清晰
def downsample_data(data, factor=10):
    """按照指定的因子对数据进行降采样"""
    return data[::factor]

# 设置减采样因子 - 可以根据实际情况调整
downsample_factor = 20  # 每20个点取一个
x_indices = np.arange(0, num_rows, downsample_factor)
mse1_downsampled = mse1_by_row[::downsample_factor]
mse2_downsampled = mse2_by_row[::downsample_factor]
mse3_downsampled = mse3_by_row[::downsample_factor]
rmse1_downsampled = rmse1_by_row[::downsample_factor]
rmse2_downsampled = rmse2_by_row[::downsample_factor]
rmse3_downsampled = rmse3_by_row[::downsample_factor]

# 创建四张独立的图表
print("\n创建四张独立的图表...")

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 图1: 实验1 vs 实验3 - MSE对比
plt.figure(figsize=(12, 7))
step=2;
plt.plot(x_indices[::step], mse1_downsampled[::step], 'b-', linewidth=1.5, alpha=0.8, label='ADAS3 vs ADAS1')
plt.plot(x_indices[::step], mse3_downsampled[::step], 'r-', linewidth=1.5, alpha=0.8, label='ADAS3 vs Fusion_Feature')
plt.title('test1 vs test3: MSE', fontsize=14)
plt.xlabel('data_line', fontsize=12)
plt.ylabel('MSE', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.axhline(y=np.mean(mse1_by_row), color='b', linestyle='--', 
           label=f'ADAS1_average: {np.mean(mse1_by_row):.4f}')
plt.axhline(y=np.mean(mse3_by_row), color='r', linestyle='--', 
           label=f'Fusion_average: {np.mean(mse3_by_row):.4f}')
plt.legend(fontsize=10)
plt.tight_layout()
mse1_comparison_filename = os.path.join(output_dir, "exp1_vs_exp3_mse_comparison.png")
plt.savefig(mse1_comparison_filename, dpi=300)
print(f"图表1已保存为 {mse1_comparison_filename}")

# # 图2: 实验1 vs 实验3 - RMSE对比
# plt.figure(figsize=(12, 7))
# plt.plot(x_indices, rmse1_downsampled, 'g-', linewidth=1.5, alpha=0.8, label='ADAS3 vs ADAS1')
# plt.plot(x_indices, rmse3_downsampled, 'r-', linewidth=1.5, alpha=0.8, label='ADAS3 vs Fusion_Feature')
# plt.ylabel('RMSE', fontsize=12)
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.axhline(y=np.mean(rmse1_by_row), color='b', linestyle='--', 
#            label=f'ADAS1_average: {np.mean(rmse1_by_row):.4f}')
# plt.axhline(y=np.mean(rmse3_by_row), color='r', linestyle='--', 
#            label=f'Fusion_average: {np.mean(rmse3_by_row):.4f}')
# plt.legend(fontsize=10)
# plt.tight_layout()
# rmse1_comparison_filename = os.path.join(output_dir, "exp1_vs_exp3_rmse_comparison.png")
# plt.savefig(rmse1_comparison_filename, dpi=300)
# print(f"图表2已保存为 {rmse1_comparison_filename}")

# # 图3: 实验2 vs 实验3 - MSE对比
# plt.figure(figsize=(12, 7))
# plt.plot(x_indices, mse2_downsampled, 'g-', linewidth=1.5, alpha=0.8, label='ADAS3 vs ADAS2')
# plt.plot(x_indices, mse3_downsampled, 'r-', linewidth=1.5, alpha=0.8, label='ADAS3 vs Fusion_Feature')
# plt.title('test2 vs test3: MSE', fontsize=14)
# plt.xlabel('data_line', fontsize=12)
# plt.ylabel('MSE', fontsize=12)
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.axhline(y=np.mean(mse2_by_row), color='g', linestyle='--', 
#            label=f'ADAS2_average: {np.mean(mse2_by_row):.4f}')
# plt.axhline(y=np.mean(mse3_by_row), color='r', linestyle='--', 
#            label=f'Fusion_average: {np.mean(mse3_by_row):.4f}')
# plt.legend(fontsize=10)
# plt.tight_layout()
# mse2_comparison_filename = os.path.join(output_dir, "exp2_vs_exp3_mse_comparison.png")
# plt.savefig(mse2_comparison_filename, dpi=300)
# print(f"图表3已保存为 {mse2_comparison_filename}")

# # 图4: 实验2 vs 实验3 - RMSE对比
# plt.figure(figsize=(12, 7))
# plt.plot(x_indices, rmse2_downsampled, 'g-', linewidth=1.5, alpha=0.8, label='ADAS3 vs ADAS2')
# plt.plot(x_indices, rmse3_downsampled, 'r-', linewidth=1.5, alpha=0.8, label='ADAS3 vs Fusion_Feature')
# plt.title('test2 vs test3: RMSE', fontsize=14)
# plt.xlabel('data_line', fontsize=12)
# plt.ylabel('RMSE', fontsize=12)
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.axhline(y=np.mean(rmse2_by_row), color='g', linestyle='--', 
#            label=f'ADAS2_average: {np.mean(rmse2_by_row):.4f}')
# plt.axhline(y=np.mean(rmse3_by_row), color='r', linestyle='--', 
#            label=f'Fusion_average: {np.mean(rmse3_by_row):.4f}')
# plt.legend(fontsize=10)
# plt.tight_layout()
# rmse2_comparison_filename = os.path.join(output_dir, "exp2_vs_exp3_rmse_comparison.png")
# plt.savefig(rmse2_comparison_filename, dpi=300)
# print(f"图表4已保存为 {rmse2_comparison_filename}")

# 生成统计摘要
print("\n生成统计摘要...")
summary_filename = os.path.join(output_dir, "detailed_metrics_summary.csv")

import csv
with open(summary_filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['指标', 'ADAS3 vs ADAS1', 'ADAS3 vs ADAS2', 'ADAS3 vs Fusion_Feature'])
    
    # MSE统计
    writer.writerow(['平均MSE', f"{np.mean(mse1_by_row):.6f}", f"{np.mean(mse2_by_row):.6f}", f"{np.mean(mse3_by_row):.6f}"])
    writer.writerow(['MSE标准差', f"{np.std(mse1_by_row):.6f}", f"{np.std(mse2_by_row):.6f}", f"{np.std(mse3_by_row):.6f}"])
    writer.writerow(['MSE最小值', f"{np.min(mse1_by_row):.6f}", f"{np.min(mse2_by_row):.6f}", f"{np.min(mse3_by_row):.6f}"])
    writer.writerow(['MSE最大值', f"{np.max(mse1_by_row):.6f}", f"{np.max(mse2_by_row):.6f}", f"{np.max(mse3_by_row):.6f}"])
    
    # RMSE统计
    writer.writerow([])
    writer.writerow(['平均RMSE', f"{np.mean(rmse1_by_row):.6f}", f"{np.mean(rmse2_by_row):.6f}", f"{np.mean(rmse3_by_row):.6f}"])
    writer.writerow(['RMSE标准差', f"{np.std(rmse1_by_row):.6f}", f"{np.std(rmse2_by_row):.6f}", f"{np.std(rmse3_by_row):.6f}"])
    writer.writerow(['RMSE最小值', f"{np.min(rmse1_by_row):.6f}", f"{np.min(rmse2_by_row):.6f}", f"{np.min(rmse3_by_row):.6f}"])
    writer.writerow(['RMSE最大值', f"{np.max(rmse1_by_row):.6f}", f"{np.max(rmse2_by_row):.6f}", f"{np.max(rmse3_by_row):.6f}"])
    
    
    # 计算融合结果与最佳单一ADAS的改进百分比
    best_single_mse_mean = min(np.mean(mse1_by_row), np.mean(mse2_by_row))
    best_single_rmse_mean = min(np.mean(rmse1_by_row), np.mean(rmse2_by_row))
    
    mse_improvement = (best_single_mse_mean - np.mean(mse3_by_row)) / best_single_mse_mean * 100
    rmse_improvement = (best_single_rmse_mean - np.mean(rmse3_by_row)) / best_single_rmse_mean * 100
    
    writer.writerow([])
    writer.writerow(['融合改进分析', '', '', ''])
    writer.writerow(['相比最佳单一ADAS的MSE改进率', '', '', f"{mse_improvement:.2f}% {'(更好)' if mse_improvement > 0 else '(更差)'}"])
    writer.writerow(['相比最佳单一ADAS的RMSE改进率', '', '', f"{rmse_improvement:.2f}% {'(更好)' if rmse_improvement > 0 else '(更差)'}"])

print(f"详细统计摘要已保存到 {summary_filename}")

# 评估结论
print("\n评估结论:")
if np.mean(mse3_by_row) < min(np.mean(mse1_by_row), np.mean(mse2_by_row)):
    print("✓ 融合特征的平均MSE低于单一ADAS，表明融合效果好")
else:
    print("✗ 融合特征的平均MSE未能低于单一ADAS")

if np.mean(rmse3_by_row) < min(np.mean(rmse1_by_row), np.mean(rmse2_by_row)):
    print("✓ 融合特征的平均RMSE低于单一ADAS，表明融合效果好")
else:
    print("✗ 融合特征的平均RMSE未能低于单一ADAS")

print(f"\nMSE改进: {mse_improvement:.2f}% {'(更好)' if mse_improvement > 0 else '(更差)'}")
print(f"RMSE改进: {rmse_improvement:.2f}% {'(更好)' if rmse_improvement > 0 else '(更差)'}")

print("\n评估完成!")