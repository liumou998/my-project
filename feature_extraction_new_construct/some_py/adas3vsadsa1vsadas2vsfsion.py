# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from scipy.stats import pearsonr
# import math
# import os
# import pandas as pd
# from datetime import datetime

# print("=" * 60)
# print("ADAS评估分析工具 (增强版)")
# print("请依次输入各个特征数据的npy文件路径")
# print("=" * 60)
# # 运行时输入文件路径
# adas1_path = "/home/linux/ShenGang/feature_extraction_new_construct/fusion_test/test1_mix_maxeye_motivis_fusion/featuere/maxeye_feature/reduce6167_features.npy"
# adas2_path = "/home/linux/ShenGang/feature_extraction_new_construct/fusion_test/test1_mix_maxeye_motivis_fusion/featuere/motovis_feature/reduced_6167_feature.npy"
# adas3_path = "/home/linux/ShenGang/feature_extraction_new_construct/results/new_log_result/extract/all_adas/me630-6167.npy"
# fused_path = "/home/linux/ShenGang/feature_extraction_new_construct/fusion_test/test1_mix_maxeye_motivis_fusion/fusion_test11/fusion_features/jms2_test_reduced_feature.npy"

# # 设置地面真值(GT)
# gt = float(input("请输入地面真值(GT)数值: "))

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

# # 创建GT数组 (重复gt值以匹配特征数组长度)
# gt_array = np.full_like(adas3_features, gt)

# # 计算评估指标
# def calculate_metrics(target, prediction, name1, name2):
#     # 将数据转换为平坦数组以便计算
#     target_flat = target.flatten()
#     prediction_flat = prediction.flatten()
    
#     # 移除无效值（NaN或无穷大）
#     valid_indices = np.logical_and(
#         np.logical_and(~np.isnan(target_flat), ~np.isnan(prediction_flat)),
#         np.logical_and(~np.isinf(target_flat), ~np.isinf(prediction_flat))
#     )
#     valid_target = target_flat[valid_indices]
#     valid_prediction = prediction_flat[valid_indices]
    
#     if len(valid_target) == 0:
#         print(f"警告: {name1} vs {name2} 没有有效数据点!")
#         return {'name': f"{name1} vs {name2}", 'mse': np.nan, 'rmse': np.nan, 'mae': np.nan, 
#                 'mape': np.nan, 'bias': np.nan, 'r2': np.nan, 'r': np.nan, 
#                 'rstd': np.nan, 'maxae': np.nan}
    
#     # 计算MSE和RMSE
#     mse = mean_squared_error(valid_target, valid_prediction)
#     rmse = np.sqrt(mse)
    
#     # 计算MAE
#     mae = mean_absolute_error(valid_target, valid_prediction)
    
#     # 计算MAPE (排除目标为0的数据点)
#     non_zero_indices = valid_target != 0
#     if np.any(non_zero_indices):
#         mape = np.mean(np.abs((valid_target[non_zero_indices] - valid_prediction[non_zero_indices]) / valid_target[non_zero_indices])) * 100
#     else:
#         mape = np.nan
    
#     # 计算Bias (平均误差)
#     bias = np.mean(valid_prediction - valid_target)
    
#     # 计算R² (决定系数)
#     r2 = r2_score(valid_target, valid_prediction)
    
#     # 计算R (相关系数)
#     r, _ = pearsonr(valid_target, valid_prediction)
    
#     # 计算RSTD (残差标准差)
#     residuals = valid_prediction - valid_target
#     rstd = np.std(residuals)
    
#     # 计算MaxAE (最大绝对误差)
#     maxae = np.max(np.abs(valid_prediction - valid_target))
    
#     # 打印结果
#     print(f"比较 {name1} vs {name2}:")
#     print(f"MSE: {mse:.4f}")
#     print(f"RMSE: {rmse:.4f}")
#     print(f"MAE: {mae:.4f}")
#     print(f"MAPE: {mape:.2f}%")
#     print(f"Bias: {bias:.4f}")
#     print(f"R²: {r2:.4f}")
#     print(f"Pearson相关系数(r): {r:.4f}")
#     print(f"RSTD: {rstd:.4f}")
#     print(f"MaxAE: {maxae:.4f}")
#     print("-" * 40)
    
#     return {
#         'name': f"{name1} vs {name2}",
#         'mse': mse,
#         'rmse': rmse,
#         'mae': mae,
#         'mape': mape,
#         'bias': bias,
#         'r2': r2,
#         'r': r,
#         'rstd': rstd,
#         'maxae': maxae
#     }

# # 执行对比实验
# print("\n开始执行对比实验...")
# metrics_results = []

# # 0. ADAS3 vs GT (新增)
# metrics0 = calculate_metrics(gt_array, adas3_features, "GT", "ADAS3")
# metrics_results.append(metrics0)

# # 1. ADAS3 vs ADAS1
# metrics1 = calculate_metrics(adas3_features, adas1_features, "ADAS3", "ADAS1")
# metrics_results.append(metrics1)

# # 2. ADAS3 vs ADAS2
# metrics2 = calculate_metrics(adas3_features, adas2_features, "ADAS3", "ADAS2")
# metrics_results.append(metrics2)

# # 3. ADAS3 vs 融合结果
# metrics3 = calculate_metrics(adas3_features, fused_features, "ADAS3", "融合结果")
# metrics_results.append(metrics3)

# # 创建表格: 原始评估指标
# metrics_df = pd.DataFrame(metrics_results)
# metrics_df = metrics_df.set_index('name')

# # 设置输出图表文件名
# output_filename = input("\n请输入保存图表的文件名 (默认: adas_metrics_comparison.png): ")
# if not output_filename.strip():
#     output_filename = "adas_metrics_comparison.png"
# if not output_filename.endswith('.png'):
#     output_filename += '.png'

# # 绘制对比图
# print(f"\n绘制对比图并保存为 {output_filename}...")
# plt.figure(figsize=(18, 5))

# plt.subplot(1, 3, 1)
# names = [m['name'] for m in metrics_results]
# mse_values = [m['mse'] for m in metrics_results]
# bars = plt.bar(names, mse_values)
# plt.title('MSE对比 (越低越好)')
# plt.ylabel('MSE值')
# plt.xticks(rotation=15)
# # 添加数值标签
# for bar in bars:
#     height = bar.get_height()
#     plt.text(bar.get_x() + bar.get_width()/2., height,
#              f'{height:.4f}',
#              ha='center', va='bottom')

# plt.subplot(1, 3, 2)
# rmse_values = [m['rmse'] for m in metrics_results]
# bars = plt.bar(names, rmse_values)
# plt.title('RMSE对比 (越低越好)')
# plt.ylabel('RMSE值')
# plt.xticks(rotation=15)
# # 添加数值标签
# for bar in bars:
#     height = bar.get_height()
#     plt.text(bar.get_x() + bar.get_width()/2., height,
#              f'{height:.4f}',
#              ha='center', va='bottom')

# plt.subplot(1, 3, 3)
# r_values = [m['r'] for m in metrics_results]
# bars = plt.bar(names, r_values)
# plt.title('相关系数(r)对比 (越高越好)')
# plt.ylabel('相关系数')
# plt.axhline(y=0, color='r', linestyle='-')
# plt.xticks(rotation=15)
# # 添加数值标签
# for bar in bars:
#     height = bar.get_height()
#     plt.text(bar.get_x() + bar.get_width()/2., height,
#              f'{height:.4f}',
#              ha='center', va='bottom')

# plt.tight_layout()
# plt.savefig(output_filename, dpi=300)
# print(f"图表已保存为 {output_filename}")

# # 可选显示图表
# show_plot = input("是否显示图表? (y/n, 默认: y): ").strip().lower()
# if show_plot != 'n':
#     plt.show()

# # 创建TXT报告文件
# txt_file = input("\n请输入TXT报告文件名 (默认: adas_metrics_report.txt): ").strip()
# if not txt_file:
#     txt_file = "adas_metrics_report.txt"
# if not txt_file.endswith('.txt'):
#     txt_file += '.txt'

# with open(txt_file, 'w', encoding='utf-8') as f:
#     # 写入标题和基本信息
#     f.write("=" * 80 + "\n")
#     f.write("ADAS特征评估指标报告\n")
#     f.write("=" * 80 + "\n\n")
    
#     f.write("数据信息:\n")
#     f.write(f"- 地面真值(GT): {gt}\n")
#     f.write(f"- ADAS1特征文件: {adas1_path}\n")
#     f.write(f"- ADAS2特征文件: {adas2_path}\n")
#     f.write(f"- ADAS3特征文件: {adas3_path}\n")
#     f.write(f"- 融合特征文件: {fused_path}\n\n")
    
#     # 写入数据形状信息
#     f.write("数据维度信息:\n")
#     f.write(f"- ADAS1特征形状: {adas1_features.shape}\n")
#     f.write(f"- ADAS2特征形状: {adas2_features.shape}\n")
#     f.write(f"- ADAS3特征形状: {adas3_features.shape}\n")
#     f.write(f"- 融合特征形状: {fused_features.shape}\n\n")
    
#     # 写入表1: 原始评估指标
#     f.write("表1: 评估指标\n")
#     f.write("-" * 100 + "\n")
    
#     # 表头
#     metrics_header = "比较组".ljust(20)
#     metrics_header += "MSE".rjust(10)
#     metrics_header += "RMSE".rjust(10)
#     metrics_header += "MAE".rjust(10)
#     metrics_header += "MAPE(%)".rjust(10)
#     metrics_header += "Bias".rjust(10)
#     metrics_header += "R²".rjust(10)
#     metrics_header += "r".rjust(10)
#     metrics_header += "RSTD".rjust(10)
#     metrics_header += "MaxAE".rjust(10)
#     f.write(metrics_header + "\n")
#     f.write("-" * 100 + "\n")
    
#     # 写入每组的指标
#     for m in metrics_results:
#         row = m['name'].ljust(20)
#         row += f"{m['mse']:.4f}".rjust(10)
#         row += f"{m['rmse']:.4f}".rjust(10)
#         row += f"{m['mae']:.4f}".rjust(10)
#         row += f"{m['mape']:.2f}".rjust(10)
#         row += f"{m['bias']:.4f}".rjust(10)
#         row += f"{m['r2']:.4f}".rjust(10)
#         row += f"{m['r']:.4f}".rjust(10)
#         row += f"{m['rstd']:.4f}".rjust(10)
#         row += f"{m['maxae']:.4f}".rjust(10)
#         f.write(row + "\n")
    
#     f.write("-" * 100 + "\n\n")
    
#     # 指标说明
#     f.write("\n指标说明:\n")
#     f.write("- MSE: 均方误差，越小越好\n")
#     f.write("- RMSE: 均方根误差，越小越好\n")
#     f.write("- MAE: 平均绝对误差，越小越好\n")
#     f.write("- MAPE: 平均绝对百分比误差，越小越好\n")
#     f.write("- Bias: 偏差，绝对值越小越好\n")
#     f.write("- R²: 决定系数，越接近1越好\n")
#     f.write("- r: Pearson相关系数，越接近1越好\n")
#     f.write("- RSTD: 残差标准差，越小越好\n")
#     f.write("- MaxAE: 最大绝对误差，越小越好\n\n")
    
#     # 写入结论
#     f.write("\n结论:\n")
    
#     # 比较四个系统的表现
#     best_system = {}
#     metrics_to_compare = ['mse', 'rmse', 'mae', 'mape', 'bias', 'r2', 'r', 'rstd', 'maxae']
#     for metric in metrics_to_compare:
#         if metric in ['r2', 'r']:  # 这些指标越大越好
#             best_value = max([m[metric] for m in metrics_results])
#         else:  # 其他指标越小越好
#             if metric == 'bias':
#                 # 对于bias，取绝对值最小的
#                 best_value = min([abs(m[metric]) for m in metrics_results])
#                 # 找回原始值
#                 for m in metrics_results:
#                     if abs(m[metric]) == best_value:
#                         best_value = m[metric]
#                         break
#             else:
#                 best_value = min([m[metric] for m in metrics_results])
        
#         for m in metrics_results:
#             if metric in ['r2', 'r']:
#                 if m[metric] == best_value:
#                     best_system[metric] = m['name']
#             else:
#                 if metric == 'bias':
#                     if abs(m[metric]) == abs(best_value):
#                         best_system[metric] = m['name']
#                 else:
#                     if m[metric] == best_value:
#                         best_system[metric] = m['name']
    
#     # 计算每个系统在多少个指标上表现最佳
#     system_counts = {}
#     for system in best_system.values():
#         if system in system_counts:
#             system_counts[system] += 1
#         else:
#             system_counts[system] = 1
    
#     # 确定总体最佳系统
#     if system_counts:
#         best_overall = max(system_counts.items(), key=lambda x: x[1])[0]
#         f.write(f"根据评估指标，{best_overall}在大多数指标上表现最佳。\n")
#     else:
#         f.write("没有找到表现最佳的系统。\n")
    
#     for metric, system in best_system.items():
#         metric_display = metric.upper()
#         if metric == 'r2':
#             metric_display = 'R²'
#         f.write(f"- {metric_display}指标: {system}表现最佳\n")
    
#     # 写入最后的时间戳
#     f.write(f"\n\n报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# print(f"TXT报告已保存到 {txt_file}")

# # 导出完整结果到Excel文件
# save_results = input("\n是否导出结果数据到Excel文件? (y/n, 默认: y): ").strip().lower()
# if save_results != 'n':
#     excel_file = input("请输入Excel文件名 (默认: adas_comparison_results.xlsx): ").strip()
#     if not excel_file:
#         excel_file = "adas_comparison_results.xlsx"
#     if not excel_file.endswith('.xlsx'):
#         excel_file += '.xlsx'
    
#     # 创建Excel写入器
#     with pd.ExcelWriter(excel_file) as writer:
#         # 评估指标
#         metrics_df.to_excel(writer, sheet_name='评估指标')
    
#     print(f"结果已保存到 {excel_file}")

# # 为论文生成LaTeX表格代码
# generate_latex = input("\n是否生成LaTeX表格代码用于论文? (y/n, 默认: y): ").strip().lower()
# if generate_latex != 'n':
#     latex_file = input("请输入LaTeX文件名 (默认: adas_tables.tex): ").strip()
#     if not latex_file:
#         latex_file = "adas_tables.tex"
#     if not latex_file.endswith('.tex'):
#         latex_file += '.tex'
    
#     with open(latex_file, 'w') as f:
#         # 评估指标表格
#         f.write("% 表1: ADAS特征评估指标\n")
#         f.write("\\begin{table}[htbp]\n")
#         f.write("\\centering\n")
#         f.write("\\caption{ADAS特征评估指标比较}\n")
#         f.write("\\label{tab:adas_metrics}\n")
#         f.write("\\begin{tabular}{lrrrrrrrrr}\n")
#         f.write("\\toprule\n")
#         f.write("比较组 & MSE & RMSE & MAE & MAPE(\\%) & Bias & $R^2$ & $r$ & RSTD & MaxAE \\\\\n")
#         f.write("\\midrule\n")
        
#         for m in metrics_results:
#             name = m['name']
#             f.write(f"{name} & {m['mse']:.4f} & {m['rmse']:.4f} & {m['mae']:.4f} & {m['mape']:.2f} & {m['bias']:.4f} & {m['r2']:.4f} & {m['r']:.4f} & {m['rstd']:.4f} & {m['maxae']:.4f} \\\\\n")
        
#         f.write("\\bottomrule\n")
#         f.write("\\end{tabular}\n")
#         f.write("\\end{table}\n\n")

#     print(f"LaTeX表格代码已保存到 {latex_file}")

# print("\n评估分析完成!")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import math
import os
import pandas as pd
from datetime import datetime

print("=" * 60)
print("ADAS评估分析工具 (增强版)")
print("请依次输入各个特征数据的npy文件路径")
print("=" * 60)
# 运行时输入文件路径
adas1_path = "/home/linux/ShenGang/feature_extraction_new_construct/e2e_fusion_result/maxeye_JMS3_fusion/maxeye_reduce6167_features.npy"
adas2_path = "/home/linux/ShenGang/feature_extraction_new_construct/e2e_fusion_result/maxeye_motovis/motovis_train_features_6167.npy"
adas3_path = "/home/linux/ShenGang/feature_extraction_new_construct/e2e_fusion_result/maxeye_JMS3_fusion/JMS2-15.68.npy"
fused_path = "/home/linux/ShenGang/feature_extraction_new_construct/e2e_fusion_result/maxeye_motovis/fused_15.65.npy"
# <--- 新增: 设置地面真值(GT)
gt = float(input("请输入地面真值(GT)数值: "))

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

# <--- 新增: 创建GT数组 (重复gt值以匹配特征数组长度)
gt_array = np.full_like(adas3_features, gt)

# 计算评估指标
def calculate_metrics(target, prediction, name1, name2):
    # 将数据转换为平坦数组以便计算
    target_flat = target.flatten()
    prediction_flat = prediction.flatten()
    
    # 移除无效值（NaN或无穷大）
    valid_indices = np.logical_and(
        np.logical_and(~np.isnan(target_flat), ~np.isnan(prediction_flat)),
        np.logical_and(~np.isinf(target_flat), ~np.isinf(prediction_flat))
    )
    valid_target = target_flat[valid_indices]
    valid_prediction = prediction_flat[valid_indices]
    
    if len(valid_target) == 0:
        print(f"警告: {name1} vs {name2} 没有有效数据点!")
        return {'name': f"{name1} vs {name2}", 'mse': np.nan, 'rmse': np.nan, 'mae': np.nan, 
                'mape': np.nan, 'bias': np.nan, 'r2': np.nan, 'r': np.nan, 
                'rstd': np.nan, 'maxae': np.nan}
    
    # 计算MSE和RMSE
    mse = mean_squared_error(valid_target, valid_prediction)
    rmse = np.sqrt(mse)
    
    # 计算MAE
    mae = mean_absolute_error(valid_target, valid_prediction)
    
    # 计算MAPE (排除目标为0的数据点)
    non_zero_indices = valid_target != 0
    if np.any(non_zero_indices):
        mape = np.mean(np.abs((valid_target[non_zero_indices] - valid_prediction[non_zero_indices]) / valid_target[non_zero_indices])) * 100
    else:
        mape = np.nan
    
    # 计算Bias (平均误差)
    bias = np.mean(valid_prediction - valid_target)
    
    # 计算R² (决定系数)
    r2 = r2_score(valid_target, valid_prediction)
    
    # 计算R (相关系数)
    # <--- 修改: 检查以确保有足够的数据点进行相关性计算
    if len(valid_target) > 1:
        r, _ = pearsonr(valid_target, valid_prediction)
    else:
        r = np.nan
    
    # 计算RSTD (残差标准差)
    residuals = valid_prediction - valid_target
    rstd = np.std(residuals)
    
    # 计算MaxAE (最大绝对误差)
    maxae = np.max(np.abs(valid_prediction - valid_target))
    
    # 打印结果
    print(f"比较 {name1} vs {name2}:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"Bias: {bias:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"Pearson相关系数(r): {r:.4f}")
    print(f"RSTD: {rstd:.4f}")
    print(f"MaxAE: {maxae:.4f}")
    print("-" * 40)
    
    return {
        'name': f"{name1} vs {name2}",
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'bias': bias,
        'r2': r2,
        'r': r,
        'rstd': rstd,
        'maxae': maxae
    }

# 执行对比实验
print("\n开始执行对比实验...")
metrics_results = []

# <--- 新增: 0. ADAS3 vs GT 
# 注意这里的顺序，GT是目标(target)，ADAS3是预测(prediction)
metrics0 = calculate_metrics(gt_array, adas3_features, "GT", "ADAS3")
metrics_results.append(metrics0)

# 1. ADAS3 vs ADAS1
metrics1 = calculate_metrics(adas3_features, adas1_features, "ADAS3", "ADAS1")
metrics_results.append(metrics1)

# 2. ADAS3 vs ADAS2
metrics2 = calculate_metrics(adas3_features, adas2_features, "ADAS3", "ADAS2")
metrics_results.append(metrics2)

# 3. ADAS3 vs 融合结果
metrics3 = calculate_metrics(adas3_features, fused_features, "ADAS3", "融合结果")
metrics_results.append(metrics3)

# 创建表格: 原始评估指标
metrics_df = pd.DataFrame(metrics_results)
metrics_df = metrics_df.set_index('name')

# # 设置输出图表文件名
# output_filename = input("\n请输入保存图表的文件名 (默认: adas_metrics_comparison.png): ")
# if not output_filename.strip():
#     output_filename = "adas_metrics_comparison.png"
# if not output_filename.endswith('.png'):
#     output_filename += '.png'

# # 绘制对比图
# print(f"\n绘制对比图并保存为 {output_filename}...")
# plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
# <--- 修改: names的顺序现在是正确的
names = [m['name'] for m in metrics_results]
mse_values = [m['mse'] for m in metrics_results]
bars = plt.bar(names, mse_values)
plt.title('MSE对比 (越低越好)')
plt.ylabel('MSE值')
plt.xticks(rotation=15, ha="right") # <--- 修改: 调整旋转以获得更好的显示效果
# 添加数值标签
for bar in bars:
    height = bar.get_height()
    if not np.isnan(height):
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.4f}',
                 ha='center', va='bottom')

plt.subplot(1, 3, 2)
rmse_values = [m['rmse'] for m in metrics_results]
bars = plt.bar(names, rmse_values)
plt.title('RMSE对比 (越低越好)')
plt.ylabel('RMSE值')
plt.xticks(rotation=15, ha="right") # <--- 修改: 调整旋转
# 添加数值标签
for bar in bars:
    height = bar.get_height()
    if not np.isnan(height):
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.4f}',
                 ha='center', va='bottom')

plt.subplot(1, 3, 3)
r_values = [m['r'] for m in metrics_results]
bars = plt.bar(names, r_values)
plt.title('相关系数(r)对比 (越高越好)')
plt.ylabel('相关系数')
plt.axhline(y=0, color='r', linestyle='-')
plt.xticks(rotation=15, ha="right") # <--- 修改: 调整旋转
# 添加数值标签
for bar in bars:
    height = bar.get_height()
    if not np.isnan(height):
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.4f}',
                 ha='center', va='bottom')

plt.tight_layout()
# plt.savefig(output_filename, dpi=300)
# print(f"图表已保存为 {output_filename}")

# # 可选显示图表
# show_plot = input("是否显示图表? (y/n, 默认: y): ").strip().lower()
# if show_plot != 'n':
#     plt.show()

# 创建TXT报告文件
txt_file = input("\n请输入TXT报告文件名 (默认: adas_metrics_report.txt): ").strip()
if not txt_file:
    txt_file = "adas_metrics_report.txt"
if not txt_file.endswith('.txt'):
    txt_file += '.txt'

with open(txt_file, 'w', encoding='utf-8') as f:
    # 写入标题和基本信息
    f.write("=" * 80 + "\n")
    f.write("ADAS特征评估指标报告\n")
    f.write("=" * 80 + "\n\n")
    
    # <--- 修改: 在报告中加入GT值信息
    f.write("数据信息:\n")
    f.write(f"- 地面真值(GT): {gt}\n")
    f.write(f"- ADAS1特征文件: {adas1_path}\n")
    f.write(f"- ADAS2特征文件: {adas2_path}\n")
    f.write(f"- ADAS3特征文件: {adas3_path}\n")
    f.write(f"- 融合特征文件: {fused_path}\n\n")
    
    # 写入数据形状信息
    f.write("数据维度信息:\n")
    f.write(f"- ADAS1特征形状: {adas1_features.shape}\n")
    f.write(f"- ADAS2特征形状: {adas2_features.shape}\n")
    f.write(f"- ADAS3特征形状: {adas3_features.shape}\n")
    f.write(f"- 融合特征形状: {fused_features.shape}\n\n")
    
    # 写入表1: 原始评估指标
    f.write("表1: 评估指标\n")
    f.write("-" * 100 + "\n")
    
    # 表头
    metrics_header = "比较组".ljust(20)
    metrics_header += "MSE".rjust(10)
    metrics_header += "RMSE".rjust(10)
    metrics_header += "MAE".rjust(10)
    metrics_header += "MAPE(%)".rjust(10)
    metrics_header += "Bias".rjust(10)
    metrics_header += "R²".rjust(10)
    metrics_header += "r".rjust(10)
    metrics_header += "RSTD".rjust(10)
    metrics_header += "MaxAE".rjust(10)
    f.write(metrics_header + "\n")
    f.write("-" * 100 + "\n")
    
    # 写入每组的指标
    for m in metrics_results:
        row = m['name'].ljust(20)
        row += f"{m['mse']:.4f}".rjust(10)
        row += f"{m['rmse']:.4f}".rjust(10)
        row += f"{m['mae']:.4f}".rjust(10)
        row += f"{m['mape']:.2f}".rjust(10)
        row += f"{m['bias']:.4f}".rjust(10)
        row += f"{m['r2']:.4f}".rjust(10)
        row += f"{m['r']:.4f}".rjust(10)
        row += f"{m['rstd']:.4f}".rjust(10)
        row += f"{m['maxae']:.4f}".rjust(10)
        f.write(row + "\n")
    
    f.write("-" * 100 + "\n\n")
    
    # 指标说明
    f.write("\n指标说明:\n")
    f.write("- MSE: 均方误差，越小越好\n")
    f.write("- RMSE: 均方根误差，越小越好\n")
    f.write("- MAE: 平均绝对误差，越小越好\n")
    f.write("- MAPE: 平均绝对百分比误差，越小越好\n")
    f.write("- Bias: 偏差，绝对值越小越好\n")
    f.write("- R²: 决定系数，越接近1越好\n")
    f.write("- r: Pearson相关系数，越接近1越好\n")
    f.write("- RSTD: 残差标准差，越小越好\n")
    f.write("- MaxAE: 最大绝对误差，越小越好\n\n")
    
    # 写入结论
    f.write("\n结论:\n")
    
    # <--- 修改: 结论分析部分现在也会自动包含新的GT对比组
    best_system = {}
    metrics_to_compare = ['mse', 'rmse', 'mae', 'mape', 'bias', 'r2', 'r', 'rstd', 'maxae']
    # 筛选出非NaN的有效结果进行比较
    valid_metrics_results = [m for m in metrics_results if not np.isnan(m['mse'])]

    if valid_metrics_results:
        for metric in metrics_to_compare:
            if metric in ['r2', 'r']:  # 这些指标越大越好
                best_value = max([m[metric] for m in valid_metrics_results if not np.isnan(m[metric])])
            else:  # 其他指标越小越好
                if metric == 'bias':
                    # 对于bias，取绝对值最小的
                    abs_biases = [abs(m[metric]) for m in valid_metrics_results if not np.isnan(m[metric])]
                    if not abs_biases: continue
                    best_abs_value = min(abs_biases)
                    # 找回原始值
                    for m in valid_metrics_results:
                        if abs(m[metric]) == best_abs_value:
                            best_value = m[metric]
                            break
                else:
                    values = [m[metric] for m in valid_metrics_results if not np.isnan(m[metric])]
                    if not values: continue
                    best_value = min(values)
            
            for m in valid_metrics_results:
                if np.isnan(m[metric]): continue
                if metric in ['r2', 'r']:
                    if m[metric] == best_value:
                        best_system[metric] = m['name']
                else:
                    if metric == 'bias':
                        if abs(m[metric]) == abs(best_value):
                            best_system[metric] = m['name']
                    else:
                        if m[metric] == best_value:
                            best_system[metric] = m['name']
    
    # 计算每个系统在多少个指标上表现最佳
    system_counts = {}
    for system in best_system.values():
        system_counts[system] = system_counts.get(system, 0) + 1
    
    # 确定总体最佳系统
    if system_counts:
        best_overall = max(system_counts.items(), key=lambda x: x[1])[0]
        f.write(f"根据评估指标，{best_overall}在大多数指标上表现最佳。\n")
    else:
        f.write("没有找到表现最佳的系统。\n")
    
    for metric, system in best_system.items():
        metric_display = metric.upper()
        if metric == 'r2':
            metric_display = 'R²'
        f.write(f"- {metric_display}指标: {system}表现最佳\n")
    
    # 写入最后的时间戳
    f.write(f"\n\n报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

print(f"TXT报告已保存到 {txt_file}")

# # 导出完整结果到Excel文件
# save_results = input("\n是否导出结果数据到Excel文件? (y/n, 默认: y): ").strip().lower()
# if save_results != 'n':
#     excel_file = input("请输入Excel文件名 (默认: adas_comparison_results.xlsx): ").strip()
#     if not excel_file:
#         excel_file = "adas_comparison_results.xlsx"
#     if not excel_file.endswith('.xlsx'):
#         excel_file += '.xlsx'
    
#     # 创建Excel写入器
#     with pd.ExcelWriter(excel_file) as writer:
#         # 评估指标
#         metrics_df.to_excel(writer, sheet_name='评估指标')
    
#     print(f"结果已保存到 {excel_file}")

# # 为论文生成LaTeX表格代码
# generate_latex = input("\n是否生成LaTeX表格代码用于论文? (y/n, 默认: y): ").strip().lower()
# if generate_latex != 'n':
#     latex_file = input("请输入LaTeX文件名 (默认: adas_tables.tex): ").strip()
#     if not latex_file:
#         latex_file = "adas_tables.tex"
#     if not latex_file.endswith('.tex'):
#         latex_file += '.tex'
    
#     with open(latex_file, 'w', encoding='utf-8') as f:
#         # 评估指标表格
#         f.write("% 表1: ADAS特征评估指标\n")
#         f.write("\\begin{table}[htbp]\n")
#         f.write("\\centering\n")
#         f.write("\\caption{ADAS特征评估指标比较}\n")
#         f.write("\\label{tab:adas_metrics}\n")
#         # <--- 修改: 调整列数以适应更多指标
#         f.write("\\begin{tabular}{lrrrrrrrrr}\n")
#         f.write("\\toprule\n")
#         f.write("比较组 & MSE & RMSE & MAE & MAPE(\\%) & Bias & $R^2$ & $r$ & RSTD & MaxAE \\\\\n")
#         f.write("\\midrule\n")
        
#         for m in metrics_results:
#             name = m['name'].replace('_', '\\_') # <--- 修改: 自动转义LaTeX中的下划线
#             f.write(f"{name} & {m['mse']:.4f} & {m['rmse']:.4f} & {m['mae']:.4f} & {m['mape']:.2f} & {m['bias']:.4f} & {m['r2']:.4f} & {m['r']:.4f} & {m['rstd']:.4f} & {m['maxae']:.4f} \\\\\n")
        
#         f.write("\\bottomrule\n")
#         f.write("\\end{tabular}\n")
#         f.write("\\end{table}\n\n")

#     print(f"LaTeX表格代码已保存到 {latex_file}")

print("\n评估分析完成!")