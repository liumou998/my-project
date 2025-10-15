import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import math
import os
import pandas as pd

print("=" * 60)
print("ADAS评估分析工具 (增强版)")
print("请依次输入各个特征数据的npy文件路径")
print("=" * 60)

# 运行时输入文件路径
adas1_path = "/home/linux/ShenGang/feature_extraction_new_construct/results/new_log_result/extract/adas1_feature_gt=15/features/jms2_test_reduced_feature.npy"
adas2_path = "/home/linux/ShenGang/feature_extraction_new_construct/results/new_log_result/extract/adas2_feature_gt=15/features/jms2_test_reduced_feature.npy"
adas3_path = "/home/linux/ShenGang/feature_extraction_new_construct/results/new_log_result/extract/jms2_adas3_gt=15/features/train_features.npy"
fused_path = "/home/linux/ShenGang/feature_extraction_new_construct/results/new_log_result/fusion/fusion_test29/fusion_features/fused_feature_real.npy"


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

# 计算增强版评估指标
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
    r, _ = pearsonr(valid_target, valid_prediction)
    
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

# 执行三组对比实验
print("\n开始执行对比实验...")
metrics_results = []

# 1. ADAS3 vs ADAS1
metrics1 = calculate_metrics(adas3_features, adas1_features, "ADAS3", "ADAS1")
metrics_results.append(metrics1)

# 2. ADAS3 vs ADAS2
metrics2 = calculate_metrics(adas3_features, adas2_features, "ADAS3", "ADAS2")
metrics_results.append(metrics2)

# 3. ADAS3 vs 融合结果
metrics3 = calculate_metrics(adas3_features, fused_features, "ADAS3", "融合结果")
metrics_results.append(metrics3)

# 创建表格1: 原始评估指标
metrics_df = pd.DataFrame(metrics_results)
metrics_df = metrics_df.set_index('name')

# 创建表格2: 融合特征相对于最佳单一ADAS的改进百分比
improvement_data = {}
metrics_to_compare = ['mse', 'rmse', 'mae', 'mape', 'bias', 'r2', 'r', 'rstd', 'maxae']

# 确定最佳单一ADAS的指标值
best_single_metrics = {}
for metric in metrics_to_compare:
    # 对于MSE, RMSE, MAE, MAPE, Bias, RSTD, MaxAE越小越好
    if metric in ['mse', 'rmse', 'mae', 'mape', 'rstd', 'maxae']:
        if np.isnan(metrics1[metric]) and np.isnan(metrics2[metric]):
            best_single_metrics[metric] = np.nan
        elif np.isnan(metrics1[metric]):
            best_single_metrics[metric] = metrics2[metric]
        elif np.isnan(metrics2[metric]):
            best_single_metrics[metric] = metrics1[metric]
        else:
            best_single_metrics[metric] = min(metrics1[metric], metrics2[metric])
    # 对于bias，取绝对值最小的
    elif metric == 'bias':
        if np.isnan(metrics1[metric]) and np.isnan(metrics2[metric]):
            best_single_metrics[metric] = np.nan
        elif np.isnan(metrics1[metric]):
            best_single_metrics[metric] = metrics2[metric]
        elif np.isnan(metrics2[metric]):
            best_single_metrics[metric] = metrics1[metric]
        else:
            if abs(metrics1[metric]) < abs(metrics2[metric]):
                best_single_metrics[metric] = metrics1[metric]
            else:
                best_single_metrics[metric] = metrics2[metric]
    # 对于R²和R，越大越好
    else:  # r2, r
        if np.isnan(metrics1[metric]) and np.isnan(metrics2[metric]):
            best_single_metrics[metric] = np.nan
        elif np.isnan(metrics1[metric]):
            best_single_metrics[metric] = metrics2[metric]
        elif np.isnan(metrics2[metric]):
            best_single_metrics[metric] = metrics1[metric]
        else:
            best_single_metrics[metric] = max(metrics1[metric], metrics2[metric])

# 计算改进百分比
for metric in metrics_to_compare:
    if metric in ['mse', 'rmse', 'mae', 'mape', 'rstd', 'maxae']:
        # 这些指标越小越好，所以改进 = (最佳单一值 - 融合值) / 最佳单一值
        if np.isnan(best_single_metrics[metric]) or np.isnan(metrics3[metric]) or best_single_metrics[metric] == 0:
            improvement_data[metric] = np.nan
        else:
            improvement_data[metric] = (best_single_metrics[metric] - metrics3[metric]) / best_single_metrics[metric] * 100
    elif metric == 'bias':
        # 对于bias，改进 = (|最佳单一值| - |融合值|) / |最佳单一值|
        if np.isnan(best_single_metrics[metric]) or np.isnan(metrics3[metric]) or abs(best_single_metrics[metric]) == 0:
            improvement_data[metric] = np.nan
        else:
            improvement_data[metric] = (abs(best_single_metrics[metric]) - abs(metrics3[metric])) / abs(best_single_metrics[metric]) * 100
    else:  # r2, r
        # 这些指标越大越好，所以改进 = (融合值 - 最佳单一值) / |最佳单一值|
        if np.isnan(best_single_metrics[metric]) or np.isnan(metrics3[metric]) or best_single_metrics[metric] == 0:
            improvement_data[metric] = np.nan
        else:
            improvement_data[metric] = (metrics3[metric] - best_single_metrics[metric]) / abs(best_single_metrics[metric]) * 100

# 打印改进百分比
print("\n融合特征相对于最佳单一ADAS的改进百分比:")
for metric, value in improvement_data.items():
    if not np.isnan(value):
        better_or_worse = "更好" if value > 0 else "更差"
        print(f"{metric.upper()}: {value:.2f}% ({better_or_worse})")
    else:
        print(f"{metric.upper()}: N/A")

# 设置输出图表文件名
output_filename = input("请输入保存图表的文件名 (默认: metrics_comparison.png): ")
if not output_filename.strip():
    output_filename = "metrics_comparison.png"
if not output_filename.endswith('.png'):
    output_filename += '.png'

# 绘制对比图 (只选三个主要指标进行可视化)
print(f"\n绘制对比图并保存为 {output_filename}...")
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
names = [m['name'] for m in metrics_results]
mse_values = [m['mse'] for m in metrics_results]
plt.bar(names, mse_values)
plt.title('MSE对比 (越低越好)')
plt.ylabel('MSE值')
plt.xticks(rotation=15)

plt.subplot(1, 3, 2)
rmse_values = [m['rmse'] for m in metrics_results]
plt.bar(names, rmse_values)
plt.title('RMSE对比 (越低越好)')
plt.ylabel('RMSE值')
plt.xticks(rotation=15)

plt.subplot(1, 3, 3)
r_values = [m['r'] for m in metrics_results]
plt.bar(names, r_values)
plt.title('相关系数(r)对比 (越高越好)')
plt.ylabel('相关系数')
plt.axhline(y=0, color='r', linestyle='-')
plt.xticks(rotation=15)

plt.tight_layout()
plt.savefig(output_filename, dpi=300)
print(f"图表已保存为 {output_filename}")

# 可选显示图表
show_plot = input("是否显示图表? (y/n, 默认: y): ").strip().lower()
if show_plot != 'n':
    plt.show()

# 导出完整结果到Excel文件
save_results = input("\n是否导出结果数据到Excel文件? (y/n, 默认: y): ").strip().lower()
if save_results != 'n':
    excel_file = input("请输入Excel文件名 (默认: adas_comparison_results.xlsx): ").strip()
    if not excel_file:
        excel_file = "adas_comparison_results.xlsx"
    if not excel_file.endswith('.xlsx'):
        excel_file += '.xlsx'
    
    # 创建Excel写入器
    with pd.ExcelWriter(excel_file) as writer:
        # 表1: 原始评估指标
        metrics_df.to_excel(writer, sheet_name='原始评估指标')
        
        # 表2: 改进百分比
        improvement_df = pd.DataFrame([improvement_data], index=['融合特征相对最佳单一ADAS改进(%)'])
        improvement_df.to_excel(writer, sheet_name='改进百分比')
    
    print(f"结果已保存到 {excel_file}")

# 为论文生成LaTeX表格代码
generate_latex = input("\n是否生成LaTeX表格代码用于论文? (y/n, 默认: y): ").strip().lower()
if generate_latex != 'n':
    latex_file = input("请输入LaTeX文件名 (默认: adas_tables.tex): ").strip()
    if not latex_file:
        latex_file = "adas_tables.tex"
    if not latex_file.endswith('.tex'):
        latex_file += '.tex'
    
    with open(latex_file, 'w') as f:
        # 表1: 原始评估指标表格
        f.write("% 表1: ADAS特征评估指标表\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{ADAS特征评估指标}\n")
        f.write("\\label{tab:adas_metrics}\n")
        f.write("\\begin{tabular}{lrrrrrrrrr}\n")
        f.write("\\toprule\n")
        f.write("比较组 & MSE & RMSE & MAE & MAPE(\\%) & Bias & $R^2$ & $r$ & RSTD & MaxAE \\\\\n")
        f.write("\\midrule\n")
        
        for m in metrics_results:
            name = m['name']
            f.write(f"{name} & {m['mse']:.4f} & {m['rmse']:.4f} & {m['mae']:.4f} & {m['mape']:.2f} & {m['bias']:.4f} & {m['r2']:.4f} & {m['r']:.4f} & {m['rstd']:.4f} & {m['maxae']:.4f} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n\n")
        
        # 表2: 改进百分比表格
        f.write("% 表2: ADAS融合特征改进百分比表\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{融合特征相对于最佳单一ADAS的性能改进百分比}\n") 
        f.write("\\label{tab:adas_improvement}\n")
        f.write("\\begin{tabular}{lrrrrrrrrr}\n")
        f.write("\\toprule\n")
        f.write(" & MSE & RMSE & MAE & MAPE & Bias & $R^2$ & $r$ & RSTD & MaxAE \\\\\n")
        f.write("\\midrule\n")
        
        improvement_str = "改进百分比(\\%)"
        f.write(f"{improvement_str}")
        for metric in ['mse', 'rmse', 'mae', 'mape', 'bias', 'r2', 'r', 'rstd', 'maxae']:
            value = improvement_data[metric]
            if np.isnan(value):
                f.write(" & N/A")
            else:
                if value > 0:
                    f.write(f" & \\textbf{{{value:.2f}}}")  # 加粗显示正改进
                else:
                    f.write(f" & {value:.2f}")
        f.write(" \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"LaTeX表格代码已保存到 {latex_file}")

print("\n评估完成!")
