import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import pandas as pd
from datetime import datetime

print("=" * 60)
print("ADAS评估分析工具 (简化版)")
print("=" * 60)

# 设置地面真值(GT)
gt = 15 # 这里设置为您的真实值

# 运行时输入文件路径

adas1_path = "/home/linux/ShenGang/feature_extraction_new_construct/e2e_fusion_result/maxeye_JMS3_fusion/maxeye_reduce6167_features.npy"
adas2_path = "/home/linux/ShenGang/feature_extraction_new_construct/results/new_log_result/extract/all_adas/motivis_gt=15_6168/features/train_features_6167.npy"
fused_path = "/home/linux/ShenGang/feature_extraction_new_construct/e2e_self_supervised_results/fused_feature.npy"

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
    fused_features = np.load(fused_path)
    print(f"成功加载融合特征，形状为: {fused_features.shape}")
except Exception as e:
    print(f"加载融合特征时出错: {str(e)}")
    exit(1)

# 检查所有数据的行数是否一致
shapes = [arr.shape[0] for arr in [adas1_features, adas2_features, fused_features]]
if len(set(shapes)) > 1:
    print(f"警告: 数据行数不一致! 行数分别为: {shapes}")
    print("为确保比较的准确性，将截断至最小长度")
    min_rows = min(shapes)
    adas1_features = adas1_features[:min_rows]
    adas2_features = adas2_features[:min_rows]
    fused_features = fused_features[:min_rows]
    
    print(f"所有数据已截断至 {min_rows} 行")

# 创建GT数组 (重复gt值以匹配特征数组长度)
gt_array = np.full_like(adas1_features, gt)

# 计算评估指标
def calculate_metrics(target, prediction, name):
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
        print(f"警告: {name} 没有有效数据点!")
        return {'name': name, 'mse': np.nan, 'rmse': np.nan, 'mae': np.nan, 
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
    print(f"{name} vs GT:")
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
        'name': name,
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
print(f"\n开始执行与GT值({gt})的对比实验...")
metrics_results = []

# 1. ADAS1 vs GT
metrics1 = calculate_metrics(gt_array, adas1_features, "ADAS1")
metrics_results.append(metrics1)

# 2. ADAS2 vs GT
metrics2 = calculate_metrics(gt_array, adas2_features, "ADAS2")
metrics_results.append(metrics2)

# 3. 融合结果 vs GT
metrics3 = calculate_metrics(gt_array, fused_features, "融合结果")
metrics_results.append(metrics3)

# 创建表格: 评估指标
metrics_df = pd.DataFrame(metrics_results)
metrics_df = metrics_df.set_index('name')


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
    
    f.write("数据信息:\n")
    f.write(f"- 地面真值(GT): {gt}\n")
    f.write(f"- ADAS1特征文件: {adas1_path}\n")
    f.write(f"- ADAS2特征文件: {adas2_path}\n")
    f.write(f"- 融合特征文件: {fused_path}\n\n")
    
    # 写入数据形状信息
    f.write("数据维度信息:\n")
    f.write(f"- ADAS1特征形状: {adas1_features.shape}\n")
    f.write(f"- ADAS2特征形状: {adas2_features.shape}\n")
    f.write(f"- 融合特征形状: {fused_features.shape}\n\n")
    
    # 写入表: 评估指标（与GT的对比）
    f.write("表1: 与地面真值(GT)比较的评估指标\n")
    f.write("-" * 100 + "\n")
    
    # 表头
    metrics_header = "系统".ljust(15)
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
        row = m['name'].ljust(15)
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
    
    # 比较三个系统的表现
    best_system = {}
    for metric in ['mse', 'rmse', 'mae', 'mape', 'bias', 'r2', 'r', 'rstd', 'maxae']:
        if metric in ['r2', 'r']:  # 这些指标越大越好
            best_value = max([m[metric] for m in metrics_results])
        else:  # 其他指标越小越好
            if metric == 'bias':
                # 对于bias，取绝对值最小的
                best_value = min([abs(m[metric]) for m in metrics_results])
                # 找回原始值
                for m in metrics_results:
                    if abs(m[metric]) == best_value:
                        best_value = m[metric]
                        break
            else:
                best_value = min([m[metric] for m in metrics_results])
        
        for m in metrics_results:
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
        if system in system_counts:
            system_counts[system] += 1
        else:
            system_counts[system] = 1
    
    # 确定总体最佳系统
    best_overall = max(system_counts.items(), key=lambda x: x[1])[0]
    
    # 写入结论
    f.write(f"根据评估指标，{best_overall}在大多数指标上表现最佳。\n")
    for metric, system in best_system.items():
        metric_display = metric.upper()
        if metric == 'r2':
            metric_display = 'R²'
        f.write(f"- {metric_display}指标: {system}表现最佳\n")
    
    # 写入最后的时间戳
    f.write(f"\n\n报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

print(f"TXT报告已保存到 {txt_file}")


print("\n评估分析完成!")
