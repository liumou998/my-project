# import matplotlib.pyplot as plt
# import numpy as np

# # --- 确保PDF字体质量 ---
# import matplotlib
# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42
# # -------------------------

# # 1. 数据准备 (保持不变)
# # ------------------------------------
# groups = ['High Precision', 'Low Precision', 'Mixed Precision']
# mse_adas1, mse_adas2, mse_fusion = [3.4587, 5.1101, 3.4587], [3.3667, 9.2994, 9.2994], [1.2726, 3.1524, 1.1341]
# rmse_adas1, rmse_adas2, rmse_fusion = [1.8597, 2.2606, 1.8597], [1.8349, 3.0495, 3.0494], [1.1282, 1.7754, 1.0649]
# professional_colors = {'ADAS1': '#4472C4', 'ADAS2': '#70AD47', 'Fusion': '#FFC000'}

# # 2. 生成对齐的设备信息文本 (保持不变)
# # ------------------------------------
# def get_aligned_device_info(config):
#     col_width_group, col_width_adas1, col_width_adas2 = 19, 9, 9
#     table_lines = [" " * col_width_group + "ADAS1".ljust(col_width_adas1) + "ADAS2".ljust(col_width_adas2)]
#     for pg_key, devices in config.items():
#         table_lines.append(f"{pg_key + ':':<{col_width_group}}{devices['ADAS1']:<{col_width_adas1}}{devices['ADAS2']:<{col_width_adas2}}")
#     return "\n".join(table_lines)

# DEVICE_CONFIG = {
#     'High Precision':  {'ADAS1': 'MAXEYE',  'ADAS2': 'JMS3'},
#     'Low Precision':   {'ADAS1': 'MINIEYE', 'ADAS2': 'MOTOVIS'},
#     'Mixed Precision': {'ADAS1': 'MAXEYE',  'ADAS2': 'MOTOVIS'}
# }
# device_info_text = get_aligned_device_info(DEVICE_CONFIG)

# # 3. 最终绘图函数 (终极版)
# # ------------------------------------
# def create_ultimate_publication_chart(y_axis_label, data_adas1, data_adas2, data_fusion, groups, colors, annotation_text):
#     """
#     生成顶部元素完美水平对齐、且左右框内行距分别经过独立微调的最终图表。
#     """
#     plt.rcParams['font.sans-serif'] = ['Arial']
#     fig, ax = plt.subplots(figsize=(10, 7))

#     # --- 绘制图表主体 (保持不变) ---
#     x = np.arange(len(groups))
#     bar_width = 0.20
#     for i, (system_type, data) in enumerate(zip(['ADAS1', 'ADAS2', 'Fusion'], [data_adas1, data_adas2, data_fusion])):
#         bar_positions = x + (i - 1) * bar_width
#         rects = ax.bar(bar_positions, data, bar_width, label=system_type, color=colors[system_type], edgecolor='black', linewidth=0.7)
#         for rect in rects:
#             height = rect.get_height()
#             ax.annotate(f'{height:.4f}', xy=(rect.get_x() + rect.get_width() / 2, height),
#                         xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=9)
#     ax.set_ylabel(y_axis_label, fontsize=14)
#     ax.set_xticks(x)
#     ax.set_xticklabels(groups, fontsize=12, fontweight='bold')
#     ax.tick_params(axis='y', labelsize=12)
#     max_val = max(max(data_adas1), max(data_adas2), max(data_fusion))
#     ax.set_ylim(0, max_val * 1.4)
#     ax.grid(axis='y', linestyle='--', alpha=0.7)
#     ax.set_axisbelow(True)

#     ## ======================= 关键步骤：实现顶部精确对齐及独立行距调整 ======================= ##

#     # 步骤 1: 绘制左侧图例框，并【缩减】其内部行距 (labelspacing)
#     legend = ax.legend(title="System Type", loc='upper left', bbox_to_anchor=(0.02, 0.97),
#                        fontsize=10, title_fontsize=11, frameon=True, facecolor='white',
#                        edgecolor='darkgrey', 
#                        labelspacing=0.5) # <--- 行距从 0.8 缩减到 0.5
#     legend.get_frame().set_linewidth(0.7)
#     legend.get_title().set_fontweight('bold')

#     # 步骤 2: 强制渲染以获取左侧图例框的精确位置
#     fig.canvas.draw()
#     legend_bbox = legend.get_window_extent().transformed(ax.transAxes.inverted())
#     legend_top_y = legend_bbox.y1

#     # 步骤 3: 使用获取的y坐标定位右侧文本框，并【增大】其内部行距 (linespacing)
#     ax.text(0.98, legend_top_y, annotation_text,
#             transform=ax.transAxes,
#             fontsize=10,
#             verticalalignment='top',
#             horizontalalignment='right',
#             bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='darkgrey', lw=0.7),
#             family='monospace',
#             linespacing=1.7) # <--- 行距从 1.5 增大到 1.7

#     ## ==================================================================================== ##

#     # --- 布局和保存 ---
#     fig.tight_layout(rect=[0.04, 0.05, 0.96, 0.98])
#     metric_name = y_axis_label.split('(')[1].split(')')[0]
#     pdf_filename = f"{metric_name}_publication_ultimate.pdf"
#     plt.savefig(pdf_filename, format='pdf')
#     plt.close(fig)
#     print(f"图表已成功保存为: {pdf_filename}")

# # 4. 调用函数生成最终的PDF文件
# # ------------------------------------
# print("正在生成最终的、经过独立行距微调的图表...")
# create_ultimate_publication_chart('Mean Square Error (MSE)', mse_adas1, mse_adas2, mse_fusion, groups, professional_colors, device_info_text)
# create_ultimate_publication_chart('Root Mean Square Error (RMSE)', rmse_adas1, rmse_adas2, rmse_fusion, groups, professional_colors, device_info_text)
# print("\n所有PDF文件已生成完毕！")




#time news

# import matplotlib.pyplot as plt
# import numpy as np

# # --- 确保PDF字体质量 ---
# import matplotlib
# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42
# # -------------------------

# # 1. 数据准备 (保持不变)
# # ------------------------------------
# groups = ['High Precision', 'Low Precision', 'Mixed Precision']
# mse_adas1, mse_adas2, mse_fusion = [3.4587, 5.1101, 3.4587], [3.3667, 9.2994, 9.2994], [1.2726, 3.1524, 1.1341]
# rmse_adas1, rmse_adas2, rmse_fusion = [1.8597, 2.2606, 1.8597], [1.8349, 3.0495, 3.0494], [1.1282, 1.7754, 1.0649]
# professional_colors = {'ADAS1': '#4472C4', 'ADAS2': '#70AD47', 'Fusion': '#FFC000'}

# # 2. 生成对齐的设备信息文本 (保持您代码中的原有逻辑不变)
# # ------------------------------------
# def get_aligned_device_info(config):
#     col_width_group, col_width_adas1, col_width_adas2 = 19, 9, 9
#     table_lines = [" " * col_width_group + "ADAS1".ljust(col_width_adas1) + "ADAS2".ljust(col_width_adas2)]
#     for pg_key, devices in config.items():
#         table_lines.append(f"{pg_key + ':':<{col_width_group}}{devices['ADAS1']:<{col_width_adas1}}{devices['ADAS2']:<{col_width_adas2}}")
#     return "\n".join(table_lines)

# DEVICE_CONFIG = {
#     'High Precision':  {'ADAS1': 'MAXEYE',  'ADAS2': 'JMS3'},
#     'Low Precision':   {'ADAS1': 'MINIEYE', 'ADAS2': 'MOTOVIS'},
#     'Mixed Precision': {'ADAS1': 'MAXEYE',  'ADAS2': 'MOTOVIS'}
# }
# device_info_text = get_aligned_device_info(DEVICE_CONFIG)

# # 3. 最终绘图函数 (终极版)
# # ------------------------------------
# def create_ultimate_publication_chart(y_axis_label, data_adas1, data_adas2, data_fusion, groups, colors, annotation_text):
#     """
#     生成顶部元素完美水平对齐、且左右框内行距分别经过独立微调的最终图表。
#     """
#     # --- 核心修改：将全局字体设置为 Times New Roman ---
#     plt.rcParams['font.family'] = 'serif'
#     plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
#     # ----------------------------------------------------
    
#     fig, ax = plt.subplots(figsize=(10, 7))

#     # --- 绘制图表主体 (保持不变) ---
#     x = np.arange(len(groups))
#     bar_width = 0.20
#     for i, (system_type, data) in enumerate(zip(['ADAS1', 'ADAS2', 'Fusion'], [data_adas1, data_adas2, data_fusion])):
#         bar_positions = x + (i - 1) * bar_width
#         rects = ax.bar(bar_positions, data, bar_width, label=system_type, color=colors[system_type], edgecolor='black', linewidth=0.7)
#         for rect in rects:
#             height = rect.get_height()
#             ax.annotate(f'{height:.4f}', xy=(rect.get_x() + rect.get_width() / 2, height),
#                         xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=9)
#     ax.set_ylabel(y_axis_label, fontsize=14)
#     ax.set_xticks(x)
#     ax.set_xticklabels(groups, fontsize=12, fontweight='bold')
#     ax.tick_params(axis='y', labelsize=12)
#     max_val = max(max(data_adas1), max(data_adas2), max(data_fusion))
#     ax.set_ylim(0, max_val * 1.4)
#     ax.grid(axis='y', linestyle='--', alpha=0.7)
#     ax.set_axisbelow(True)

#     ## ======================= 关键步骤：实现顶部精确对齐及独立行距调整 ======================= ##
#     legend = ax.legend(title="System Type", loc='upper left', bbox_to_anchor=(0.02, 0.97),
#                        fontsize=10, title_fontsize=11, frameon=True, facecolor='white',
#                        edgecolor='darkgrey', 
#                        labelspacing=0.5)
#     legend.get_frame().set_linewidth(0.7)
#     legend.get_title().set_fontweight('bold')

#     fig.canvas.draw()
#     legend_bbox = legend.get_window_extent().transformed(ax.transAxes.inverted())
#     legend_top_y = legend_bbox.y1

#     # --- 核心修改：移除右上角文本框的 family='monospace' 参数 ---
#     ax.text(0.98, legend_top_y, annotation_text,
#             transform=ax.transAxes,
#             fontsize=10,
#             verticalalignment='top',
#             horizontalalignment='right',
#             bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='darkgrey', lw=0.7),
#             # family='monospace', # <--- 已移除此行，使其继承全局的新罗马字体
#             linespacing=1.7)
#     # -------------------------------------------------------------

#     ## ==================================================================================== ##

#     # --- 布局和保存 ---
#     fig.tight_layout(rect=[0.04, 0.05, 0.96, 0.98])
#     metric_name = y_axis_label.split('(')[1].split(')')[0]
#     # 更新文件名以反映字体更改
#     pdf_filename = f"{metric_name}_publication_times.pdf"
#     plt.savefig(pdf_filename, format='pdf')
#     plt.close(fig)
#     print(f"图表已成功保存为: {pdf_filename}")

# # 4. 调用函数生成最终的PDF文件
# # ------------------------------------
# print("正在生成使用Times New Roman字体的图表...")
# create_ultimate_publication_chart('Mean Square Error (MSE)', mse_adas1, mse_adas2, mse_fusion, groups, professional_colors, device_info_text)
# create_ultimate_publication_chart('Root Mean Square Error (RMSE)', rmse_adas1, rmse_adas2, rmse_fusion, groups, professional_colors, device_info_text)
# print("\n所有PDF文件已生成完毕！")


import matplotlib.pyplot as plt
import numpy as np

# --- 确保PDF字体质量 ---
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# -------------------------

# 1. 数据准备 (保持不变)
# ------------------------------------
groups = ['High Precision', 'Low Precision', 'Mixed Precision']
mse_adas1, mse_adas2, mse_fusion = [3.4587, 5.1101, 3.4587], [3.3667, 9.2994, 9.2994], [1.2726, 3.1524, 1.1341]
rmse_adas1, rmse_adas2, rmse_fusion = [1.8597, 2.2606, 1.8597], [1.8349, 3.0495, 3.0494], [1.1282, 1.7754, 1.0649]
professional_colors = {'ADAS1': '#4472C4', 'ADAS2': '#70AD47', 'Fusion': '#FFC000'}

# 2. 优化的设备信息文本生成函数
# ------------------------------------
def get_formatted_device_info(config):
    """
    生成格式化的设备信息表格，适用于Times New Roman字体的精确对齐
    """
    # 使用更精确的列宽设置
    lines = []
    
    # 表头
    header = f"{'':>20}{'ADAS1':>12}{'ADAS2':>12}"
    lines.append(header)
    
    # 分隔线（可选，增加美观性）
    separator = f"{'':>20}{'─────':>12}{'─────':>12}"
    lines.append(separator)
    
    # 数据行
    for precision_type, devices in config.items():
        line = f"{precision_type + ':':>20}{devices['ADAS1']:>12}{devices['ADAS2']:>12}"
        lines.append(line)
    
    return "\n".join(lines)

DEVICE_CONFIG = {
    'High Precision':  {'ADAS1': 'MAXEYE',  'ADAS2': 'JMS3'},
    'Low Precision':   {'ADAS1': 'MINIEYE', 'ADAS2': 'MOTOVIS'},
    'Mixed Precision': {'ADAS1': 'MAXEYE',  'ADAS2': 'MOTOVIS'}
}

# 3. 进一步优化的绘图函数
# ------------------------------------
def create_optimized_publication_chart(y_axis_label, data_adas1, data_adas2, data_fusion, groups, colors, device_config):
    """
    生成具有完美对齐和美观布局的最终图表
    """
    # --- 设置Times New Roman字体 ---
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    # ----------------------------------------------------
    
    fig, ax = plt.subplots(figsize=(12, 8))  # 稍微增大图表尺寸以容纳更好的布局

    # --- 绘制图表主体 ---
    x = np.arange(len(groups))
    bar_width = 0.22
    
    for i, (system_type, data) in enumerate(zip(['ADAS1', 'ADAS2', 'Fusion'], [data_adas1, data_adas2, data_fusion])):
        bar_positions = x + (i - 1) * bar_width
        rects = ax.bar(bar_positions, data, bar_width, label=system_type, 
                      color=colors[system_type], edgecolor='black', linewidth=0.8,
                      alpha=0.85)  # 添加轻微透明度增加视觉层次
        
        # 添加数值标签
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}', 
                       xy=(rect.get_x() + rect.get_width() / 2, height),
                       xytext=(0, 5), textcoords="offset points", 
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 设置坐标轴
    ax.set_ylabel(y_axis_label, fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=13, fontweight='bold')
    ax.tick_params(axis='y', labelsize=13)
    ax.tick_params(axis='x', labelsize=13)
    
    # 设置Y轴范围
    max_val = max(max(data_adas1), max(data_adas2), max(data_fusion))
    ax.set_ylim(0, max_val * 1.35)
    
    # 移除网格线，保持图表简洁
    # ax.grid(axis='y', linestyle='--', alpha=0.6, color='gray')
    ax.set_axisbelow(True)

    # --- 创建图例 ---
    legend = ax.legend(title="System Type", loc='upper left', 
                      bbox_to_anchor=(0.02, 0.98),
                      fontsize=12, title_fontsize=13, 
                      frameon=True, facecolor='none',  # 无填充
                      edgecolor='darkgray',
                      labelspacing=0.6, handlelength=1.5)
    legend.get_frame().set_linewidth(0.8)  # 设置边框线宽与右侧框一致
    legend.get_frame().set_alpha(1.0)  # 设置为不透明但无背景
    legend.get_title().set_fontweight('bold')
    
    # 获取图例的位置信息以便与右侧表格对齐
    fig.canvas.draw()
    legend_bbox = legend.get_window_extent().transformed(ax.transAxes.inverted())
    legend_top_y = legend_bbox.y1

    # --- 创建设备信息表格 ---
    # 使用表格形式而非纯文本，确保完美对齐
    table_data = []
    headers = ['', 'ADAS1', 'ADAS2']
    
    for precision_type, devices in device_config.items():
        table_data.append([precision_type + ':', devices['ADAS1'], devices['ADAS2']])
    
    # 在右上角添加带边框的无边线表格，与左侧图例顶部对齐
    table = ax.table(cellText=table_data,
                    colLabels=headers,
                    cellLoc='center',
                    loc='upper right',
                    bbox=[0.62, legend_top_y-0.15, 0.36, 0.15])  # 左移并稍微增加宽度
    
    # 美化表格
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.1)  # 进一步减小垂直缩放以缩短行间距
    
    # 设置无边框表格样式
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('none')  # 改为无填充
        table[(0, i)].set_text_props(weight='bold')
        table[(0, i)].set_edgecolor('none')  # 去掉边框
    
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            table[(i, j)].set_facecolor('none')  # 改为无填充
            table[(i, j)].set_edgecolor('none')  # 去掉边框
            if j == 0:  # 第一列（precision type）加粗
                table[(i, j)].set_text_props(weight='bold')
    
    # 添加外边框以匹配左上角图例的样式，与图例顶部对齐
    table_bbox = [0.62, legend_top_y-0.15, 0.36, 0.15]  # 与table的bbox一致，左移并调整宽度
    rect = plt.Rectangle((table_bbox[0], table_bbox[1]), table_bbox[2], table_bbox[3], 
                        linewidth=0.8, edgecolor='darkgray', facecolor='none', 
                        transform=ax.transAxes, clip_on=False)
    ax.add_patch(rect)

    # --- 添加边框和阴影效果 ---
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_linewidth(0.8)
    ax.spines['right'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.spines['left'].set_linewidth(0.8)

    # --- 布局优化 ---
    fig.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
    
    # 保存文件
    metric_name = y_axis_label.split('(')[1].split(')')[0]
    pdf_filename = f"{metric_name}_final_optimized.pdf"
    plt.savefig(pdf_filename, format='pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"最终优化图表已保存为: {pdf_filename}")

# 4. 调用函数生成优化后的PDF文件
# ------------------------------------
print("正在生成最终优化版本的Times New Roman字体图表...")
create_optimized_publication_chart('Mean Square Error (MSE)', mse_adas1, mse_adas2, mse_fusion, groups, professional_colors, DEVICE_CONFIG)
create_optimized_publication_chart('Root Mean Square Error (RMSE)', rmse_adas1, rmse_adas2, rmse_fusion, groups, professional_colors, DEVICE_CONFIG)
print("\n所有最终优化的PDF文件已生成完毕！")

# 5. 额外提供一个保持原始文本框风格但优化对齐的版本
# ------------------------------------
def create_text_box_version(y_axis_label, data_adas1, data_adas2, data_fusion, groups, colors, device_config):
    """
    保持文本框风格但优化对齐的版本
    """
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    
    fig, ax = plt.subplots(figsize=(11, 7.5))

    # 绘制图表主体（与上面相同）
    x = np.arange(len(groups))
    bar_width = 0.22
    
    for i, (system_type, data) in enumerate(zip(['ADAS1', 'ADAS2', 'Fusion'], [data_adas1, data_adas2, data_fusion])):
        bar_positions = x + (i - 1) * bar_width
        rects = ax.bar(bar_positions, data, bar_width, label=system_type, 
                      color=colors[system_type], edgecolor='black', linewidth=0.8)
        
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}', 
                       xy=(rect.get_x() + rect.get_width() / 2, height),
                       xytext=(0, 5), textcoords="offset points", 
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel(y_axis_label, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=12, fontweight='bold')
    ax.tick_params(axis='y', labelsize=12)
    
    max_val = max(max(data_adas1), max(data_adas2), max(data_fusion))
    ax.set_ylim(0, max_val * 1.4)
    # 移除网格线
    # ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    # 图例 - 与右侧框保持一致的样式
    legend = ax.legend(title="System Type", loc='upper left', 
                      bbox_to_anchor=(0.02, 0.97),
                      fontsize=11, title_fontsize=12, 
                      frameon=True, facecolor='none',  # 无填充
                      edgecolor='darkgray',
                      labelspacing=0.5)
    legend.get_frame().set_linewidth(0.8)  # 设置边框线宽与右侧框一致
    legend.get_title().set_fontweight('bold')

    # 优化的文本框 - 使用Times New Roman字体并精确格式化，改为无填充
    device_text = """                    ADAS1        ADAS2
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
High Precision:        MAXEYE        JMS3
 Low Precision:       MINIEYE      MOTOVIS
Mixed Precision:       MAXEYE      MOTOVIS"""

    ax.text(0.98, 0.97, device_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='none',  # 改为无填充
                     edgecolor='darkgray', alpha=1.0),
            # 移除fontfamily参数，让它继承全局的Times New Roman设置
            linespacing=1.6)

    fig.tight_layout(rect=[0.04, 0.05, 0.96, 0.95])
    
    metric_name = y_axis_label.split('(')[1].split(')')[0]
    pdf_filename = f"{metric_name}_textbox_final.pdf"
    plt.savefig(pdf_filename, format='pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"文本框最终版本已保存为: {pdf_filename}")

# 生成文本框优化版本
print("\n正在生成文本框最终版本...")
create_text_box_version('Mean Square Error (MSE)', mse_adas1, mse_adas2, mse_fusion, groups, professional_colors, DEVICE_CONFIG)
create_text_box_version('Root Mean Square Error (RMSE)', rmse_adas1, rmse_adas2, rmse_fusion, groups, professional_colors, DEVICE_CONFIG)