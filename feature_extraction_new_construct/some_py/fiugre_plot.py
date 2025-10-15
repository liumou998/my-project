# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.ticker import PercentFormatter, MaxNLocator
# import pandas as pd
# from datetime import datetime

# # 设备名称配置 (根据你提供的最新表格更新)
# # 顺序: High, Low, Mixed
# DEVICE_CONFIG = {
#     'High Precision':  {'ADAS1': 'MAXEYE',  'ADAS2': 'JMS3',    'SUT': 'JMS2 (Reference)', 'FUSION_DESC': 'Fusion of MAXEYE & JMS3'},
#     'Low Precision':   {'ADAS1': 'MINIEYE', 'ADAS2': 'MOTOVIS', 'SUT': 'JMS2 (Reference)', 'FUSION_DESC': 'Fusion of MINIEYE & MOTOVIS'},
#     'Mixed Precision': {'ADAS1': 'MAXEYE',  'ADAS2': 'MOTOVIS', 'SUT': 'JMS2 (Reference)', 'FUSION_DESC': 'Fusion of MAXEYE & MOTOVIS'}
# }
# # 我为每个组添加了一个 'FUSION_DESC' 来描述融合的构成，如果不需要这么具体，可以简化。

# def get_user_input():
#     """Get user input data for all precision groups"""
#     print("Please input ADAS metrics data:")
#     metrics_data_list = []
    
#     precision_groups = ['High Precision', 'Low Precision', 'Mixed Precision']
    
#     improvement_df = pd.DataFrame(columns=['MSE', 'RMSE', 'MAE', 'MAPE', 'BIAS'],
#                                  index=precision_groups)

#     for group_name in precision_groups:
#         print(f"\n--- {group_name} Group Data ---")
#         devices = DEVICE_CONFIG[group_name] 
#         print(f"   (ADAS1: {devices['ADAS1']}, ADAS2: {devices['ADAS2']}, SUT/Reference: {devices['SUT']})")
#         print(f"   (Fusion is based on: {devices.get('FUSION_DESC', 'ADAS1 & ADAS2')})") # 提示Fusion的构成

#         print("\nMSE values (vs SUT/Reference):")
#         mse_adas1 = float(input(f"{group_name} - {devices['ADAS1']} (as ADAS1) MSE: "))
#         mse_adas2 = float(input(f"{group_name} - {devices['ADAS2']} (as ADAS2) MSE: "))
#         mse_fusion = float(input(f"{group_name} - Fusion MSE: "))

#         print("\nRMSE values (vs SUT/Reference):")
#         rmse_adas1 = float(input(f"{group_name} - {devices['ADAS1']} (as ADAS1) RMSE: "))
#         rmse_adas2 = float(input(f"{group_name} - {devices['ADAS2']} (as ADAS2) RMSE: "))
#         rmse_fusion = float(input(f"{group_name} - Fusion RMSE: "))

#         metrics_data_list.append({'group': group_name, 'system_type': 'ADAS1', 'system_name': devices['ADAS1'], 'MSE': mse_adas1, 'RMSE': rmse_adas1})
#         metrics_data_list.append({'group': group_name, 'system_type': 'ADAS2', 'system_name': devices['ADAS2'], 'MSE': mse_adas2, 'RMSE': rmse_adas2})
#         # For Fusion, system_name can remain 'Fusion' for plotting, details in text box
#         metrics_data_list.append({'group': group_name, 'system_type': 'Fusion', 'system_name': 'Fusion', 'MSE': mse_fusion, 'RMSE': rmse_fusion})


#         print(f"\n--- {group_name} Average Improvement Rate (%) for Fusion ---")
#         imp_mse = float(input("MSE improvement rate: "))
#         imp_rmse = float(input("RMSE improvement rate: "))
#         imp_mae = float(input("MAE improvement rate: "))
#         imp_mape = float(input("MAPE improvement rate: "))
#         imp_bias = float(input("BIAS improvement rate: "))
#         improvement_df.loc[group_name] = [imp_mse, imp_rmse, imp_mae, imp_mape, imp_bias]

#     metrics_plotting_df = pd.DataFrame(metrics_data_list)
#     metrics_plotting_df['group'] = pd.Categorical(metrics_plotting_df['group'], categories=precision_groups, ordered=True)
#     metrics_plotting_df = metrics_plotting_df.sort_values('group')
    
#     return metrics_plotting_df, improvement_df

# def create_mse_rmse_chart(metrics_df, metric_to_plot, y_label, chart_title_base, output_file_base):
#     """Create MSE or RMSE comparison chart with device names as text and adaptive y-axis."""
#     plt.rcParams['font.sans-serif'] = ['Arial']
#     plt.rcParams['font.size'] = 10
#     plt.rcParams['axes.unicode_minus'] = False
#     fig, ax = plt.subplots(figsize=(12, 7))

#     precision_groups_ordered = ['High Precision', 'Low Precision', 'Mixed Precision']
    
#     num_precision_groups = len(precision_groups_ordered)
#     bar_width = 0.25
#     index = np.arange(num_precision_groups)
#     system_types_in_plot = ['ADAS1', 'ADAS2', 'Fusion'] # These will be the legend labels
    
#     color_map = {'ADAS1': '#4472C4', 'ADAS2': '#70AD47', 'Fusion': '#FFC000'}

#     all_values_for_ylim = []

#     for i, system_type_label in enumerate(system_types_in_plot): # Use system_type_label for legend
#         values = []
#         for pg_group in precision_groups_ordered:
#             # Fetch data using 'system_type' which is 'ADAS1', 'ADAS2', or 'Fusion' in the DataFrame
#             row = metrics_df[(metrics_df['group'] == pg_group) & (metrics_df['system_type'] == system_type_label)]
#             values.append(row[metric_to_plot].iloc[0] if not row.empty else 0)
        
#         all_values_for_ylim.extend(values)

#         bar_positions = index + (i - (len(system_types_in_plot) - 1) / 2) * bar_width
#         # Use system_type_label for the bar's label (for the legend)
#         rects = ax.bar(bar_positions, values, bar_width, label=system_type_label, color=color_map.get(system_type_label, 'gray'))

#         for rect_idx, rect in enumerate(rects):
#             height = rect.get_height()
#             ax.text(rect.get_x() + rect.get_width() / 2., height, f'{height:.2f}',
#                     ha='center', va='bottom', fontsize=8, rotation=0,
#                     bbox=dict(facecolor='white', alpha=0.5, pad=1, edgecolor='none'))

#     # --- MODIFIED Right-side text box ---
#     device_info_text_right = "System Configurations (vs SUT/Reference):\n" # Changed title slightly
#     for pg in precision_groups_ordered:
#         cfg = DEVICE_CONFIG[pg]
#         device_info_text_right += (f"\n{pg} Group:\n"
                                   
#                                    f"  ADAS1: {cfg['ADAS1']}\n"
#                                    f"  ADAS2: {cfg['ADAS2']}\n"
#                                    f"  Fusion: {cfg.get('FUSION_DESC', 'Fusion Algorithm Output')}\n" # Get fusion desc
#                                    f"  SUT/Ref: {cfg['SUT']}") # Added SUT here for clarity per group
    
#     fig.text(0.99, 0.95, device_info_text_right.strip(), transform=ax.transAxes,
#              fontsize=7, verticalalignment='top', horizontalalignment='right', # Adjusted x to 0.99
#              bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.3))

#     # --- Left-side text box (remains the same if you still want it, or can be removed/modified) ---
#     annotation_lines_left = []
#     for group_key in precision_groups_ordered: 
#         cfg = DEVICE_CONFIG[group_key]
#         annotation_lines_left.append(f"{group_key} Group:\n  (ADAS1: {cfg['ADAS1']}\n   ADAS2: {cfg['ADAS2']})") # Simplified for left
    
#     left_annotation_block = "\n\n".join(annotation_lines_left)
    
#     # ax.text(-0.25, 0.5, left_annotation_block, transform=ax.transAxes, fontsize=8,
#     #          va='center', ha='left',
#     #          bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', alpha=0.9))
#     # You can comment out the above ax.text for the left annotation if the right one is now sufficient.

#     ax.set_ylabel(y_label, fontsize=12)
#     ax.set_title(chart_title_base, fontsize=14, pad=20)
#     ax.set_xticks(index)
#     ax.set_xticklabels(precision_groups_ordered)
#     ax.legend(title="System Type", loc='upper left', bbox_to_anchor=(0, 0.95), fontsize=9) # Legend will now show ADAS1, ADAS2, Fusion
#     ax.grid(axis='y', linestyle='--', alpha=0.7)

#     if all_values_for_ylim:
#         min_val = 0
#         max_val = max(all_values_for_ylim) if all_values_for_ylim else 1
#         ax.set_ylim([min_val, max_val * 1.4])
#         ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune='upper'))

#     output_pdf_file = f"{output_file_base}.pdf"
#     # Adjusted rect for potentially wider right text box
#     plt.tight_layout(rect=[0.05, 0, 0.80, 1]) # Reduced right more, increased left slightly
#     plt.savefig(output_pdf_file, dpi=300)
#     print(f"Chart saved to: {output_file_base}")
#     plt.close(fig)


# def create_improvement_chart(improvement_df, output_file_base):
#     """Create improvement rate comparison chart for all precision groups."""
#     plt.rcParams['font.sans-serif'] = ['Arial']
#     plt.rcParams['font.size'] = 10
#     plt.rcParams['axes.unicode_minus'] = False
#     fig, ax = plt.subplots(figsize=(12, 7))

#     metrics_to_plot = improvement_df.columns.tolist()
#     precision_groups_ordered = ['High Precision', 'Low Precision', 'Mixed Precision']
#     improvement_df = improvement_df.reindex(precision_groups_ordered)

#     num_metrics = len(metrics_to_plot)
#     num_precision_groups = len(precision_groups_ordered)
#     bar_width = 0.22 
#     group_colors = ['#5B9BD5', '#ED7D31', '#A5A5A5']
#     all_imp_values = []

#     for i, metric_name in enumerate(metrics_to_plot):
#         values_for_metric = improvement_df[metric_name].values.astype(float)
#         all_imp_values.extend(values_for_metric)
#         base_pos_metric = i * (num_precision_groups * bar_width + 0.3) 

#         for j, precision_group_name in enumerate(precision_groups_ordered):
#             value = values_for_metric[j]
#             bar_pos = base_pos_metric + j * bar_width
#             bar = ax.bar(bar_pos, value, bar_width, color=group_colors[j], 
#                          label=precision_group_name if i == 0 else "")
#             ax.text(bar_pos, value + (0.02 * value if value > 0 else -1), f'{value:.1f}%', 
#                     ha='center', va='bottom' if value >=0 else 'top', fontsize=8)

#     ax.set_ylabel('Improvement Rate (%)', fontsize=12)
#     ax.set_title('Average Improvement Rate Comparison (Fusion vs Others)', fontsize=14, pad=20)
#     metric_tick_positions = [i * (num_precision_groups * bar_width + 0.3) + (num_precision_groups -1) * bar_width / 2
#                              for i in range(num_metrics)]
#     ax.set_xticks(metric_tick_positions)
#     ax.set_xticklabels(metrics_to_plot)
    
#     handles, labels = ax.get_legend_handles_labels()
#     unique_labels = []
#     unique_handles = []
#     for handle, label in zip(handles, labels):
#         if label not in unique_labels:
#             unique_labels.append(label)
#             unique_handles.append(handle)
#     sorted_legend_items = sorted(zip(unique_labels, unique_handles), key=lambda x: precision_groups_ordered.index(x[0]))
#     sorted_labels = [item[0] for item in sorted_legend_items]
#     sorted_handles = [item[1] for item in sorted_legend_items]
#     ax.legend(sorted_handles, sorted_labels, title="Precision Group", loc='upper right', fontsize=9)
    
#     ax.grid(axis='y', linestyle='--', alpha=0.7)
#     ax.yaxis.set_major_formatter(PercentFormatter())
#     ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')

#     if all_imp_values:
#         min_imp_val = min(all_imp_values) if all_imp_values else 0
#         max_imp_val = max(all_imp_values) if all_imp_values else 10
#         padding_factor = 0.15 
#         lower_bound = min_imp_val - abs(min_imp_val) * padding_factor if min_imp_val < 0 else 0
#         upper_bound = max_imp_val + abs(max_imp_val) * padding_factor if max_imp_val > 0 else 10 
#         ax.set_ylim([min(lower_bound, -5), max(upper_bound, 10)]) 
#         ax.yaxis.set_major_locator(MaxNLocator(nbins=7, prune='both'))

#     output_pdf_file = f"{output_file_base}.pdf"
#     plt.tight_layout()
#     plt.savefig(output_pdf_file, dpi=300, bbox_inches='tight')
#     print(f"Chart saved to: {output_file_base}")
#     plt.close(fig)

# def main():
#     print("ADAS Metrics Visualization Tool")
#     print("==============================")
#     metrics_plotting_df, improvement_df = get_user_input()

#     create_mse_rmse_chart(metrics_plotting_df,
#                           metric_to_plot='MSE',
#                           y_label='Mean Square Error (MSE)',
#                           chart_title_base='MSE Comparison (vs SUT/Reference) by Precision Group',
#                           output_file_base='mse_comparison_final')

#     create_mse_rmse_chart(metrics_plotting_df,
#                           metric_to_plot='RMSE',
#                           y_label='Root Mean Square Error (RMSE)',
#                           chart_title_base='RMSE Comparison (vs SUT/Reference) by Precision Group',
#                           output_file_base='rmse_comparison_final')

#     create_improvement_chart(improvement_df,
#                              output_file_base='improvement_comparison_final')

#     print("\nAll visualizations complete!")
#     print("Generated PDF files:")
#     print("1. mse_comparison_final.pdf")
#     print("2. rmse_comparison_final.pdf")
#     print("3. improvement_comparison_final.pdf")

# if __name__ == "__main__":
#     main()


# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.ticker import PercentFormatter, MaxNLocator
# import pandas as pd
# from datetime import datetime

# # 设备名称配置 (根据你提供的最新表格更新)
# # 顺序: High, Low, Mixed
# DEVICE_CONFIG = {
#     'High Precision':  {'ADAS1': 'MAXEYE',  'ADAS2': 'JMS3',    'SUT': 'JMS2 '},
#     'Low Precision':   {'ADAS1': 'MINIEYE', 'ADAS2': 'MOTOVIS', 'SUT': 'JMS2 '},
#     'Mixed Precision': {'ADAS1': 'MAXEYE',  'ADAS2': 'MOTOVIS', 'SUT': 'JMS2 '}
# }

# def get_user_input():
#     """Get user input data for all precision groups for MSE and RMSE"""
#     print("Please input ADAS metrics data (MSE and RMSE only):")
#     metrics_data_list = []
    
#     # 精度组顺序: High, Low, Mixed
#     precision_groups = ['High Precision', 'Low Precision', 'Mixed Precision']
    
#     for group_name in precision_groups:
#         print(f"\n--- {group_name} Group Data ---")
#         devices = DEVICE_CONFIG[group_name] 
#         print(f"   (ADAS1: {devices['ADAS1']}, ADAS2: {devices['ADAS2']}, SUT/Reference: {devices['SUT']})")

#         print("\nMSE values (vs SUT/Reference):")
#         mse_adas1 = float(input(f"{group_name} - {devices['ADAS1']} (as ADAS1) MSE: "))
#         mse_adas2 = float(input(f"{group_name} - {devices['ADAS2']} (as ADAS2) MSE: "))
#         mse_fusion = float(input(f"{group_name} - Fusion MSE: "))

#         print("\nRMSE values (vs SUT/Reference):")
#         rmse_adas1 = float(input(f"{group_name} - {devices['ADAS1']} (as ADAS1) RMSE: "))
#         rmse_adas2 = float(input(f"{group_name} - {devices['ADAS2']} (as ADAS2) RMSE: "))
#         rmse_fusion = float(input(f"{group_name} - Fusion RMSE: "))

#         metrics_data_list.append({'group': group_name, 'system_type': 'ADAS1', 'system_name': devices['ADAS1'], 'MSE': mse_adas1, 'RMSE': rmse_adas1})
#         metrics_data_list.append({'group': group_name, 'system_type': 'ADAS2', 'system_name': devices['ADAS2'], 'MSE': mse_adas2, 'RMSE': rmse_adas2})
#         metrics_data_list.append({'group': group_name, 'system_type': 'Fusion', 'system_name': 'Fusion', 'MSE': mse_fusion, 'RMSE': rmse_fusion})

#     metrics_plotting_df = pd.DataFrame(metrics_data_list)
#     metrics_plotting_df['group'] = pd.Categorical(metrics_plotting_df['group'], categories=precision_groups, ordered=True)
#     metrics_plotting_df = metrics_plotting_df.sort_values('group')
    
#     return metrics_plotting_df

# def create_mse_rmse_chart(metrics_df, metric_to_plot, y_label, chart_title_base, output_file_base):
#     """Create MSE or RMSE comparison chart with device names as text and adaptive y-axis."""
#     plt.rcParams['font.sans-serif'] = ['Arial'] # Or your preferred sans-serif font
#     plt.rcParams['font.size'] = 10
#     plt.rcParams['axes.unicode_minus'] = False
#     # Ensure monospace font is available or specify one, e.g., 'Courier New', 'DejaVu Sans Mono'
#     # plt.rcParams['font.monospace'] = ['Courier New'] 
    
#     fig, ax = plt.subplots(figsize=(12, 7.5)) # Slightly increased height for more spacing

#     precision_groups_ordered = ['High Precision', 'Low Precision', 'Mixed Precision']
    
#     num_precision_groups = len(precision_groups_ordered)
#     bar_width = 0.25
#     index = np.arange(num_precision_groups)
#     system_types_in_plot = ['ADAS1', 'ADAS2', 'Fusion'] 
    
#     color_map = {'ADAS1': '#4472C4', 'ADAS2': '#70AD47', 'Fusion': '#FFC000'} # Fusion color

#     all_values_for_ylim = []

#     for i, system_type_label in enumerate(system_types_in_plot):
#         values = []
#         for pg_group in precision_groups_ordered:
#             row = metrics_df[(metrics_df['group'] == pg_group) & (metrics_df['system_type'] == system_type_label)]
#             values.append(row[metric_to_plot].iloc[0] if not row.empty else 0)
        
#         all_values_for_ylim.extend(values)

#         bar_positions = index + (i - (len(system_types_in_plot) - 1) / 2) * bar_width
#         rects = ax.bar(bar_positions, values, bar_width, label=system_type_label, color=color_map.get(system_type_label, 'gray'))

#         for rect_idx, rect in enumerate(rects):
#             height = rect.get_height()
#             label_text = f'{height:.2f}'
#             if height is not None and not (isinstance(height, float) and np.isnan(height)):
#                  ax.text(rect.get_x() + rect.get_width() / 2., height + 0.05 * max(all_values_for_ylim if all_values_for_ylim else [1]), # Dynamic offset for label
#                         label_text,
#                         ha='center', va='bottom', fontsize=7.5, rotation=0, # Slightly smaller font for bar labels
#                         bbox=dict(facecolor='white', alpha=0.3, pad=0.5, edgecolor='none'))

#     # --- Right-side text box (Table-like format) ---
#     # Adjust column widths based on your actual longest device names
#     col_width_group = 16  # Example: "Mixed Precision"
#     col_width_adas1 = 10  # Example: "MAXEYE"
#     col_width_adas2 = 10  # Example: "MOTOVIS"
#     col_width_sut = 16    # Example: "JMS2 (Reference)"

#     table_lines = []
#     # Header row for the table
#     header_line = " " * col_width_group + \
#                   "ADAS1".ljust(col_width_adas1) + \
#                   "ADAS2".ljust(col_width_adas2) + \
#                   "SUT".ljust(col_width_sut)
#     table_lines.append(header_line)
#     table_lines.append("-" * (len(header_line) - col_width_group)) # Separator line, adjusted length

#     # Data rows
#     for pg_label, pg_key in [("High", "High Precision"), 
#                              ("Low", "Low Precision"), 
#                              ("Mixed", "Mixed Precision")]:
#         cfg = DEVICE_CONFIG[pg_key]
#         # Using a shorter display label for the group column if needed for alignment
#         group_display_label = f"{pg_label} Precision:".ljust(col_width_group) 
#         adas1_display = cfg['ADAS1'].ljust(col_width_adas1)
#         adas2_display = cfg['ADAS2'].ljust(col_width_adas2)
#         sut_display = cfg['SUT'].ljust(col_width_sut)
#         table_lines.append(f"{group_display_label}{adas1_display}{adas2_display}{sut_display}")

#     device_info_text_right_table = "\n".join(table_lines)
    
#     # Position the text box: x, y are the coordinates of the top-left corner of the text box
#     # relative to the axes. Adjust x to move left/right.
#     # Ensure there's enough space by adjusting `plt.tight_layout` or `fig.subplots_adjust`
#     fig.text(0.65, 0.96, device_info_text_right_table, # x moved left to 0.65, y slightly up
#              transform=ax.transAxes, 
#              fontsize=6.5, # Smaller font for the table
#              verticalalignment='top',
#              horizontalalignment='left', 
#              bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='grey', lw=0.5, alpha=0.95), # White background, grey border
#              family='monospace', # Crucial for alignment
#              linespacing=1.3) # Adjust line spacing if needed

#     ax.set_ylabel(y_label, fontsize=12)
#     ax.set_title(chart_title_base, fontsize=14, pad=25) # Increased pad for title
#     ax.set_xticks(index)
#     ax.set_xticklabels(precision_groups_ordered)
    
#     legend = ax.legend(title="System Type", 
#                        loc='upper left', 
#                        bbox_to_anchor=(0.01, 0.96), # y slightly up to align with top of right text
#                        fontsize=7, 
#                        frameon=True, 
#                        facecolor='white', 
#                        edgecolor='grey', 
#                        labelspacing=0.7, 
#                        title_fontsize=8)
#     legend.get_frame().set_linewidth(0.5)
#     legend.get_frame().set_alpha(0.95)


#     ax.grid(axis='y', linestyle='--', alpha=0.7)

#     # Adjust Y-axis limit to ensure bar labels are visible
#     if all_values_for_ylim:
#         min_val = 0
#         max_val_data = max(all_values_for_ylim) if all_values_for_ylim else 1
        
#         # Increase top margin more significantly to accommodate labels on high bars
#         y_axis_top_limit = max_val_data * 1.45 # Increased from 1.35 to 1.45
        
#         if max_val_data == 0 : # If all data is zero
#             y_axis_top_limit = 1 # Set a default top limit
#         elif y_axis_top_limit < max_val_data + 0.2 * max_val_data and max_val_data > 0: # Ensure some absolute padding for small values
#              y_axis_top_limit = max_val_data + 0.3 * (max_val_data if max_val_data > 0.1 else 1)


#         ax.set_ylim([min_val, y_axis_top_limit])
#         ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune='upper'))

#     output_pdf_file = f"{output_file_base}.pdf"
    
#     # Adjust layout. `rect` defines the box in Figure coords that the Axes will fill.
#     # [left, bottom, right, top]
#     # To move the main plot left to make space on the right, decrease 'right'.
#     plt.tight_layout(rect=[0.06, 0.05, 0.98, 0.92]) # top slightly down for title, left slightly for y-label
#                                                     # right increased to allow more space for the plot itself before fig.text
    
#     # Or, for more manual control:
#     # fig.subplots_adjust(left=0.1, right=0.7, top=0.9, bottom=0.1, hspace=0.2, wspace=0.2)


#     plt.savefig(output_pdf_file, dpi=300, bbox_inches='tight')
#     print(f"Chart saved to: {output_file_base}")
#     plt.close(fig)


# def main():
#     print("ADAS Metrics Visualization Tool (MSE/RMSE only)")
#     print("===============================================")
#     metrics_plotting_df = get_user_input() 

#     create_mse_rmse_chart(metrics_plotting_df,
#                           metric_to_plot='MSE',
#                           y_label='Mean Square Error (MSE)',
#                           chart_title_base='MSE Comparison (vs SUT/Reference) by Precision Group',
#                           output_file_base='mse_comparison_final')

#     create_mse_rmse_chart(metrics_plotting_df,
#                           metric_to_plot='RMSE',
#                           y_label='Root Mean Square Error (RMSE)',
#                           chart_title_base='RMSE Comparison (vs SUT/Reference) by Precision Group',
#                           output_file_base='rmse_comparison_final')

#     print("\nVisualizations for MSE and RMSE complete!")
#     print("Generated PDF files:")
#     print("1. mse_comparison_final.pdf")
#     print("2. rmse_comparison_final.pdf")

# if __name__ == "__main__":
#     main()

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.ticker import MaxNLocator 
# import pandas as pd
# from datetime import datetime
# import matplotlib # Import matplotlib itself

# # --- IMPROVE PDF FONT QUALITY ---
# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42
# # --- END ---

# # 设备名称配置
# DEVICE_CONFIG = {
#     'High Precision':  {'ADAS1': 'MAXEYE',  'ADAS2': 'JMS3',    'SUT': 'JMS2'}, # SUT key still used for input phase
#     'Low Precision':   {'ADAS1': 'MINIEYE', 'ADAS2': 'MOTOVIS', 'SUT': 'JMS2'},
#     'Mixed Precision': {'ADAS1': 'MAXEYE',  'ADAS2': 'MOTOVIS', 'SUT': 'JMS2'}
# }

# def get_user_input():
#     """Get user input data for all precision groups for MSE and RMSE"""
#     print("Please input ADAS metrics data (MSE and RMSE only):")
#     metrics_data_list = []
#     precision_groups = ['High Precision', 'Low Precision', 'Mixed Precision']
    
#     for group_name in precision_groups:
#         print(f"\n--- {group_name} Group Data ---")
#         devices = DEVICE_CONFIG[group_name] 
#         # Display SUT device name during input if needed, but it won't be in the top-right table
#         print(f"   (ADAS1: {devices['ADAS1']}, ADAS2: {devices['ADAS2']}, SUT Device for context: {devices['SUT']})")

#         print("\nMSE values (vs GT):") 
#         while True:
#             try:
#                 mse_adas1 = float(input(f"{group_name} - {devices['ADAS1']} (as ADAS1) MSE: "))
#                 mse_adas2 = float(input(f"{group_name} - {devices['ADAS2']} (as ADAS2) MSE: "))
#                 mse_fusion = float(input(f"{group_name} - Fusion MSE: "))
#                 break
#             except ValueError:
#                 print("Invalid input. Please enter numeric values.")
        
#         print("\nRMSE values (vs GT):") 
#         while True:
#             try:
#                 rmse_adas1 = float(input(f"{group_name} - {devices['ADAS1']} (as ADAS1) RMSE: "))
#                 rmse_adas2 = float(input(f"{group_name} - {devices['ADAS2']} (as ADAS2) RMSE: "))
#                 rmse_fusion = float(input(f"{group_name} - Fusion RMSE: "))
#                 break
#             except ValueError:
#                 print("Invalid input. Please enter numeric values.")

#         metrics_data_list.append({'group': group_name, 'system_type': 'ADAS1', 'system_name': devices['ADAS1'], 'MSE': mse_adas1, 'RMSE': rmse_adas1})
#         metrics_data_list.append({'group': group_name, 'system_type': 'ADAS2', 'system_name': devices['ADAS2'], 'MSE': mse_adas2, 'RMSE': rmse_adas2})
#         metrics_data_list.append({'group': group_name, 'system_type': 'Fusion', 'system_name': 'Fusion', 'MSE': mse_fusion, 'RMSE': rmse_fusion})

#     metrics_plotting_df = pd.DataFrame(metrics_data_list)
#     metrics_plotting_df['group'] = pd.Categorical(metrics_plotting_df['group'], categories=precision_groups, ordered=True)
#     metrics_plotting_df = metrics_plotting_df.sort_values('group')
    
#     return metrics_plotting_df

# def create_mse_rmse_chart(metrics_df, metric_to_plot, y_label, chart_title_base, output_file_base):
#     plt.rcParams['font.sans-serif'] = ['Arial'] 
#     plt.rcParams['font.size'] = 10 
#     plt.rcParams['axes.unicode_minus'] = False 
    
#     fig, ax = plt.subplots(figsize=(12, 7.5)) 

#     precision_groups_ordered = ['High Precision', 'Low Precision', 'Mixed Precision']
#     num_precision_groups = len(precision_groups_ordered)
#     bar_width = 0.25
#     index = np.arange(num_precision_groups)
#     system_types_in_plot = ['ADAS1', 'ADAS2', 'Fusion'] 
#     color_map = {'ADAS1': '#4472C4', 'ADAS2': '#70AD47', 'Fusion': '#FFC000'}

#     all_values_for_ylim = []
#     all_rects_groups = [] 

#     for i, system_type_label in enumerate(system_types_in_plot):
#         values = []
#         for pg_group in precision_groups_ordered:
#             row = metrics_df[(metrics_df['group'] == pg_group) & (metrics_df['system_type'] == system_type_label)]
#             val_from_df = row[metric_to_plot].iloc[0] if not row.empty else 0
#             try:
#                 values.append(float(val_from_df))
#             except (ValueError, TypeError):
#                 values.append(0.0)

#         all_values_for_ylim.extend(values)

#         bar_positions = index + (i - (len(system_types_in_plot) - 1) / 2) * bar_width
#         rects = ax.bar(bar_positions, values, bar_width, label=system_type_label, color=color_map.get(system_type_label, 'gray'), edgecolor='black', linewidth=0.5)
#         all_rects_groups.append(rects)

#     if all_values_for_ylim:
#         min_y_val = 0
#         max_data_val = max(all_values_for_ylim) if all_values_for_ylim else 1.0 
#         if not all_values_for_ylim or max(all_values_for_ylim) == 0 : 
#              max_data_val = 1.0

#         padding_factor = 0.50 
#         y_axis_top_limit = max_data_val + (max_data_val * padding_factor)
        
#         if max_data_val == 0: 
#             y_axis_top_limit = 1.0 
#         elif max_data_val < 0.1: 
#             y_axis_top_limit = max(0.5, max_data_val + 0.2) 
#         elif y_axis_top_limit <= max_data_val : 
#              y_axis_top_limit = max_data_val * 1.5 

#         ax.set_ylim([min_y_val, y_axis_top_limit])
#         ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune='both', integer=False)) 
#     else:
#         ax.set_ylim([0, 1])


#     current_y_axis_top_for_offset = ax.get_ylim()[1] 
#     for rect_group in all_rects_groups:
#         for rect in rect_group:
#             height = rect.get_height()
#             label_text = f'{height:.4f}' 

#             offset_percentage = 0.015 
#             offset = offset_percentage * current_y_axis_top_for_offset
            
#             if height < offset * 2 and height > 0: 
#                 offset = height * 0.15 + 0.005 * current_y_axis_top_for_offset 
#             elif height == 0:
#                 offset = 0.01 * current_y_axis_top_for_offset

#             if height is not None and not (isinstance(height, float) and np.isnan(height)):
#                  ax.text(rect.get_x() + rect.get_width() / 2., 
#                         height + offset, 
#                         label_text,
#                         ha='center', va='bottom', fontsize=6.5, rotation=0,
#                         bbox=dict(facecolor='white', alpha=0.0, pad=0.1, edgecolor='none'))

#     col_width_group = 19      
#     col_width_adas1 = 9       
#     col_width_adas2 = 9       
#     # col_width_sut_header = 7 # No longer needed

#     table_lines = []
#     # Header: Remove SUT header
#     header_line = " " * col_width_group + \
#                   "ADAS1".ljust(col_width_adas1) + \
#                   "ADAS2".ljust(col_width_adas2)
#     table_lines.append(header_line)
    
#     for pg_key in precision_groups_ordered: 
#         pg_label_display = pg_key 
#         cfg = DEVICE_CONFIG[pg_key]
#         group_display_label = f"{pg_label_display}:".ljust(col_width_group) 
#         adas1_display = cfg['ADAS1'].ljust(col_width_adas1)
#         adas2_display = cfg['ADAS2'].ljust(col_width_adas2)
#         table_lines.append(f"{group_display_label}{adas1_display}{adas2_display}") # SUT display removed

#     device_info_text_right_table = "\n".join(table_lines)
    
#     legend_fontsize = 7
#     legend_title_fontsize = 8
#     legend_labelspacing = 0.6 
#     text_box_fontsize = 7 
#     text_box_linespacing = 1.25 

#     ax.text(0.66, 0.955, device_info_text_right_table, 
#              transform=ax.transAxes, 
#              fontsize=text_box_fontsize,
#              verticalalignment='top', 
#              horizontalalignment='left', 
#              bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='darkgrey', lw=0.7, alpha=0.95),
#              family='monospace', 
#              linespacing=text_box_linespacing)

#     ax.set_ylabel(y_label, fontsize=12, fontname='Arial', weight='normal')
#     ax.set_title(chart_title_base, fontsize=14, pad=25, fontname='Arial', weight='bold')
    
#     ax.set_xticks(index)
#     ax.set_xticklabels(precision_groups_ordered, fontname='Arial', weight='bold')
#     ax.tick_params(axis='both', which='major', labelsize=plt.rcParams['font.size'])


#     handles, labels = ax.get_legend_handles_labels()
#     by_label = dict(zip(labels, handles)) 
    
#     legend = ax.legend(by_label.values(), by_label.keys(), title="System Type", 
#                        loc='upper left', 
#                        bbox_to_anchor=(0.01, 0.955), 
#                        fontsize=legend_fontsize, 
#                        frameon=True, 
#                        facecolor='white', 
#                        edgecolor='darkgrey', 
#                        labelspacing=legend_labelspacing, 
#                        title_fontsize=legend_title_fontsize)
#     legend.get_frame().set_linewidth(0.7)
#     legend.get_frame().set_alpha(0.95)
#     for text in legend.get_texts(): text.set_fontname('Arial')
#     legend.get_title().set_fontname('Arial')
#     legend.get_title().set_fontweight('bold')


#     ax.grid(axis='y', linestyle='--', alpha=0.7)

#     output_pdf_file = f"{output_file_base}.pdf"
    
#     plt.tight_layout(rect=[0.03, 0.05, 0.97, 0.92]) 
    
#     plt.savefig(output_pdf_file, dpi=300, bbox_inches=None) 
#     print(f"Chart saved to: {output_pdf_file}")
#     plt.close(fig)

# def main():
#     print("ADAS Metrics Visualization Tool (MSE/RMSE only)")
#     print("===============================================")
#     metrics_plotting_df = get_user_input() 
    
#     create_mse_rmse_chart(metrics_plotting_df,
#                           metric_to_plot='MSE',
#                           y_label='Mean Square Error (MSE)',
#                           chart_title_base='MSE Comparison by Precision Group', 
#                           output_file_base='mse_comparison_final_py') 
#     create_mse_rmse_chart(metrics_plotting_df,
#                           metric_to_plot='RMSE',
#                           y_label='Root Mean Square Error (RMSE)',
#                           chart_title_base='RMSE Comparison by Precision Group', 
#                           output_file_base='rmse_comparison_final_py') 
                          
#     print("\nVisualizations for MSE and RMSE complete!")
#     print("Generated PDF files:")
#     print("1. mse_comparison_final_py.pdf")
#     print("2. rmse_comparison_final_py.pdf")

# if __name__ == "__main__":
#     main()



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator 
import pandas as pd
from datetime import datetime
import matplotlib # Import matplotlib itself

# --- 提高PDF字体质量 ---
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# --- END ---

# 设备名称配置
DEVICE_CONFIG = {
    'High Precision':  {'ADAS1': 'MAXEYE',  'ADAS2': 'JMS3',    'SUT': 'JMS2'},
    'Low Precision':   {'ADAS1': 'MINIEYE', 'ADAS2': 'MOTOVIS', 'SUT': 'JMS2'},
    'Mixed Precision': {'ADAS1': 'MAXEYE',  'ADAS2': 'MOTOVIS', 'SUT': 'JMS2'}
}

def get_user_input():
    """Get user input data for all precision groups for MSE and RMSE"""
    print("Please input ADAS metrics data (MSE and RMSE only):")
    metrics_data_list = []
    precision_groups = ['High Precision', 'Low Precision', 'Mixed Precision']
    
    for group_name in precision_groups:
        print(f"\n--- {group_name} Group Data ---")
        devices = DEVICE_CONFIG[group_name] 
        print(f"   (ADAS1: {devices['ADAS1']}, ADAS2: {devices['ADAS2']}, SUT Device for context: {devices['SUT']})")

        print("\nMSE values (vs GT):") 
        while True:
            try:
                mse_adas1 = float(input(f"{group_name} - {devices['ADAS1']} (as ADAS1) MSE: "))
                mse_adas2 = float(input(f"{group_name} - {devices['ADAS2']} (as ADAS2) MSE: "))
                mse_fusion = float(input(f"{group_name} - Fusion MSE: "))
                break
            except ValueError:
                print("Invalid input. Please enter numeric values.")
        
        print("\nRMSE values (vs GT):") 
        while True:
            try:
                rmse_adas1 = float(input(f"{group_name} - {devices['ADAS1']} (as ADAS1) RMSE: "))
                rmse_adas2 = float(input(f"{group_name} - {devices['ADAS2']} (as ADAS2) RMSE: "))
                rmse_fusion = float(input(f"{group_name} - Fusion RMSE: "))
                break
            except ValueError:
                print("Invalid input. Please enter numeric values.")

        metrics_data_list.append({'group': group_name, 'system_type': 'ADAS1', 'system_name': devices['ADAS1'], 'MSE': mse_adas1, 'RMSE': rmse_adas1})
        metrics_data_list.append({'group': group_name, 'system_type': 'ADAS2', 'system_name': devices['ADAS2'], 'MSE': mse_adas2, 'RMSE': rmse_adas2})
        metrics_data_list.append({'group': group_name, 'system_type': 'Fusion', 'system_name': 'Fusion', 'MSE': mse_fusion, 'RMSE': rmse_fusion})

    metrics_plotting_df = pd.DataFrame(metrics_data_list)
    metrics_plotting_df['group'] = pd.Categorical(metrics_plotting_df['group'], categories=precision_groups, ordered=True)
    metrics_plotting_df = metrics_plotting_df.sort_values('group')
    
    return metrics_plotting_df

def create_mse_rmse_chart(metrics_df, metric_to_plot, y_label, chart_title_base, output_file_base):
    plt.rcParams['font.sans-serif'] = ['Arial'] 
    plt.rcParams['font.size'] = 10 
    plt.rcParams['axes.unicode_minus'] = False 
    
    fig, ax = plt.subplots(figsize=(10, 7)) 

    precision_groups_ordered = ['High Precision', 'Low Precision', 'Mixed Precision']
    num_precision_groups = len(precision_groups_ordered)
    
    bar_width = 0.20
    
    index = np.arange(num_precision_groups)
    system_types_in_plot = ['ADAS1', 'ADAS2', 'Fusion'] 
    color_map = {'ADAS1': '#4472C4', 'ADAS2': '#70AD47', 'Fusion': '#FFC000'}

    all_values_for_ylim = []
    all_rects_groups = [] 

    for i, system_type_label in enumerate(system_types_in_plot):
        values = []
        for pg_group in precision_groups_ordered:
            row = metrics_df[(metrics_df['group'] == pg_group) & (metrics_df['system_type'] == system_type_label)]
            val_from_df = row[metric_to_plot].iloc[0] if not row.empty else 0
            try:
                values.append(float(val_from_df))
            except (ValueError, TypeError):
                values.append(0.0)

        all_values_for_ylim.extend(values)

        bar_positions = index + (i - (len(system_types_in_plot) - 1) / 2) * bar_width
        rects = ax.bar(bar_positions, values, bar_width, label=system_type_label, color=color_map.get(system_type_label, 'gray'), edgecolor='black', linewidth=0.5)
        all_rects_groups.append(rects)

    if all_values_for_ylim:
        min_y_val = 0
        max_data_val = max(all_values_for_ylim) if all_values_for_ylim else 1.0 
        if not all_values_for_ylim or max(all_values_for_ylim) == 0 : 
             max_data_val = 1.0

        padding_factor = 0.50 
        y_axis_top_limit = max_data_val + (max_data_val * padding_factor)
        
        if max_data_val == 0: 
            y_axis_top_limit = 1.0 
        elif max_data_val < 0.1: 
            y_axis_top_limit = max(0.5, max_data_val + 0.2) 
        elif y_axis_top_limit <= max_data_val : 
             y_axis_top_limit = max_data_val * 1.5 

        ax.set_ylim([min_y_val, y_axis_top_limit])
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune='both', integer=False)) 
    else:
        ax.set_ylim([0, 1])


    current_y_axis_top_for_offset = ax.get_ylim()[1] 
    for rect_group in all_rects_groups:
        for rect in rect_group:
            height = rect.get_height()
            label_text = f'{height:.4f}' 

            offset_percentage = 0.015 
            offset = offset_percentage * current_y_axis_top_for_offset
            
            if height < offset * 2 and height > 0: 
                offset = height * 0.15 + 0.005 * current_y_axis_top_for_offset 
            elif height == 0:
                offset = 0.01 * current_y_axis_top_for_offset

            if height is not None and not (isinstance(height, float) and np.isnan(height)):
                 ax.text(rect.get_x() + rect.get_width() / 2., 
                        height + offset, 
                        label_text,
                        ha='center', va='bottom', fontsize=6.5, rotation=0,
                        bbox=dict(facecolor='white', alpha=0.0, pad=0.1, edgecolor='none'))

    col_width_group = 19      
    col_width_adas1 = 9       
    col_width_adas2 = 9       

    table_lines = []
    header_line = " " * col_width_group + \
                  "ADAS1".ljust(col_width_adas1) + \
                  "ADAS2".ljust(col_width_adas2)
    table_lines.append(header_line)
    
    for pg_key in precision_groups_ordered: 
        pg_label_display = pg_key 
        cfg = DEVICE_CONFIG[pg_key]
        group_display_label = f"{pg_label_display}:".ljust(col_width_group) 
        adas1_display = cfg['ADAS1'].ljust(col_width_adas1)
        adas2_display = cfg['ADAS2'].ljust(col_width_adas2)
        table_lines.append(f"{group_display_label}{adas1_display}{adas2_display}")

    device_info_text_right_table = "\n".join(table_lines)
    
    legend_fontsize = 7
    legend_title_fontsize = 8
    legend_labelspacing = 0.6 
    text_box_fontsize = 7 
    text_box_linespacing = 1.25 

    # === 主要修改点：调整设备信息文本框的位置，为右上角图例腾出空间 ===
    ax.text(0.62, 0.78, device_info_text_right_table, 
             transform=ax.transAxes, 
             fontsize=text_box_fontsize,
             verticalalignment='top', 
             horizontalalignment='left', 
             bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='darkgrey', lw=0.7, alpha=0.95),
             family='monospace', 
             linespacing=text_box_linespacing)

    ax.set_ylabel(y_label, fontsize=12, fontname='Arial', weight='normal')
    
    # === 主要修改点 1: 注释掉下面这行代码以删除标题 ===
    # ax.set_title(chart_title_base, fontsize=14, pad=25, fontname='Arial', weight='bold')
    
    ax.set_xticks(index)
    ax.set_xticklabels(precision_groups_ordered, fontname='Arial', weight='bold')
    ax.tick_params(axis='both', which='major', labelsize=plt.rcParams['font.size'])


    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles)) 
    
    # === 主要修改点：将图例移动到右上角 ===
    legend = ax.legend(by_label.values(), by_label.keys(), title="System Type", 
                       loc='upper right', 
                       bbox_to_anchor=(0.99, 0.99), 
                       fontsize=legend_fontsize, 
                       frameon=True, 
                       facecolor='white', 
                       edgecolor='darkgrey', 
                       labelspacing=legend_labelspacing, 
                       title_fontsize=legend_title_fontsize)
    legend.get_frame().set_linewidth(0.7)
    legend.get_frame().set_alpha(0.95)
    for text in legend.get_texts(): text.set_fontname('Arial')
    legend.get_title().set_fontname('Arial')
    legend.get_title().set_fontweight('bold')


    ax.grid(axis='y', linestyle='--', alpha=0.7)

    output_pdf_file = f"{output_file_base}.pdf"
    
    # === 主要修改点 2: 调整布局以减少顶部空白 ===
    plt.tight_layout(rect=[0.03, 0.05, 0.97, 0.95]) 
    
    plt.savefig(output_pdf_file, dpi=300, bbox_inches=None) 
    print(f"Chart saved to: {output_pdf_file}")
    plt.close(fig)

def main():
    print("ADAS Metrics Visualization Tool (MSE/RMSE only)")
    print("===============================================")
    metrics_plotting_df = get_user_input() 
    
    # 由于标题参数 'chart_title_base' 不再被使用，我们可以将其设置为空字符串或保留不动，
    # 因为函数内部已经不再调用 ax.set_title()
    create_mse_rmse_chart(metrics_plotting_df,
                          metric_to_plot='MSE',
                          y_label='Mean Square Error (MSE)',
                          chart_title_base='MSE Comparison by Precision Group', # 此参数现在被忽略
                          output_file_base='mse_comparison_final_py') 
    create_mse_rmse_chart(metrics_plotting_df,
                          metric_to_plot='RMSE',
                          y_label='Root Mean Square Error (RMSE)',
                          chart_title_base='RMSE Comparison by Precision Group', # 此参数现在被忽略
                          output_file_base='rmse_comparison_final_py') 
                          
    print("\nVisualizations for MSE and RMSE complete!")
    print("Generated PDF files:")
    print("1. mse_comparison_final_py.pdf")
    print("2. rmse_comparison_final_py.pdf")

if __name__ == "__main__":
    main()