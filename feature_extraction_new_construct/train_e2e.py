# # # file: train_e2e.py (最终版本)

# # import torch
# # import torch.nn as nn
# # import numpy as np
# # from torch.utils.data import DataLoader, TensorDataset
# # from torch.optim import Adam

# # # 确保导入路径正确
# # from models.end_to_end_model import EndToEndADASModel
# # from config import CONFIG

# # # 辅助函数：损失函数定义
# # def diversity_loss_fn(feat1, feat2, eps=1e-8):
# #     """计算多样性损失 (负的余弦相似度)"""
# #     feat1_norm = torch.nn.functional.normalize(feat1, p=2, dim=1)
# #     feat2_norm = torch.nn.functional.normalize(feat2, p=2, dim=1)
# #     cosine_sim = torch.sum(feat1_norm * feat2_norm, dim=1)
# #     return -torch.mean(cosine_sim)

# # # 训练器类 (包含数据加载逻辑)
# # class EndToEndTrainer:
# #     def __init__(self, config):
# #         self.config = config
# #         self.e2e_config = config['end_to_end']
        
# #         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
# #         # 实例化模型
# #         self.model = EndToEndADASModel(config).to(self.device)
        
# #         self.task_loss_fn = nn.MSELoss()
# #         self.optimizer = Adam(self.model.parameters(), lr=self.e2e_config['learning_rate'])
        
# #         print("--- End-to-End Model Structure ---")
# #         print(self.model)
# #         print(f"\nTraining on device: {self.device}")

# #     def prepare_data(self):
# #         """根据CONFIG中的文件路径和固定GT值加载和准备数据。"""
# #         print("\n--- Preparing Data ---")
# #         adas1_path = self.e2e_config['adas1']['data_path']
# #         adas2_path = self.e2e_config['adas2']['data_path']
# #         fixed_gt_value = self.e2e_config['ground_truth_value']
# #         batch_size = self.e2e_config['batch_size']

# #         try:
# #             print(f"Loading ADAS1 data from: {adas1_path}")
# #             # =================== CHANGE IS HERE ===================
# #             # 删除了 delimiter=' '，让 numpy 自动处理多个空格
# #             data1 = np.loadtxt(adas1_path, skiprows=1)
# #             # ======================================================

# #             print(f"Loading ADAS2 data from: {adas2_path}")
# #             # =================== CHANGE IS HERE ===================
# #             # 删除了 delimiter=' '，让 numpy 自动处理多个空格
# #             data2 = np.loadtxt(adas2_path, skiprows=1)
# #             # ======================================================
            
# #         except Exception as e:
# #             print("\n\n!!! DATA LOADING ERROR !!!")
# #             print(f"Failed to load data files: {e}")
# #             print("Please check if the file paths in config.py are correct, the files exist,")
# #             print("and the data format is correct (e.g., consistent number of columns).")
# #             import traceback
# #             traceback.print_exc()
# #             exit() 

# #         if data1.shape[0] != data2.shape[0]:
# #             print(f"Warning: ADAS1 ({data1.shape[0]} samples) and ADAS2 ({data2.shape[0]} samples) data have different lengths. Truncating to the smaller size.")
# #             min_len = min(data1.shape[0], data2.shape[0])
# #             data1 = data1[:min_len]
# #             data2 = data2[:min_len]
        
# #         num_samples = data1.shape[0]
# #         if num_samples == 0:
# #             print("Error: No data samples found after alignment. Exiting.")
# #             exit()

# #         print(f"Number of aligned samples: {num_samples}")
# #         labels = np.full((num_samples, 1), fixed_gt_value)
# #         print(f"Using fixed Ground Truth value: {fixed_gt_value}")

# #         tensor1 = torch.FloatTensor(data1)
# #         tensor2 = torch.FloatTensor(data2)
# #         labels_tensor = torch.FloatTensor(labels)

# #         e2e_dataset = TensorDataset(tensor1, tensor2, labels_tensor)
# #         train_loader = DataLoader(e2e_dataset, batch_size=batch_size, shuffle=True)
        
# #         print("--- Data preparation complete. ---")
# #         return train_loader

# #     def train(self):
# #         train_loader = self.prepare_data()

# #         print("\n============== 端到端训练开始 ==============")
        
# #         alpha = self.e2e_config['loss_weights']['alpha']
# #         beta = self.e2e_config['loss_weights']['beta']
# #         best_loss = float('inf')

# #         for epoch in range(self.e2e_config['epochs']):
# #             self.model.train()
# #             total_loss, total_task_loss, total_div_loss = 0, 0, 0
            
# #             for batch_adas1, batch_adas2, batch_gt in train_loader:
# #                 batch_adas1 = batch_adas1.to(self.device)
# #                 batch_adas2 = batch_adas2.to(self.device)
# #                 batch_gt = batch_gt.to(self.device)

# #                 self.optimizer.zero_grad()
# #                 predicted_distance, xe1, xe2 = self.model(batch_adas1, batch_adas2)
                
# #                 l_task = self.task_loss_fn(predicted_distance, batch_gt)
# #                 l_div = diversity_loss_fn(xe1, xe2)
                
# #                 loss = alpha * l_task + beta * l_div
                
# #                 loss.backward()
# #                 self.optimizer.step()
                
# #                 total_loss += loss.item()
# #                 total_task_loss += l_task.item()
# #                 total_div_loss += l_div.item()

# #             avg_loss = total_loss / len(train_loader)
# #             avg_task_loss = total_task_loss / len(train_loader)
# #             avg_div_loss = total_div_loss / len(train_loader)
            
# #             print(
# #                 f"Epoch [{epoch+1}/{self.e2e_config['epochs']}] | "
# #                 f"Total Loss: {avg_loss:.6f} | "
# #                 f"Task Loss: {avg_task_loss:.6f} | "
# #                 f"Div Loss: {avg_div_loss:.6f}"
# #             )
            
# #             if avg_loss < best_loss:
# #                 best_loss = avg_loss
# #                 # 修改这里的路径
# #             torch.save(self.model.state_dict(), "e2e_results/best_e2e_model.pth")
# #             print(f"  -> New best model saved in 'results/' with loss: {best_loss:.6f}")

# # # 脚本的执行入口
# # if __name__ == '__main__':
# #     try:
# #         trainer = EndToEndTrainer(CONFIG)
# #         trainer.train()
# #     except Exception as e:
# #         print("\n\n!!! AN UNEXPECTED ERROR OCCURRED !!!")
# #         print(f"Error: {e}")
# #         import traceback
# #         traceback.print_exc()



# # file: train_e2e.py (最终版 - 期望3个返回值)

# import torch
# import torch.nn as nn
# import numpy as np
# from torch.utils.data import DataLoader, TensorDataset
# from torch.optim import Adam
# import os
# import traceback

# # 确保导入路径正确
# from models.end_to_end_model import EndToEndADASModel
# from config import CONFIG

# # 辅助函数：损失函数定义
# def diversity_loss_fn(feat1, feat2, eps=1e-8):
#     """计算多样性损失 (负的余弦相似度)"""
#     feat1_norm = torch.nn.functional.normalize(feat1, p=2, dim=1)
#     feat2_norm = torch.nn.functional.normalize(feat2, p=2, dim=1)
#     cosine_sim = torch.sum(feat1_norm * feat2_norm, dim=1)
#     return -torch.mean(cosine_sim)

# # 训练器类
# class EndToEndTrainer:
#     def __init__(self, config):
#         self.config = config
#         self.e2e_config = config['end_to_end']
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
#         # 创建结果文件夹
#         self.output_dir = "e2e_results"
#         os.makedirs(self.output_dir, exist_ok=True)
#         self.model_path = os.path.join(self.output_dir, "best_e2e_model.pth")
        
#         self.model = EndToEndADASModel(config).to(self.device)
#         self.task_loss_fn = nn.MSELoss()
#         self.optimizer = Adam(self.model.parameters(), lr=self.e2e_config['learning_rate'])
        
#         print("--- End-to-End Model Structure ---")
#         print(self.model)
#         print(f"\nResults will be saved in: '{self.output_dir}'")
#         print(f"Training on device: {self.device}")

#     def prepare_data(self):
#         print("\n--- Preparing Data for Training ---")
#         adas1_path = self.e2e_config['adas1']['data_path']
#         adas2_path = self.e2e_config['adas2']['data_path']
#         fixed_gt_value = self.e2e_config['ground_truth_value']
#         batch_size = self.e2e_config['batch_size']

#         try:
#             data1 = np.loadtxt(adas1_path, skiprows=1)
#             data2 = np.loadtxt(adas2_path, skiprows=1)
#         except Exception as e:
#             print(f"\n\n!!! DATA LOADING ERROR !!!\nFailed to load data files: {e}")
#             exit() 

#         min_len = min(data1.shape[0], data2.shape[0])
#         data1, data2 = data1[:min_len], data2[:min_len]
        
#         num_samples = data1.shape[0]
#         if num_samples == 0:
#             print("Error: No data samples found. Exiting.")
#             exit()

#         print(f"Number of aligned samples: {num_samples}")
#         labels = np.full((num_samples, 1), fixed_gt_value)
        
#         tensor1, tensor2 = torch.FloatTensor(data1), torch.FloatTensor(data2)
#         labels_tensor = torch.FloatTensor(labels)

#         e2e_dataset = TensorDataset(tensor1, tensor2, labels_tensor)
#         train_loader = DataLoader(e2e_dataset, batch_size=batch_size, shuffle=True)
        
#         print("--- Data preparation complete. ---")
#         return train_loader

#     def train(self):
#         train_loader = self.prepare_data()

#         print("\n============== 端到端训练开始 ==============")
#         alpha = self.e2e_config['loss_weights']['alpha']
#         beta = self.e2e_config['loss_weights']['beta']
#         best_loss = float('inf')

#         for epoch in range(self.e2e_config['epochs']):
#             self.model.train()
#             total_loss = 0
            
#             for batch_adas1, batch_adas2, batch_gt in train_loader:
#                 batch_adas1, batch_adas2, batch_gt = batch_adas1.to(self.device), batch_adas2.to(self.device), batch_gt.to(self.device)

#                 self.optimizer.zero_grad()
                
#                 # ===================  修改点 1  ===================
#                 # 只期望接收3个返回值
#                 predicted_distance, xe1, xe2 = self.model(batch_adas1, batch_adas2)
#                 # ===============================================
                
#                 l_task = self.task_loss_fn(predicted_distance, batch_gt)
#                 l_div = diversity_loss_fn(xe1, xe2)
#                 loss = alpha * l_task + beta * l_div
                
#                 loss.backward()
#                 self.optimizer.step()
#                 total_loss += loss.item()

#             avg_loss = total_loss / len(train_loader)
#             print(f"Epoch [{epoch+1}/{self.e2e_config['epochs']}] | Total Loss: {avg_loss:.6f}")
            
#             if avg_loss < best_loss:
#                 best_loss = avg_loss
#                 torch.save(self.model.state_dict(), self.model_path)
#                 print(f"  -> New best model saved with loss: {best_loss:.6f}")
        
#         print("\n============== 训练完成 ==============")
#         self.extract_and_compare_distances()

#     def extract_and_compare_distances(self):
#         print("\n--- Starting E2E Distance Prediction and Extraction ---")
        
#         if not os.path.exists(self.model_path):
#             print(f"Error: Model file not found at '{self.model_path}'.")
#             return

#         print(f"Loading best trained model from: {self.model_path}")
#         self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
#         self.model.eval()

#         adas1_path = self.e2e_config['adas1']['data_path']
#         adas2_path = self.e2e_config['adas2']['data_path']
#         data1, data2 = np.loadtxt(adas1_path, skiprows=1), np.loadtxt(adas2_path, skiprows=1)
#         min_len = min(data1.shape[0], data2.shape[0])
#         data1, data2 = data1[:min_len], data2[:min_len]

#         comparison_column_index = 7
#         raw_adas1_distances, raw_adas2_distances = data1[:, comparison_column_index], data2[:, comparison_column_index]

#         tensor1, tensor2 = torch.FloatTensor(data1), torch.FloatTensor(data2)
#         inference_dataset = TensorDataset(tensor1, tensor2)
#         inference_loader = DataLoader(inference_dataset, batch_size=self.e2e_config['batch_size'], shuffle=False)
        
#         all_predicted_distances = []
#         with torch.no_grad():
#             for batch_adas1, batch_adas2 in inference_loader:
#                 batch_adas1, batch_adas2 = batch_adas1.to(self.device), batch_adas2.to(self.device)
                
#                 # ===================  修改点 2  ===================
#                 # 只期望接收3个返回值，并且只关心第一个
#                 predicted_distance_batch, _, _ = self.model(batch_adas1, batch_adas2)
#                 # ===============================================
                
#                 all_predicted_distances.append(predicted_distance_batch.cpu().numpy())

#         predicted_distances_final = np.vstack(all_predicted_distances).flatten()

#         np.save(os.path.join(self.output_dir, "predicted_distance.npy"), predicted_distances_final)
#         np.save(os.path.join(self.output_dir, "raw_adas1_distance.npy"), raw_adas1_distances)
#         np.save(os.path.join(self.output_dir, "raw_adas2_distance.npy"), raw_adas2_distances)

#         print(f"\n--- Extraction Complete --- \nResults saved in folder: '{self.output_dir}'")
        
#         gt_value = self.e2e_config['ground_truth_value']
#         ground_truth_distances = np.full_like(predicted_distances_final, gt_value)
#         mse_model_vs_gt = np.mean((predicted_distances_final - ground_truth_distances)**2)
#         mse_raw1_vs_gt = np.mean((raw_adas1_distances - ground_truth_distances)**2)
#         mse_raw2_vs_gt = np.mean((raw_adas2_distances - ground_truth_distances)**2)
        
#         print("\n--- Quick Performance Evaluation (MSE vs GT) ---")
#         print(f"Model Prediction MSE: {mse_model_vs_gt:.4f}")
#         print(f"Raw ADAS1 (col {comparison_column_index+1}) MSE: {mse_raw1_vs_gt:.4f}")
#         print(f"Raw ADAS2 (col {comparison_column_index+1}) MSE: {mse_raw2_vs_gt:.4f}")

# if __name__ == '__main__':
#     try:
#         trainer = EndToEndTrainer(CONFIG)
#         trainer.train()
#     except Exception as e:
#         print(f"\n\n!!! AN UNEXPECTED ERROR OCCURRED !!!\nError: {e}")
#         traceback.print_exc()
# file: train_e2e.py (最终自监督版本 - 包含训练和提取功能)

# import torch
# import torch.nn as nn
# import numpy as np
# from torch.utils.data import DataLoader, TensorDataset
# from torch.optim import Adam
# import os
# import traceback

# from models.end_to_end_model import EndToEndADASModel
# from config import CONFIG

# def diversity_loss_fn(feat1, feat2, eps=1e-8):
#     feat1_norm = torch.nn.functional.normalize(feat1, p=2, dim=1)
#     feat2_norm = torch.nn.functional.normalize(feat2, p=2, dim=1)
#     cosine_sim = torch.sum(feat1_norm * feat2_norm, dim=1)
#     return -torch.mean(cosine_sim)

# class EndToEndTrainer:
#     def __init__(self, config):
#         self.config = config
#         self.e2e_config = config['end_to_end']
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
#         self.output_dir = "e2e_self_supervised_results"
#         os.makedirs(self.output_dir, exist_ok=True)
#         self.model_path = os.path.join(self.output_dir, "best_e2e_model.pth")
        
#         self.model = EndToEndADASModel(config).to(self.device)
#         self.task_loss_fn = nn.MSELoss()
#         self.optimizer = Adam(self.model.parameters(), lr=self.e2e_config['learning_rate'])
        
#         print("--- Self-Supervised End-to-End Model Structure ---")
#         print(self.model)
#         print(f"\nResults will be saved in: '{self.output_dir}'")
#         print(f"Device: {self.device}")

#     def prepare_data(self):
#         print("\n--- Preparing Data for Self-Supervised Training ---")
#         adas1_path = self.e2e_config['adas1']['data_path']
#         adas2_path = self.e2e_config['adas2']['data_path']
#         batch_size = self.e2e_config['batch_size']

#         try:
#             data1 = np.loadtxt(adas1_path, skiprows=1)
#             data2 = np.loadtxt(adas2_path, skiprows=1)
#         except Exception as e:
#             print(f"\n\n!!! DATA LOADING ERROR !!!\nFailed to load data files: {e}")
#             exit() 

#         min_len = min(data1.shape[0], data2.shape[0])
#         data1, data2 = data1[:min_len], data2[:min_len]
#         print(f"Number of aligned samples: {data1.shape[0]}")

#         tensor1, tensor2 = torch.FloatTensor(data1), torch.FloatTensor(data2)
#         e2e_dataset = TensorDataset(tensor1, tensor2)
#         train_loader = DataLoader(e2e_dataset, batch_size=batch_size, shuffle=True)
#         print("--- Data preparation complete. ---")
#         return train_loader

#     def train(self):
#         train_loader = self.prepare_data()

#         print("\n============== 自监督端到端训练开始 ==============")
#         alpha = self.e2e_config['loss_weights']['alpha']
#         beta = self.e2e_config['loss_weights']['beta']
#         best_loss = float('inf')

#         w1 = self.model.fusion_module.feature1_weight
#         w2 = self.model.fusion_module.feature2_weight

#         for epoch in range(self.e2e_config['epochs']):
#             self.model.train()
#             total_loss, total_task_loss, total_div_loss = 0, 0, 0
            
#             for batch_adas1, batch_adas2 in train_loader:
#                 batch_adas1, batch_adas2 = batch_adas1.to(self.device), batch_adas2.to(self.device)
#                 self.optimizer.zero_grad()
#                 fused_feature, xe1, xe2 = self.model(batch_adas1, batch_adas2)
                
#                 with torch.no_grad():
#                     adjusted_xe1 = self.model.fusion_module.adjust_dim1(xe1)
#                     adjusted_xe2 = self.model.fusion_module.adjust_dim2(xe2)
#                     y_pseudo = w1 * adjusted_xe1 + w2 * adjusted_xe2
                
#                 l_task = self.task_loss_fn(fused_feature, y_pseudo)
#                 l_div = diversity_loss_fn(xe1, xe2)
#                 loss = alpha * l_task + beta * l_div
                
#                 loss.backward()
#                 self.optimizer.step()
                
#                 total_loss += loss.item()
#                 total_task_loss += l_task.item()
#                 total_div_loss += l_div.item()

#             avg_loss = total_loss / len(train_loader)
#             avg_task_loss = total_task_loss / len(train_loader)
#             avg_div_loss = total_div_loss / len(train_loader)
            
#             print(f"Epoch [{epoch+1}/{self.e2e_config['epochs']}] | Total Loss: {avg_loss:.6f} | Task Loss (Self-Supervised): {avg_task_loss:.6f} | Div Loss: {avg_div_loss:.6f}")
            
#             if avg_loss < best_loss:
#                 best_loss = avg_loss
#                 torch.save(self.model.state_dict(), self.model_path)
#                 print(f"  -> New best self-supervised model saved with loss: {best_loss:.6f}")
        
#         print("\n============== 训练完成 ==============")


# # ========================================================
# def extract_features(config):
#     print("\n--- Starting Feature Extraction from Trained Model ---")
    
#     e2e_config = config['end_to_end']
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     output_dir = "e2e_self_supervised_results"
#     model_path = os.path.join(output_dir, "best_e2e_model.pth")
    
#     if not os.path.exists(model_path):
#         print(f"Error: Model file not found at '{model_path}'")
#         return

#     print(f"Loading trained model from: {model_path}")
#     model = EndToEndADASModel(config).to(device)
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.eval()

#     # --- 1. 加载原始数据 ---
#     adas1_path = e2e_config['adas1']['data_path']
#     adas2_path = e2e_config['adas2']['data_path']
#     data1 = np.loadtxt(adas1_path, skiprows=1) 
#     data2 = np.loadtxt(adas2_path, skiprows=1)

#     min_len = min(data1.shape[0], data2.shape[0])
#     data1, data2 = data1[:min_len], data2[:min_len]

#     # --- 2. 提取原始数据的第八列，并计算其统计数据 (均值和标准差) ---
#     comparison_column_index = 7
#     raw_adas1_distances = data1[:, [comparison_column_index]]
#     raw_adas2_distances = data2[:, [comparison_column_index]]
    
#     # 将两个原始距离数据合并，以计算一个统一的、代表“现实世界”的数值范围
#     combined_raw_distances = np.concatenate([raw_adas1_distances, raw_adas2_distances])
#     mean_real_world = combined_raw_distances.mean()
#     std_real_world = combined_raw_distances.std()
    
#     print(f"\nOriginal data scale (from column {comparison_column_index+1}): Mean={mean_real_world:.4f}, Std={std_real_world:.4f}")

#     # --- 3. 用模型进行预测，得到在“标准化世界”的输出 ---
#     tensor1, tensor2 = torch.FloatTensor(data1), torch.FloatTensor(data2)
#     inference_dataset = TensorDataset(tensor1, tensor2)
#     inference_loader = DataLoader(inference_dataset, batch_size=e2e_config['batch_size'], shuffle=False)
    
#     all_fused_features = []
#     with torch.no_grad():
#         for batch_adas1, batch_adas2 in inference_loader:
#             batch_adas1, batch_adas2 = batch_adas1.to(device), batch_adas2.to(device)
#             fused_feature_batch, _, _ = model(batch_adas1, batch_adas2)
#             all_fused_features.append(fused_feature_batch.cpu().numpy())

#     fused_features_normalized = np.vstack(all_fused_features)

#     # --- 4. 对模型的输出进行反标准化，映射回“现实世界”的数值范围 ---
#     mean_model_world = fused_features_normalized.mean()
#     std_model_world = fused_features_normalized.std()
    
#     print(f"Model output scale (normalized): Mean={mean_model_world:.4f}, Std={std_model_world:.4f}")

#     # 这是反标准化的核心公式: z-score reversal
#     fused_features_rescaled = (fused_features_normalized - mean_model_world) / std_model_world * std_real_world + mean_real_world
    
#     print(f"Rescaled model output scale: Mean={fused_features_rescaled.mean():.4f}, Std={fused_features_rescaled.std():.4f}")

#     # --- 5. 保存 rescaled 之后的结果 ---
#     np.save(os.path.join(output_dir, "fused_feature.npy"), fused_features_rescaled) # 保存重新缩放后的融合特征
#     np.save(os.path.join(output_dir, "raw_adas1_distance.npy"), raw_adas1_distances)
#     np.save(os.path.join(output_dir, "raw_adas2_distance.npy"), raw_adas2_distances)

#     print("\n--- Extraction Complete ---")
#     print(f"Results saved in folder: '{output_dir}'")
#     print(f"  - fused_feature.npy (Shape: {fused_features_rescaled.shape})")
#     print(f"  - raw_adas1_distance.npy (Shape: {raw_adas1_distances.shape})")
#     print(f"  - raw_adas2_distance.npy (Shape: {raw_adas2_distances.shape})")
# # ========================================================







# if __name__ == '__main__':
#     import argparse
#     parser = argparse.ArgumentParser(description="Self-Supervised End-to-End ADAS Model")
#     parser.add_argument('--mode', type=str, default='train_extract', choices=['train', 'extract', 'train_extract'],
#                         help="Choose mode: 'train' to only train, 'extract' to only extract, 'train_extract' to train then extract.")
#     args = parser.parse_args()

#     try:
#         if args.mode == 'train' or args.mode == 'train_extract':
#             trainer = EndToEndTrainer(CONFIG)
#             trainer.train()
        
#         if args.mode == 'extract' or args.mode == 'train_extract':
#             extract_features(CONFIG)

#     except Exception as e:
#         print(f"\n\n!!! AN UNEXPECTED ERROR OCCURRED !!!\nError: {e}")
#         traceback.print_exc()


# file: train_e2e.py (最终简洁版 - 只处理单列.npy输入, 只输出融合结果)

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
import os
import traceback

from models.end_to_end_model import EndToEndADASModel
from config import CONFIG

def diversity_loss_fn(feat1, feat2, eps=1e-8):
    feat1_norm = torch.nn.functional.normalize(feat1, p=2, dim=1)
    feat2_norm = torch.nn.functional.normalize(feat2, p=2, dim=1)
    cosine_sim = torch.sum(feat1_norm * feat2_norm, dim=1)
    return -torch.mean(cosine_sim)

class EndToEndPipeline:
    def __init__(self, config):
        self.config = config
        self.e2e_config = config['end_to_end']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.output_dir = "e2e_self_supervised_results"
        os.makedirs(self.output_dir, exist_ok=True)
        self.model_path = os.path.join(self.output_dir, "best_e2e_model.pth")
        
        self.model = EndToEndADASModel(config).to(self.device)
        self.task_loss_fn = nn.MSELoss()
        self.optimizer = Adam(self.model.parameters(), lr=self.e2e_config['learning_rate'])
        
        print("--- Self-Supervised End-to-End Model Structure ---")
        print(self.model)
        print(f"\nResults will be saved in: '{self.output_dir}'")
        print(f"Device: {self.device}")

    def _prepare_data_for_training(self):
        print("\n--- Preparing Data for Training ---")
        adas1_path = self.e2e_config['adas1']['data_path']
        adas2_path = self.e2e_config['adas2']['data_path']
        batch_size = self.e2e_config['batch_size']

        try:
            data1 = np.load(adas1_path)
            data2 = np.load(adas2_path)
        except Exception as e:
            print(f"\n\n!!! DATA LOADING ERROR !!!\nFailed to load data files: {e}")
            exit() 

        if data1.ndim == 1: data1 = data1.reshape(-1, 1)
        if data2.ndim == 1: data2 = data2.reshape(-1, 1)

        min_len = min(data1.shape[0], data2.shape[0])
        data1, data2 = data1[:min_len], data2[:min_len]
        
        if data1.shape[0] == 0:
            print("\n\n!!! FATAL ERROR: No data samples after alignment. Check input files. !!!")
            exit()
            
        print(f"Number of aligned samples: {data1.shape[0]}")
        print(f"Input data shape: ADAS1={data1.shape}, ADAS2={data2.shape}")

        tensor1, tensor2 = torch.FloatTensor(data1), torch.FloatTensor(data2)
        e2e_dataset = TensorDataset(tensor1, tensor2)
        train_loader = DataLoader(e2e_dataset, batch_size=batch_size, shuffle=True)
        
        print("--- Data preparation complete. ---")
        return train_loader

    def train(self):
        train_loader = self._prepare_data_for_training()
        print("\n============== 自监督端到端训练开始 ==============")
        
        alpha = self.e2e_config['loss_weights']['alpha']
        beta = self.e2e_config['loss_weights']['beta']
        best_loss = float('inf')

        w1 = self.model.fusion_module.feature1_weight
        w2 = self.model.fusion_module.feature2_weight

        for epoch in range(self.e2e_config['epochs']):
            self.model.train()
            total_loss = 0
            
            for batch_adas1, batch_adas2 in train_loader:
                batch_adas1, batch_adas2 = batch_adas1.to(self.device), batch_adas2.to(self.device)
                self.optimizer.zero_grad()
                fused_feature, xe1, xe2 = self.model(batch_adas1, batch_adas2)
                
                with torch.no_grad():
                    adjusted_xe1 = self.model.fusion_module.adjust_dim1(xe1)
                    adjusted_xe2 = self.model.fusion_module.adjust_dim2(xe2)
                    y_pseudo = w1 * adjusted_xe1 + w2 * adjusted_xe2
                
                l_task = self.task_loss_fn(fused_feature, y_pseudo)
                l_div = diversity_loss_fn(xe1, xe2)
                loss = alpha * l_task + beta * l_div
                
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{self.e2e_config['epochs']}] | Total Loss: {avg_loss:.6f}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.model.state_dict(), self.model_path)
                print(f"  -> New best self-supervised model saved with loss: {best_loss:.6f}")
        
        print("\n============== 训练完成 ==============")

    def extract_fused_feature(self):
        print("\n--- Starting Fused Feature Extraction ---")
        
        if not os.path.exists(self.model_path):
            print(f"Error: Model file not found at '{self.model_path}'. Please train first.")
            return

        print(f"Loading best trained model from: {self.model_path}")
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()

        adas1_path = self.e2e_config['adas1']['data_path']
        adas2_path = self.e2e_config['adas2']['data_path']
        data1, data2 = np.load(adas1_path), np.load(adas2_path)

        if data1.ndim == 1: data1 = data1.reshape(-1, 1)
        if data2.ndim == 1: data2 = data2.reshape(-1, 1)

        min_len = min(data1.shape[0], data2.shape[0])
        data1, data2 = data1[:min_len], data2[:min_len]

        # 计算反标准化的统计数据
        mean_real_world = data1.mean() # 因为是单列，直接计算即可
        std_real_world = data1.std()

        tensor1, tensor2 = torch.FloatTensor(data1), torch.FloatTensor(data2)
        inference_dataset = TensorDataset(tensor1, tensor2)
        inference_loader = DataLoader(inference_dataset, batch_size=self.e2e_config['batch_size'], shuffle=False)
        
        all_fused_features = []
        with torch.no_grad():
            for batch_adas1, batch_adas2 in inference_loader:
                batch_adas1, batch_adas2 = batch_adas1.to(self.device), batch_adas2.to(self.device)
                fused_feature_batch, _, _ = self.model(batch_adas1, batch_adas2)
                all_fused_features.append(fused_feature_batch.cpu().numpy())

        fused_features_normalized = np.vstack(all_fused_features)

        # 反标准化
        mean_model_world = fused_features_normalized.mean()
        std_model_world = fused_features_normalized.std()
        fused_features_rescaled = (fused_features_normalized - mean_model_world) / (std_model_world + 1e-8) * std_real_world + mean_real_world
        
        # 只保存融合后的特征
        save_path = os.path.join(self.output_dir, "fused_feature.npy")
        np.save(save_path, fused_features_rescaled)

        print("\n--- Extraction Complete ---")
        print(f"Fused feature saved to: {save_path}")
        print(f"Shape: {fused_features_rescaled.shape}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Self-Supervised End-to-End ADAS Model")
    parser.add_argument('--mode', type=str, default='train_extract', choices=['train', 'extract', 'train_extract'],
                        help="Choose mode: 'train' to only train, 'extract' to only extract, 'train_extract' to train then extract.")
    args = parser.parse_args()

    try:
        pipeline = EndToEndPipeline(CONFIG)
        if args.mode == 'train' or args.mode == 'train_extract':
            pipeline.train()
        
        if args.mode == 'extract' or args.mode == 'train_extract':
            pipeline.extract_fused_feature()

    except Exception as e:
        print(f"\n\n!!! AN UNEXPECTED ERROR OCCURRED !!!\nError: {e}")
        traceback.print_exc()