# # train.py
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, TensorDataset
# from torch.optim import Adam

# class Trainer:
#     def __init__(self, model, config, output_manager):
#         self.model = model
#         self.config = config
#         self.output_manager = output_manager
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.model = self.model.to(self.device)
        
#         self.criterion = nn.MSELoss()
#         self.optimizer = Adam(self.model.parameters(), 
#                             lr=config['learning_rate'])
    
#     def train(self, train_data, train_labels, val_data, val_labels):
#         train_dataset = TensorDataset(
#             torch.FloatTensor(train_data),
#             torch.FloatTensor(train_labels)
#         )
#         train_loader = DataLoader(train_dataset, 
#                                 batch_size=self.config['batch_size'],
#                                 shuffle=True)
        
#         best_val_loss = float('inf')
        
#         # 记录训练配置
#         self.output_manager.log_info("============== 训练开始 ==============")
#         self.output_manager.log_info(f"总epoch数: {self.config['epochs']}")
#         self.output_manager.log_info(f"批大小: {self.config['batch_size']}")
#         self.output_manager.log_info(f"学习率: {self.config['learning_rate']}")
#         self.output_manager.log_info(f"使用设备: {self.device}")
        
#         for epoch in range(self.config['epochs']):
#             self.model.train()
#             total_loss = 0
            
#             for batch_x, batch_y in train_loader:
#                 batch_x = batch_x.to(self.device)
#                 batch_y = batch_y.to(self.device)
                
#                 self.optimizer.zero_grad()
#                 features, predicted = self.model(batch_x)
#                 loss = self.criterion(predicted, batch_y)
                
#                 loss.backward()
#                 self.optimizer.step()
                
#                 total_loss += loss.item()
            
#             avg_train_loss = total_loss / len(train_loader)
#             val_loss = self.validate(val_data, val_labels)
            
#             if val_loss < best_val_loss:
#                 best_val_loss = val_loss
#                 torch.save(self.model.state_dict(), 
#                           self.output_manager.get_model_path())
            
#             # 记录训练进度
#             self.output_manager.log_info(
#                 f"Epoch [{epoch+1}/{self.config['epochs']}] "
#                 f"Train Loss: {avg_train_loss:.6f} "
#                 f"Val Loss: {val_loss:.6f}"
#             )
    
#     def validate(self, val_data, val_labels):
#         self.model.eval()
#         with torch.no_grad():
#             val_x = torch.FloatTensor(val_data).to(self.device)
#             val_y = torch.FloatTensor(val_labels).to(self.device)
            
#             features, predicted = self.model(val_x)
#             val_loss = self.criterion(predicted, val_y)
            
#         return val_loss.item()

# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, TensorDataset
# from torch.optim import Adam

# class Trainer:
#     def __init__(self, model, config, output_manager):
#         self.model = model
#         self.config = config
#         self.output_manager = output_manager
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.model = self.model.to(self.device)
        
#         self.feature_criterion = nn.L1Loss()
#         self.gt_criterion = nn.MSELoss()
#         self.optimizer = Adam(self.model.parameters(), 
#                             lr=config['learning_rate'])
    
#     def train(self, train_data, train_labels):
#         train_dataset = TensorDataset(
#             torch.FloatTensor(train_data),
#             torch.FloatTensor(train_labels)
#         )
#         train_loader = DataLoader(train_dataset, 
#                                 batch_size=self.config['batch_size'],
#                                 shuffle=True)
        
#         best_train_loss = float('inf')
        
#         # 记录训练配置
#         self.output_manager.log_info("============== 训练开始 ==============")
#         self.output_manager.log_info(f"总epoch数: {self.config['epochs']}")
#         self.output_manager.log_info(f"批大小: {self.config['batch_size']}")
#         self.output_manager.log_info(f"学习率: {self.config['learning_rate']}")
#         self.output_manager.log_info(f"使用设备: {self.device}")
#         self.output_manager.log_info("=====================================")
        
#         for epoch in range(self.config['epochs']):
#             self.model.train()
#             epoch_feature_loss = 0
#             epoch_gt_loss = 0
#             epoch_total_loss = 0
#             total_batches = len(train_loader)
            
#             for batch_x, batch_y in train_loader:
#                 batch_x = batch_x.to(self.device)
#                 batch_y = batch_y.to(self.device)
                
#                 original_features = batch_x[:, self.model.selected_indices]
                
#                 self.optimizer.zero_grad()
#                 extracted_features, predicted_gt = self.model(batch_x)
                
#                 feature_loss = self.feature_criterion(extracted_features, original_features)
#                 gt_loss = self.gt_criterion(predicted_gt, batch_y)
                
#                 batch_total_loss = 0.9 * feature_loss + 0.1 * gt_loss
                
#                 batch_total_loss.backward()
#                 self.optimizer.step()
                
#                 epoch_feature_loss += feature_loss.item()
#                 epoch_gt_loss += gt_loss.item()
#                 epoch_total_loss += batch_total_loss.item()
            
#             # 计算平均损失
#             avg_feature_loss = epoch_feature_loss / total_batches
#             avg_gt_loss = epoch_gt_loss / total_batches
#             avg_total_loss = epoch_total_loss / total_batches
            
#             # 保存最佳模型
#             if avg_total_loss < best_train_loss:
#                 best_train_loss = avg_total_loss
#                 torch.save(self.model.state_dict(), 
#                           self.output_manager.get_model_path())
            
#             # 输出三种损失
#             self.output_manager.log_info(
#                 f"Epoch [{epoch+1}/{self.config['epochs']}] "
#                 f"Feature Loss: {avg_feature_loss:.6f} "
#                 f"GT Loss: {avg_gt_loss:.6f} "
#                 f"Total Loss: {avg_total_loss:.6f}"
#             )

# file: train_e2e.py (已修复所有问题的最终版本)

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
import os
import traceback

from models.end_to_end_model import EndToEndADASModel
from config import CONFIG

# --- 提升为全局辅助函数 ---
def load_data(file_path):
    """根据文件扩展名智能加载 .npy 或 .txt 文件。"""
    _, ext = os.path.splitext(file_path)
    if ext.lower() == '.npy':
        return np.load(file_path)
    elif ext.lower() == '.txt':
        return np.loadtxt(file_path, skiprows=1)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Please use .npy or .txt.")

def diversity_loss_fn(feat1, feat2, eps=1e-8):
    feat1_norm = torch.nn.functional.normalize(feat1, p=2, dim=1)
    feat2_norm = torch.nn.functional.normalize(feat2, p=2, dim=1)
    cosine_sim = torch.sum(feat1_norm * feat2_norm, dim=1)
    return -torch.mean(cosine_sim)

class EndToEndTrainer:
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

    def prepare_data(self):
        print("\n--- Preparing Data for Self-Supervised Training ---")
        adas1_path = self.e2e_config['adas1']['data_path']
        adas2_path = self.e2e_config['adas2']['data_path']
        batch_size = self.e2e_config['batch_size']

        try:
            print(f"Loading ADAS1 data from: {adas1_path}")
            data1 = load_data(adas1_path) # 使用全局加载函数
            print(f"Loading ADAS2 data from: {adas2_path}")
            data2 = load_data(adas2_path) # 使用全局加载函数
        except Exception as e:
            print(f"\n\n!!! DATA LOADING ERROR !!!\nFailed to load data files: {e}")
            exit() 

        if data1.ndim == 1: data1 = data1.reshape(-1, 1)
        if data2.ndim == 1: data2 = data2.reshape(-1, 1)

        min_len = min(data1.shape[0], data2.shape[0])
        data1, data2 = data1[:min_len], data2[:min_len]
        print(f"Number of aligned samples: {data1.shape[0]}")
        print(f"Input data shape for ADAS1: {data1.shape}, ADAS2: {data2.shape}")

        tensor1, tensor2 = torch.FloatTensor(data1), torch.FloatTensor(data2)
        e2e_dataset = TensorDataset(tensor1, tensor2)
        train_loader = DataLoader(e2e_dataset, batch_size=batch_size, shuffle=True)
        print("--- Data preparation complete. ---")
        return train_loader

    def train(self):
        train_loader = self.prepare_data()
        print("\n============== 自监督端到端训练开始 ==============")
        
        # --- 恢复完整的训练循环 ---
        alpha = self.e2e_config['loss_weights']['alpha']
        beta = self.e2e_config['loss_weights']['beta']
        best_loss = float('inf')

        w1 = self.model.fusion_module.feature1_weight
        w2 = self.model.fusion_module.feature2_weight

        for epoch in range(self.e2e_config['epochs']):
            self.model.train()
            total_loss, total_task_loss, total_div_loss = 0, 0, 0
            
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
                total_task_loss += l_task.item()
                total_div_loss += l_div.item()

            avg_loss = total_loss / len(train_loader)
            avg_task_loss = total_task_loss / len(train_loader)
            avg_div_loss = total_div_loss / len(train_loader)
            
            print(f"Epoch [{epoch+1}/{self.e2e_config['epochs']}] | Total Loss: {avg_loss:.6f} | Task Loss (Self-Supervised): {avg_task_loss:.6f} | Div Loss: {avg_div_loss:.6f}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.model.state_dict(), self.model_path)
                print(f"  -> New best self-supervised model saved with loss: {best_loss:.6f}")
        
        print("\n============== 训练完成 ==============")

def extract_features(config):
    # ... (这个函数保持不变，因为它已经很健壮了) ...
    # ... 只需要确保它也使用全局的 load_data 函数 ...
    print("\n--- Starting Feature Extraction from Trained Model ---")
    
    e2e_config = config['end_to_end']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = "e2e_self_supervised_results"
    model_path = os.path.join(output_dir, "best_e2e_model.pth")

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at '{model_path}'")
        return

    print(f"Loading trained model from: {model_path}")
    model = EndToEndADASModel(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    adas1_path = e2e_config['adas1']['data_path']
    adas2_path = e2e_config['adas2']['data_path']
    data1 = load_data(adas1_path) # 使用全局加载函数
    data2 = load_data(adas2_path) # 使用全局加载函数

    if data1.ndim == 1: data1 = data1.reshape(-1, 1)
    if data2.ndim == 1: data2 = data2.reshape(-1, 1)

    min_len = min(data1.shape[0], data2.shape[0])
    data1, data2 = data1[:min_len], data2[:min_len]

    if data1.shape[1] == 1:
        raw_adas1_distances = data1
        raw_adas2_distances = data2
    else:
        comparison_column_index = 7
        raw_adas1_distances = data1[:, [comparison_column_index]]
        raw_adas2_distances = data2[:, [comparison_column_index]]
    
    combined_raw_distances = np.concatenate([raw_adas1_distances, raw_adas2_distances])
    mean_real_world = combined_raw_distances.mean()
    std_real_world = combined_raw_distances.std()

    tensor1, tensor2 = torch.FloatTensor(data1), torch.FloatTensor(data2)
    inference_dataset = TensorDataset(tensor1, tensor2)
    inference_loader = DataLoader(inference_dataset, batch_size=e2e_config['batch_size'], shuffle=False)
    
    all_fused_features = []
    with torch.no_grad():
        for batch_adas1, batch_adas2 in inference_loader:
            batch_adas1, batch_adas2 = batch_adas1.to(device), batch_adas2.to(device)
            fused_feature_batch, _, _ = model(batch_adas1, batch_adas2)
            all_fused_features.append(fused_feature_batch.cpu().numpy())

    fused_features_normalized = np.vstack(all_fused_features)
    mean_model_world = fused_features_normalized.mean()
    std_model_world = fused_features_normalized.std()
    
    fused_features_rescaled = (fused_features_normalized - mean_model_world) / (std_model_world + 1e-8) * std_real_world + mean_real_world
    
    np.save(os.path.join(output_dir, "fused_feature.npy"), fused_features_rescaled)
    np.save(os.path.join(output_dir, "raw_adas1_distance.npy"), raw_adas1_distances)
    np.save(os.path.join(output_dir, "raw_adas2_distance.npy"), raw_adas2_distances)
    
    print("\n--- Extraction Complete ---")
    print(f"Results saved in folder: '{output_dir}'")
    print(f"  - fused_feature.npy (Shape: {fused_features_rescaled.shape})")
    print(f"  - raw_adas1_distance.npy (Shape: {raw_adas1_distances.shape})")
    print(f"  - raw_adas2_distance.npy (Shape: {raw_adas2_distances.shape})")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Self-Supervised End-to-End ADAS Model")
    parser.add_argument('--mode', type=str, default='train_extract', choices=['train', 'extract', 'train_extract'],
                        help="Choose mode: 'train' to only train, 'extract' to only extract, 'train_extract' to train then extract.")
    args = parser.parse_args()

    try:
        if args.mode == 'train' or args.mode == 'train_extract':
            trainer = EndToEndTrainer(CONFIG)
            trainer.train()
        
        if args.mode == 'extract' or args.mode == 'train_extract':
            extract_features(CONFIG)

    except Exception as e:
        print(f"\n\n!!! AN UNEXPECTED ERROR OCCURRED !!!\nError: {e}")
        traceback.print_exc()