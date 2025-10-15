# main.py
import torch
import numpy as np
import argparse
from models.supervised_extractor import SupervisedFeatureExtractor
from utils.data_loader import DataLoader
from utils.output_manager import OutputManager
from train import Trainer
from extract import extract_features
from config import CONFIG
from fusion.train_fusion import fusion_features  # 更新导入路径

def feature_extraction():
    # 初始化输出管理器
    output_manager = OutputManager()
    output_manager.log_info("开始特征提取任务")
    
    # 数据加载
    output_manager.log_info("正在加载数据...")
    data_loader = DataLoader(CONFIG)
    train_data, train_labels = data_loader.prepare_data()
    
    # 初始化模型
    output_manager.log_info("初始化模型...")
    model = SupervisedFeatureExtractor(
        input_dim=CONFIG['input_dim'],
        feature_dim=CONFIG['feature_dim'],
        selected_indices=CONFIG['selected_indices']
    )
    
    # 训练模型
    output_manager.log_info("开始训练模型...")
    trainer = Trainer(model, CONFIG, output_manager)
    trainer.train(train_data, train_labels)
    
    # 加载最佳模型
    output_manager.log_info("加载最佳模型...")
    model.load_state_dict(torch.load(output_manager.get_model_path()))
    
    # 提取特征
    output_manager.log_info("提取特征...")
    train_features_df = extract_features(model, train_data)
    
    # 保存特征
    output_manager.log_info("保存特征...")
    # 保存为 npy 格式供后续处理
    np.save(output_manager.get_feature_path('train_features.npy'), train_features_df.values)
    
    # 保存为 csv 格式方便查看
    train_features_df.to_csv(output_manager.get_feature_path('train_features.csv'), index=False)
    
    output_manager.log_info("特征提取完成！")
    output_manager.log_info(f"训练特征形状: {train_features_df.shape}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='extract', 
                      choices=['extract', 'fusion'],
                      help='extract: 特征提取, fusion: 特征融合')
    args = parser.parse_args()

    if args.mode == 'extract':
        print("Starting feature extraction...")
        feature_extraction()
        
    elif args.mode == 'fusion':
        print("Starting feature fusion...")
        fusion_features()
        print("Feature fusion completed!")

if __name__ == "__main__":
    main()