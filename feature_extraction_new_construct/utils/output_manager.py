# utils/output_manager.py
import os
import datetime
import logging

class OutputManager:
    def __init__(self, base_dir='results'):
        # 创建时间戳子目录
        self.timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.base_dir = base_dir
        self.run_dir = os.path.join(base_dir, self.timestamp)
        
        # 创建必要的子目录
        self.models_dir = os.path.join(self.run_dir, 'models')
        self.features_dir = os.path.join(self.run_dir, 'features')
        self.logs_dir = os.path.join(self.run_dir, 'logs')
        self.fusion_dir = os.path.join(self.run_dir, 'fusion_features')  # 新增fusion特征目录
        
        self._create_directories()
        self._setup_logger()
    
    def _create_directories(self):
        """创建所需的目录结构"""
        directories = [
            self.models_dir,
            self.features_dir,
            self.logs_dir,
            self.fusion_dir  # 添加fusion目录
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def _setup_logger(self):
        """设置日志记录器"""
        log_file = os.path.join(self.logs_dir, 'training.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def get_model_path(self, model_name='best_model.pth'):
        """获取模型保存路径"""
        return os.path.join(self.models_dir, model_name)

    
    def get_fusion_model_path(self, model_name='fusion_model.pth'):
        """获取特征融合模型保存路径"""
        return os.path.join(self.models_dir, model_name)
    
    def get_feature_path(self, feature_name):
        """获取特征保存路径"""
        return os.path.join(self.features_dir, feature_name)
    
    
    def get_fusion_feature_path(self, feature_name):
        """获取融合特征保存路径"""
        return os.path.join(self.fusion_dir, feature_name)
    
    def get_latest_feature_path(self, feature_name):
        """获取最新的特征文件路径"""
        subdirs = [d for d in os.listdir(self.base_dir) 
                  if os.path.isdir(os.path.join(self.base_dir, d))]
        subdirs.sort(reverse=True)
        
        for subdir in subdirs:
            feature_path = os.path.join(self.base_dir, subdir, 'features', feature_name)
            if os.path.exists(feature_path):
                return feature_path
        return None
    
    def get_latest_fusion_feature_path(self, feature_name):
        """获取最新的融合特征文件路径"""
        subdirs = [d for d in os.listdir(self.base_dir) 
                  if os.path.isdir(os.path.join(self.base_dir, d))]
        subdirs.sort(reverse=True)
        
        for subdir in subdirs:
            feature_path = os.path.join(self.base_dir, subdir, 'fusion_features', feature_name)
            if os.path.exists(feature_path):
                return feature_path
        return None
    
    def log_info(self, message):
        """记录信息到日志"""
        self.logger.info(message)
        
    def log_error(self, message):
        """记录错误信息到日志"""
        self.logger.error(message)