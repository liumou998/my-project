CONFIG = {
    # # 原有的模型参数
    # 'input_dim': 20,        
    # 'hidden_dims': [16, 8], 
    # 'selected_indices': [6,7],  
    # 'feature_dim': 2,       
    #     # 修改选择的列索引，包含时间戳列
    'input_dim': 20,
    'feature_dim': 4,
    'selected_indices': [7],  # [时间戳列, 特征列1, 特征列2]
    'feature_names': ['feature1'],  # 添加特征名称
    'gt_values': [15],  # 为特征分别设置GT值
    'gt_error_bound': 8.0,  # 保持误差边界为3
    # 训练参数
    'learning_rate': 0.0001, 
    'batch_size': 16,       
    'epochs': 100,                
    'random_state': 42,     
    
    # 原有的数据参数
    'file_paths': [         
        # 'data/process_ADAS2_bike.txt',
        # 'data/process_ADAS2_person1.txt',
        # 'data/process_ADAS2_person2.txt',
        # 'data/process_ADAS2_vehicle.txt',
        # "/home/linux/ShenGang/feature_extraction/data/ADAS1_sync_output.txt",
        # "/home/linux/ShenGang/feature_extraction/data/ADAS1_10m.txt"
        #  "/home/linux/ShenGang/feature_extraction_new_construct/data/jms2_adas3_process.txt"
        # "/home/linux/ShenGang/feature_extraction/data/ADAS2_process.txt"
        # "/home/linux/ShenGang/feature_extraction/data/ADAS1_processed.txt"
        # '/home/linux/ShenGang/feature_extraction_new_construct/data/all_data_process/minieye.txt'
        # '/home/linux/ShenGang/feature_extraction_new_construct/data/all_data_process/me630_process.txt'
        '/home/linux/ShenGang/feature_extraction_new_construct/data/all_data_process/maxeye_process.txt'
        # '/home/linux/ShenGang/feature_extraction_new_construct/data/all_data_process/mit500_process.txt'
        

    ],  
   #fusion配置
    'fusion': {  
        'learning_rate': 0.00008,
        'batch_size': 16,
        'epochs': 100,
        'n_iterations': 4, # 迭代次数
        'feature1_weight': 0.1,
        'feature2_weight': 0.9,
        'hidden_dim': 64,   # 隐藏层维度.
        'feature_paths': {
  # 'feature1': '/home/linux/ShenGang/feature_extraction/results/ADAS1_feature/features/train_features.npy',    # 第一个特征文件路径
            # 'feature2': '/home/linux/ShenGang/feature_extraction/results/ADAS2_feature/features/train_features.npy'     # 第二个特征文件路径
            # 'feature1': '/home/linux/ShenGang/feature_extraction/results/ADAS1_10M/features/train_features.npy',    # 第一个特征文件路径
            # 'feature2': '/home/linux/ShenGang/feature_extraction/results/ADAS2_10M/features/train_features.npy'  
            # 'feature1': '/home/linux/ShenGang/feature_extraction/results/ADAS1_feature_2/features/train_features.npy',    # 第一个特征文件路径
            # 'feature2': '/home/linux/ShenGang/feature_extraction/results/ADAS2_feature_2/features/train_features.npy' 
            # 'feature1': '/home/linux/ShenGang/feature_extraction/results/ADAS1_10m-real/features/train_features.npy',    # 第一个特征文件路径
            # 'feature2': '/home/linux/ShenGang/feature_extraction/results/ADAS2_10m-real/features/train_features.npy' 
            # 'feature1': '/home/linux/ShenGang/feature_extraction/results/ADAS1_GT=10/features/train_features.npy',    # 第一个特征文件路径
            # 'feature2': '/home/linux/ShenGang/feature_extraction/results/ADAS2_GT=10/features/train_features.npy' 

            'feature1': '/home/linux/ShenGang/feature_extraction_new_construct/results/new_log_result/extract/all_adas/maxeye_gt=15_6168/features/reduce6167_features.npy',    # 第一个特征文件路径
            'feature2': '/home/linux/ShenGang/feature_extraction_new_construct/results/new_log_result/extract/all_adas/new_adas2_avg=15.8/reduce6167_features.npy' 



            # 'feature1': '/home/linux/ShenGang/feature_extraction/results/adas_gt=15_all_result_test1/adas1_gt=15_error=3/features/reduced_feature.npy',    # 第一个特征文件路径
            # 'feature2': '/home/linux/ShenGang/feature_extraction/results/adas_gt=15_all_result_test1/adas2_gt=15_error=3/features/train_features.npy' ,
    },
    'fusion_feature_paths': {

            # 'fusion_feature1': '/home/linux/ShenGang/feature_extraction_new_construct/results/20250117_155328/fusion_features/fused_feature1_real.npy',    # 第一个特征文件路径
            # 'fusion_feature2': '/home/linux/ShenGang/feature_extraction_new_construct/results/20250117_155328/fusion_features/fused_feature2_real.npy' ,
            'fusion_feature1': '/home/linux/ShenGang/feature_extraction_new_construct/results/new_log_result/fusion/fusion_test29/fusion_features/fused_feature_real.npy',    # 第一个特征文件路径
            'fusion_feature2': '/home/linux/ShenGang/feature_extraction_new_construct/results/new_log_result/fusion/fusion_test29/fusion_features/fused_feature_real.npy',
    }
},

     'end_to_end': {
        'learning_rate': 0.0005,
        'batch_size': 32,
        'epochs': 120,
        
        'adas1': {
            # 新增: ADAS1的原始数据文件路径
            'data_path': '/home/linux/ShenGang/feature_extraction_new_construct/e2e_fusion_result/maxeye_JMS3_fusion/maxeye_reduce6167_features.npy',
            'selected_indices': [0] # ADAS1 使用的特征列
        },
        'adas2': {
            # 新增: ADAS2的原始数据文件路径
            'data_path': '/home/linux/ShenGang/feature_extraction_new_construct/results/new_log_result/extract/all_adas/motivis_gt=15_6168/features/train_features_6167.npy',
            'selected_indices': [0] # ADAS2 使用的特征列
        },
        
        # # 新增: 固定的Ground Truth值
        # 'ground_truth_value': 15.0,
        
        # 'prediction_head': {
        #     'hidden_dim': 64,
        #     'dropout': 0.3
        # },
        
        'loss_weights': {
            'alpha': 1.0,  # 任务损失(L_task)的权重
            'beta': 0.05   # 多样性损失(L_div)的权重
        }
    }
}


