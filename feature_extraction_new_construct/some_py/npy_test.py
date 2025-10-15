#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os

# 在这里设置输入文件路径
input_file = "/home/linux/ShenGang/feature_extraction_new_construct/results/new_log_result/extract/all_adas/motivis_gt=15_6168/features/reduced_feature.npy"  # 请将此处修改为您的NPY文件路径

def check_npy_dimensions(file_path):
    """
    检查 NPY 文件的维度（行数和列数）
    
    参数:
        file_path (str): NPY 文件的路径
    
    返回:
        tuple: (行数, 列数) 或者多维数组的形状
    """
    try:
        # 加载 NPY 文件
        data = np.load(file_path)
        
        # 获取形状
        shape = data.shape
        
        # 打印文件信息
        print(f"文件路径: {file_path}")
        print(f"数据类型: {data.dtype}")
        print(f"数组形状: {shape}")
        
        # 对于 2D 数组，明确打印行数和列数
        if len(shape) == 2:
            rows, cols = shape
            print(f"行数: {rows}")
            print(f"列数: {cols}")
        # 对于 1D 数组
        elif len(shape) == 1:
            print(f"这是一个一维数组，长度为: {shape[0]}")
        # 对于高维数组
        else:
            print(f"这是一个 {len(shape)} 维数组")
            for i, dim in enumerate(shape):
                print(f"维度 {i}: {dim}")
        
        # 返回形状
        return shape
        
    except Exception as e:
        print(f"发生错误: {e}")
        return None

def main():
    # 检查文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 文件 '{input_file}' 不存在")
        return
    
    # 检查文件扩展名
    if not input_file.endswith('.npy'):
        print(f"警告: 文件 '{input_file}' 可能不是 NPY 文件")
    
    # 获取文件维度
    check_npy_dimensions(input_file)

if __name__ == "__main__":
    main()