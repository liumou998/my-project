import csv
import os


file_path = '/home/linux/ShenGang/feature_extraction_new_construct/process/data_process/dataprocess/adas1_aligned.txt'
error_count = 0

with open(file_path, 'r', encoding='utf-8') as f:
   for i, line in enumerate(f, 1):
       dims = len(line.strip().split())
       if dims != 20:
           print(f"{i}: {dims}")
           error_count += 1

if error_count == 0:
   print("维度检查正常，所有行都是20维")