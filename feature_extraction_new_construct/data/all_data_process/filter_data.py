# file: filter_data.py (批处理版本)

import os

def filter_txt_file(input_path, output_path, column_index, min_val, max_val):
    """
    读取一个txt文件，根据指定列的数值范围过滤行，并写入新文件。
    """
    print(f"\n--- Processing: {os.path.basename(input_path)} ---")
    
    try:
        with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
            # 1. 读取并写入表头
            header = infile.readline()
            if not header:
                print("Warning: Input file is empty. Skipping.")
                return
            outfile.write(header)
            
            total_lines = 0
            kept_lines = 0
            
            # 2. 逐行处理数据
            for line in infile:
                total_lines += 1
                if not line.strip(): continue

                try:
                    columns = line.strip().split()
                    if len(columns) <= column_index:
                        print(f"  Warning: Skipping line with not enough columns: {line.strip()}")
                        continue
                        
                    value = float(columns[column_index])
                    
                    # 3. 应用过滤条件
                    if min_val <= value <= max_val:
                        outfile.write(line)
                        kept_lines += 1
                        
                except ValueError:
                    print(f"  Warning: Skipping line with non-numeric value in column {column_index + 1}: {line.strip()}")
                except Exception as e:
                    print(f"  An unexpected error occurred on line: {line.strip()}. Error: {e}")

            print(f"  Total data lines processed: {total_lines}")
            print(f"  Lines kept (value between {min_val} and {max_val}): {kept_lines}")
            print(f"  Lines removed: {total_lines - kept_lines}")
            print(f"  Filtered data saved to: {os.path.basename(output_path)}")

    except FileNotFoundError:
        print(f"  Error: Input file not found at '{input_path}'. Skipping.")
    except Exception as e:
        print(f"  An unexpected error occurred while processing {input_path}: {e}")

if __name__ == "__main__":
    print("--- Starting ADAS Data Batch Filtering Script ---\n")
    
    # =================================================================
    # ==================  请在这里填写您的文件路径  ==================
    # =================================================================
    # 这是一个示例列表，请用您自己的七个文件路径替换
    INPUT_FILES = [
        "/home/linux/ShenGang/feature_extraction_new_construct/data/all_data_process/jms2_process.txt",
        "/home/linux/ShenGang/feature_extraction_new_construct/data/all_data_process/maxeye_process.txt",
        "/home/linux/ShenGang/feature_extraction_new_construct/data/all_data_process/minieye.txt",
        "/home/linux/ShenGang/feature_extraction_new_construct/data/all_data_process/me630_process.txt",
        "/home/linux/ShenGang/feature_extraction_new_construct/data/all_data_process/mitaio_process.txt",
        "/home/linux/ShenGang/feature_extraction_new_construct/data/all_data_process/mit500_process.txt",
        "/home/linux/ShenGang/feature_extraction_new_construct/data/all_data_process/motovis_process.txt",
    ]
    # =================================================================
    # =================================================================
    
    # --- 过滤参数 ---
    COLUMN_TO_CHECK = 7  # 第八列 (索引从0开始)
    MIN_VALUE = 10.0
    MAX_VALUE = 30.0
    
    print(f"Filtering condition: Column {COLUMN_TO_CHECK + 1} must have a value between [{MIN_VALUE}, {MAX_VALUE}]\n")
    
    # 循环处理文件列表中的每个文件
    for input_file_path in INPUT_FILES:
        # 自动生成输出文件名
        base, ext = os.path.splitext(input_file_path)
        output_file_path = f"{base}_filtered{ext}"
        
        # 调用过滤函数
        filter_txt_file(input_file_path, output_file_path, COLUMN_TO_CHECK, MIN_VALUE, MAX_VALUE)
        
    print("\n--- Batch filtering process finished. ---")