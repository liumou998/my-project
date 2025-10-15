import sys
import os
from shutil import copyfileobj

def merge_logs(input_files, output_file, encoding='utf-8'):
    """
    Merge multiple log files into one in specified order

    Args:
        input_files (list): List of input file paths in desired order
        output_file (str): Output merged file path
        encoding (str): Encoding of the log files (default utf-8)

    Returns:
        bool: True if merge completed successfully
    """
    try:
        with open(output_file, 'wb') as fout:
            for fin_path in input_files:
                if not os.path.isfile(fin_path):
                    print(f"Skipping missing file: {fin_path}", file=sys.stderr)
                    continue
                with open(fin_path, 'rb') as fin:
                    copyfileobj(fin, fout)
        return True
    except Exception as e:
        print(f"Error merging logs: {e}", file=sys.stderr)
        return False

if __name__ == "__main__":
    # ======== 用户需要修改的部分 ========
    # 请在这里填入您的11个日志文件路径（按顺序排列）
    input_files = [
        "/home/linux/ShenGang/data/sys/sys-2025-02-27-09-39-22.log",      # 文件1路径
        "/home/linux/ShenGang/data/sys/sys-2025-02-27-09-40-57.log",      # 文件2路径
        "/home/linux/ShenGang/data/sys/sys-2025-02-27-09-42-20.log",      # 文件3路径
        "/home/linux/ShenGang/data/sys/sys-2025-02-27-09-43-46.log",      # 文件4路径
        "/home/linux/ShenGang/data/sys/sys-2025-02-27-09-45-03.log",      # 文件5路径
        "/home/linux/ShenGang/data/sys/sys-2025-02-27-09-46-20.log",      # 文件6路径
        "/home/linux/ShenGang/data/sys/sys-2025-02-27-09-47-32.log",      # 文件7路径
        "/home/linux/ShenGang/data/sys/sys-2025-02-27-09-48-44.log",      # 文件8路径
        "/home/linux/ShenGang/data/sys/sys-2025-02-27-09-49-47.log",      # 文件9路径
        "/home/linux/ShenGang/data/sys/sys-2025-02-27-09-50-54.log",     # 文件10路径
        "/home/linux/ShenGang/data/sys/sys.log"      # 文件11路径
    ]

    # 设置合并后的输出文件路径
    output_file = "/home/linux/ShenGang/data/sys/combined_sys.log"

    # 可选参数：设置日志编码（如utf-8、gbk等）
    encoding = 'utf-8'

    # 执行合并操作
    success = merge_logs(input_files, output_file, encoding)

    if success:
        print(f"\n✅ 成功合并 {len(input_files)} 个日志文件到 {output_file}")
    else:
        print(f"\n⚠️ 合并失败，请检查错误日志")