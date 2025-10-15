def remove_invalid_lines(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            if "mobileye-frameid" not in line:
                outfile.write(line)

# 使用方法
input_file = "/home/linux/ShenGang/feature_extraction_new_construct/process/data_process/mitaio_minieye_me630_process/ADAS1.txt"  # 替换为你的输入文件路径
output_file = "/home/linux/ShenGang/feature_extraction_new_construct/process/data_process/mitaio_minieye_me630_process/ADAS1_detele2.txt"  # 替换为你希望的输出文件路径
remove_invalid_lines(input_file, output_file)
