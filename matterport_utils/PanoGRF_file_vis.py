# -*- coding: utf-8 -*-
"""
根据统计文件重新划分源文件夹中的.jpg文件。

假设：
- 源文件夹包含.jpg文件，命名格式为8_depth_pred.jpg（即一个以数字为前缀，后接_depth_pred.jpg的文件）。
- 统计文件count.txt中每行包含一个前缀和对应的数量。

操作：
- 将源文件夹中的.jpg文件按顺序划分到对应的前缀文件夹中。
- 在目标文件夹中创建前缀文件夹，并将.jpg文件重命名为000.jpg, 001.jpg等。
"""

import os
import shutil
import re

def read_counts(count_file):
    """
    读取统计文件，返回一个列表，包含(前缀, 数量)的元组。

    :param count_file: 统计文件路径
    :return: List of tuples [(prefix1, count1), (prefix2, count2), ...]
    """
    counts = []
    with open(count_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            stripped_line = line.strip()
            if not stripped_line:
                continue  # 跳过空行
            parts = stripped_line.split()
            if len(parts) != 2:
                print(f"警告：第{line_num}行格式不正确，跳过。内容: {line}")
                continue
            prefix, count_str = parts
            try:
                count = int(count_str)
                counts.append((prefix, count))
            except ValueError:
                print(f"警告：第{line_num}行数量不是整数，跳过。内容: {line}")
    return counts

def get_sorted_jpg_files(source_dir):
    """
    获取源文件夹中所有.jpg文件的排序列表。

    :param source_dir: 源文件夹路径
    :return: Sorted list of .jpg file names
    """
    jpg_files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f)) and f.endswith('.jpg')]
    
    # 使用正则表达式提取前缀数字并排序
    def extract_number(filename):
        match = re.match(r"(\d+)_depth_pred\.jpg", filename)
        if match:
            return int(match.group(1))
        else:
            print(f"警告：文件名格式不符合预期，跳过: {filename}")
            return -1  # 非法文件将被排序到前面
    
    jpg_files_sorted = sorted(jpg_files, key=lambda x: extract_number(x))
    
    # 过滤掉提取数字失败的文件
    jpg_files_sorted = [f for f in jpg_files_sorted if extract_number(f) != -1]
    
    return jpg_files_sorted

def create_and_move_jpg_files(source_dir, destination_dir, counts, jpg_files_sorted):
    """
    根据统计结果创建目标文件夹，并移动重命名.jpg文件。

    :param source_dir: 源文件夹路径
    :param destination_dir: 目标文件夹路径
    :param counts: List of tuples [(prefix1, count1), (prefix2, count2), ...]
    :param jpg_files_sorted: Sorted list of .jpg file names
    """
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
        print(f"创建目标文件夹: {destination_dir}")

    total_required = sum(count for _, count in counts)
    total_available = len(jpg_files_sorted)
    if total_required != total_available:
        print(f"警告：统计文件中总数量({total_required})与源文件夹中的.jpg文件数量({total_available})不匹配。")
        proceed = input("是否继续执行？输入'y'继续，其他键取消: ")
        if proceed.lower() != 'y':
            print("操作已取消。")
            return

    current_index = 0
    for prefix, count in counts:
        prefix_dir = os.path.join(destination_dir, prefix)
        os.makedirs(prefix_dir, exist_ok=True)
        print(f"处理前缀 '{prefix}'，数量: {count}")

        for i in range(count):
            if current_index >= total_available:
                print("警告：源文件已用完，但统计文件中仍有未处理的条目。")
                break

            src_file_name = jpg_files_sorted[current_index]
            src_file_path = os.path.join(source_dir, src_file_name)

            # 新的文件名称，格式为三位数，如000.jpg, 001.jpg, ..., 999.jpg
            new_file_name = f"{i:03}.jpg"
            dest_file_path = os.path.join(prefix_dir, new_file_name)

            try:
                shutil.move(src_file_path, dest_file_path)
                print(f"移动并重命名: {src_file_name} -> {prefix}/{new_file_name}")
            except Exception as e:
                print(f"错误：无法移动 {src_file_name} 到 {prefix}/{new_file_name}。原因: {e}")

            current_index += 1

    print("所有操作完成！")

def main():
    # 定义文件路径
    count_file = '/home/yzm/Workspace/simplerecon_v2/data_splits/Matterport3d/test_num.txt'        # 统计文件路径
    source_dir = '/home/yzm/Workspace/PanoGRF/logs/mvsdepth/test_images_1.0'       # 源文件夹路径，包含8_depth_pred.jpg文件
    destination_dir = '/home/yzm/Workspace/PanoGRF/logs/mvsdepth/new_vis_v2'  # 目标文件夹路径

    # 检查文件和文件夹是否存在
    if not os.path.isfile(count_file):
        print(f"错误：统计文件 '{count_file}' 不存在。请确保文件路径正确。")
        return
    if not os.path.isdir(source_dir):
        print(f"错误：源文件夹 '{source_dir}' 不存在。请确保文件夹路径正确。")
        return

    # 读取统计文件
    counts = read_counts(count_file)
    if not counts:
        print("错误：统计文件中没有有效的数据。")
        return

    # 获取排序后的源.jpg文件列表
    jpg_files_sorted = get_sorted_jpg_files(source_dir)
    if not jpg_files_sorted:
        print("错误：源文件夹中没有符合命名格式的.jpg文件。")
        return

    # 执行移动和重命名操作
    create_and_move_jpg_files(source_dir, destination_dir, counts, jpg_files_sorted)

if __name__ == "__main__":
    main()
