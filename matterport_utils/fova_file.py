# -*- coding: utf-8 -*-
"""
根据统计文件重新划分源文件夹中的子文件夹。

假设：
- 源文件夹包含子文件夹00000至001849。
- 统计文件count.txt中每行包含一个前缀和对应的数量。

操作：
- 将源文件夹中的子文件夹按顺序划分到对应的前缀文件夹中。
- 在目标文件夹中创建前缀文件夹，并将子文件夹重命名为000、001等。
"""

import os
import shutil

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

def get_sorted_folders(source_dir):
    """
    获取源文件夹中所有子文件夹的排序列表。

    :param source_dir: 源文件夹路径
    :return: Sorted list of folder names
    """
    folders = [f for f in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, f))]
    # 假设文件夹名都是数字，进行排序
    try:
        folders_sorted = sorted(folders, key=lambda x: int(x))
    except ValueError:
        print("错误：源文件夹中存在非数字命名的子文件夹。")
        raise
    return folders_sorted

def create_and_move_folders(source_dir, destination_dir, counts, folders_sorted):
    """
    根据统计结果创建目标文件夹，并移动重命名子文件夹。

    :param source_dir: 源文件夹路径
    :param destination_dir: 目标文件夹路径
    :param counts: List of tuples [(prefix1, count1), (prefix2, count2), ...]
    :param folders_sorted: Sorted list of source folder names
    """
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
        print(f"创建目标文件夹: {destination_dir}")

    total_required = sum(count for _, count in counts)
    total_available = len(folders_sorted)
    if total_required != total_available:
        print(f"警告：统计文件中总数量({total_required})与源文件夹数量({total_available})不匹配。")
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
                print("警告：源文件夹已用完，但统计文件中仍有未处理的条目。")
                break

            src_folder_name = folders_sorted[current_index]
            src_folder_path = os.path.join(source_dir, src_folder_name)

            # 新的文件夹名称，格式为三位数，如000, 001, ..., 034
            new_folder_name = f"{i:03}"
            dest_folder_path = os.path.join(prefix_dir, new_folder_name)

            try:
                shutil.move(src_folder_path, dest_folder_path)
                print(f"移动并重命名: {src_folder_name} -> {prefix}/{new_folder_name}")
            except Exception as e:
                print(f"错误：无法移动 {src_folder_name} 到 {prefix}/{new_folder_name}。原因: {e}")

            current_index += 1

    print("所有操作完成！")

def main():
    # 定义文件路径
    count_file = '/home/yzm/Workspace/simplerecon_v2/data_splits/Matterport3d/test_num.txt'        # 统计文件路径
    source_dir = '/home/yzm/Workspace/fova-depth/test_logs/matterport360_erp/lightning_logs/version_1/visuals/'       # 源文件夹路径，包含00000至001849的子文件夹
    destination_dir = '/home/yzm/Workspace/fova-depth/test_logs/matterport360_erp/lightning_logs/version_1/visuals/new/'  # 目标文件夹路径

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

    # 获取排序后的源文件夹列表
    try:
        folders_sorted = get_sorted_folders(source_dir)
    except Exception as e:
        print(f"错误：无法获取源文件夹列表。原因: {e}")
        return

    # 执行移动和重命名操作
    create_and_move_folders(source_dir, destination_dir, counts, folders_sorted)

if __name__ == "__main__":
    main()