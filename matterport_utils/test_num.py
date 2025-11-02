# -*- coding: utf-8 -*-
"""
统计txt文件中某一前缀的行的数量，并保存在另一个txt文件中。
输入文件格式示例：
2t7WUuJeko7 000 008 011
5ZKStnWn8Zo 000 101 084
...

输出文件格式示例：
2t7WUuJeko7 33
5ZKStnWn8Zo 30
...
"""

import collections

def count_prefix_lines(input_file, output_file):
    """
    读取输入文件，统计每个前缀出现的次数，并将结果写入输出文件。

    :param input_file: 输入txt文件路径
    :param output_file: 输出txt文件路径
    """
    # 使用defaultdict来自动处理新前缀
    counter = collections.defaultdict(int)

    try:
        # 打开并读取输入文件
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                # 去除行首尾空白字符
                stripped_line = line.strip()
                if not stripped_line:
                    # 如果是空行，跳过
                    continue
                # 按空白字符分割行内容
                parts = stripped_line.split()
                if len(parts) < 1:
                    print(f"警告：第{line_num}行格式不正确，跳过。")
                    continue
                prefix = parts[0]
                counter[prefix] += 1

        # 打开并写入输出文件
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for prefix, count in counter.items():
                #f_out.write(f"{prefix} {count}\n")
                f_out.write(f"{prefix}\n")

        print(f"统计完成！结果已保存到 '{output_file}'。")

    except FileNotFoundError:
        print(f"错误：输入文件 '{input_file}' 未找到。")
    except Exception as e:
        print(f"发生错误：{e}")

if __name__ == "__main__":
    # 定义输入和输出文件路径
    input_filename = '/home/yzm/Workspace/simplerecon_v2/data_splits/Matterport3d/test_three_view_deepvmvs.txt'   # 请确保input.txt在脚本同一目录下，或者使用绝对路径
    output_filename = '/home/yzm/Workspace/simplerecon_v2/data_splits/Matterport3d/matterport3d_train.txt'

    # 调用函数进行统计
    count_prefix_lines(input_filename, output_filename)