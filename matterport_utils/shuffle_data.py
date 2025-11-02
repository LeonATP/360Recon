import random

def shuffle_training_data(input_file, output_file, seed=None):
    """
    读取输入文件，打乱行顺序，并将结果写入输出文件。

    :param input_file: 原始训练集文件路径
    :param output_file: 打乱后训练集文件路径
    :param seed: 随机种子（可选，用于结果可重复）
    """
    # 如果提供了种子，设置随机种子以确保结果可重复
    if seed is not None:
        random.seed(seed)

    # 读取所有行
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 打乱行顺序
    random.shuffle(lines)

    # 将打乱后的行写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    print(f"已将 {input_file} 中的行顺序打乱，并保存到 {output_file}。")

if __name__ == "__main__":
    # 设置输入和输出文件路径
    input_txt = 'data_splits/Matterport3d/train_three_view_deepvmvs.txt'          # 替换为你的原始文件路径
    output_txt = 'data_splits/Matterport3d/train_three_view_deepvmvs_R.txt'  # 替换为你想要保存的文件路径

    # 可选：设置随机种子以确保每次打乱的顺序相同
    random_seed = 42  # 你可以选择任意整数，或者设置为 None 以获得不同的打乱结果

    # 调用打乱函数
    shuffle_training_data(input_txt, output_txt, seed=random_seed)