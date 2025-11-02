import os
import numpy as np

def read_camera_position(file_path):
    with open(file_path, 'r') as file:
        # 读取前三个数字，假设它们在文件的第一行中
        position = list(map(float, file.readline().strip().split()[:3]))
    return np.array(position)

def calculate_euclidean_distance(pos1, pos2):
    return np.linalg.norm(pos1 - pos2)

def generate_image_list_with_avg_nearest_distance(folder_path):
    files = os.listdir(folder_path)
    
    # 筛选出所有 frame_XXX.txt 文件
    txt_files = [f for f in files if f.endswith('.txt')]
    txt_files.sort()
    
    # 提取相机位置和对应的编号
    camera_positions = []
    for txt_file in txt_files:
        file_path = os.path.join(folder_path, txt_file)
        camera_position = read_camera_position(file_path)
        file_index = int(txt_file.split('_')[1].split('.')[0])
        camera_positions.append((file_index, camera_position))
    
    # 存储每个相机与最近相机的距离
    nearest_distances = []
    
    for i, (index, position) in enumerate(camera_positions):
        nearest_distance = float('inf')  # 初始化为无穷大
        for j, (other_index, other_position) in enumerate(camera_positions):
            if i != j:
                distance = calculate_euclidean_distance(position, other_position)
                if distance < nearest_distance:
                    nearest_distance = distance
        nearest_distances.append(nearest_distance)
    
    # 计算所有相机的最近距离的平均值
    avg_nearest_distance = np.mean(nearest_distances)
    
    # 输出结果
    scene_name = os.path.basename(folder_path)
    line = f"{scene_name} {avg_nearest_distance:.3f}"
    
    return line

def process_all_folders(txt_file_path, output_file_path):
    # 读取 .txt 文件中的所有文件夹路径
    with open(txt_file_path, 'r') as file:
        folder_paths = file.readlines()

    # 去除每一行的换行符
    folder_paths = [folder.strip() for folder in folder_paths]
    
    # 打开输出文件
    with open(output_file_path, 'w') as output_file:
        # 处理每个文件夹并生成结果
        for folder_path in folder_paths:
            folder_path = os.path.join("/home/yzm/dataset/Matterport3D/", folder_path)
            if os.path.isdir(folder_path):
                result_line = generate_image_list_with_avg_nearest_distance(folder_path)
                # 将结果写入文件
                output_file.write(result_line + "\n")
            else:
                print(f"文件夹路径无效或不存在：'{folder_path}'")
                

# 指定包含文件夹路径的 .txt 文件路径
txt_file_path = '/home/yzm/Workspace/simplerecon_v2/data_splits/Matterport3d/matterport3d_test.txt'  # 请将这里替换为你的 .txt 文件路径

# 指定输出文件路径
output_file_path = 'data_splits/Matterport3d/val_avg_nearest_distance.txt'  # 请将这里替换为你希望保存结果的文件路径

# 调用函数处理所有文件夹并保存结果
process_all_folders(txt_file_path, output_file_path)
