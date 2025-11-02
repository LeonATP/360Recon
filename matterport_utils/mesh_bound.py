import os
import glob

def process_folders(folder_list_file, output_file):
    """
    处理文件夹中的.txt文件，提取相机位置数据，并计算最大值和最小值。

    :param folder_list_file: 存储文件夹名称的txt文件路径
    :param output_file: 输出结果的txt文件路径
    """
    # 读取文件夹名称
    with open(folder_list_file, 'r', encoding='utf-8') as f:
        folders = [line.strip() for line in f if line.strip()] 
        #folders = [os.path.join('/home/yzm/dataset/Matterport3D/',line.strip()) for line in f if line.strip()]
    
    # 打开输出文件并写入表头
    with open(output_file, 'w', encoding='utf-8') as out_f:
        #out_f.write("Folder\tX_max\tX_min\tY_max\tY_min\tZ_max\tZ_min\n")
        
        # 遍历每个文件夹
        for folder in folders:
            folder_path = os.path.join('/home/yzm/dataset/Matterport3D/',folder) 
            
            if not os.path.isdir(folder_path):
                print(f"警告: 文件夹 '{folder}' 不存在。")
                continue
            
            # 查找文件夹中的所有.txt文件
            txt_files = glob.glob(os.path.join(folder_path, '*.txt'))
            if not txt_files:
                print(f"警告: 文件夹 '{folder}' 中没有找到任何 .txt 文件。")
                continue
            
            x_vals = []
            y_vals = []
            z_vals = []
            
            # 遍历每个.txt文件
            for txt_file in txt_files:
                try:
                    with open(txt_file, 'r', encoding='utf-8') as tf:
                        for line_number, line in enumerate(tf, 1):
                            parts = line.strip().split()
                            if len(parts) < 3:
                                print(f"警告: 文件 '{txt_file}' 的第 {line_number} 行数据不足三列，跳过。")
                                continue
                            try:
                                x, y, z = map(float, parts[:3])
                                x_vals.append(x)
                                y_vals.append(y)
                                z_vals.append(z)
                            except ValueError:
                                print(f"警告: 文件 '{txt_file}' 的第 {line_number} 行包含非浮点数数据，跳过。")
                                continue
                except Exception as e:
                    print(f"错误: 无法读取文件 '{txt_file}'。错误信息: {e}")
                    continue
            
            # 计算最大值和最小值
            if x_vals and y_vals and z_vals:
                x_max = max(x_vals)
                x_min = min(x_vals)
                y_max = max(y_vals)
                y_min = min(y_vals)
                z_max = max(z_vals)
                z_min = min(z_vals)
                
                # 写入结果到输出文件
                out_f.write(f"{folder}\t{x_max}\t{x_min}\t{y_max}\t{y_min}\t{z_max}\t{z_min}\n")
            else:
                print(f"警告: 文件夹 '{folder}' 中没有有效的数据。")

if __name__ == "__main__":
    # 定义文件夹名称列表和输出文件的路径
    folder_list_file = "./data_splits/Matterport3d/matterport3d.txt"  # 请确保此文件存在并包含所有文件夹名称，每行一个
    output_file = "./data_splits/Matterport3d/matterport3d_camera_center.txt"        # 结果将被写入此文件

    # 检查文件夹名称列表文件是否存在
    if not os.path.isfile(folder_list_file):
        print(f"错误: 文件夹列表文件 '{folder_list_file}' 不存在。请确保文件存在并且路径正确。")
    else:
        process_folders(folder_list_file, output_file)
        print(f"处理完成。结果已保存到 '{output_file}'。")