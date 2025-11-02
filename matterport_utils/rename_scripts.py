import os

def rename_images(scan_path):
    # 获取文件夹内所有文件名
    files = os.listdir(scan_path)
    
    # 筛选出所有以 _rgb.png 为后缀的文件
    rgb_files = [os.path.join(scan_path,f) for f in files if f.endswith('_rgb.png')]
    
    # 对文件名排序，以确保按顺序重命名
    rgb_files.sort()
    
    # 逐个文件重命名
    for i, rgb_file_name in enumerate(rgb_files):
        # 生成新的基础文件名
        new_base_name = f"frame_{i:03d}"
        
        # 构建对应文件的旧路径和新路径
        related_extensions = ['_rgb.png', '_vis.png', '_depth.dpt', '_pose.txt']
        
        for ext in related_extensions:
            old_file_name = rgb_file_name.replace('_rgb.png', ext)
            if ext == '_vis.png':
               new_file_name = new_base_name + ext
            if ext == '_depth.dpt':
                new_file_name = new_base_name +'.dpt'
            if ext=='_pose.txt':
               new_file_name = new_base_name +'.txt'
            if ext == '_rgb.png':
                new_file_name= new_base_name + '.png' 
            
            old_file_path = os.path.join(scan_path, old_file_name)
            new_file_path = os.path.join(scan_path, new_file_name)
            
            # 检查文件是否存在，如果存在则重命名
            if os.path.exists(old_file_path):
                os.rename(old_file_path, new_file_path)
    
    print(f"重命名完成！文件夹 '{folder_path}' 内处理了 {len(rgb_files)} 个文件组。")

# 指定文件夹路径
dataset_path = '/home/yzm/dataset/Matterport3D/'  # 请将这里替换为你的文件夹路径
txt_file_path = '/home/yzm/Workspace/simplerecon/data_splits/Matterport3d/matterport3d.txt' 
# 读取 .txt 文件中的所有文件夹路径
with open(txt_file_path, 'r') as file:
    folder_paths = file.readlines()

# 去除每一行的换行符
folder_paths = [folder.strip() for folder in folder_paths]

# 遍历每个文件夹路径
for folder_path in folder_paths:
  scan_path= os.path.join(dataset_path,folder_path)
  # 调用函数重命名文件
  rename_images(scan_path)