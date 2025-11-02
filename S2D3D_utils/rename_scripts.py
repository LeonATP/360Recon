import os

def rename_images(scan_path):
    # Get subfolder paths
    rgb_path = os.path.join(scan_path, 'rgb')
    pose_path = os.path.join(scan_path, 'pose')
    depth_path = os.path.join(scan_path, 'depth')
    
    # Get all _rgb.png file names in RGB folder
    rgb_files = [f for f in os.listdir(rgb_path) if f.endswith('_rgb.png')]
    
    # Sort RGB file names
    rgb_files.sort()
    
    # Rename files one by one
    for i, rgb_file_name in enumerate(rgb_files):
        # Generate new base file name
        new_base_name = f"frame_{i:03d}"
        
        # Build old and new paths for RGB file
        old_rgb_path = os.path.join(rgb_path, rgb_file_name)
        new_rgb_name = new_base_name + '.png'
        new_rgb_path = os.path.join(rgb_path, new_rgb_name)
        
        # Rename RGB file
        if os.path.exists(old_rgb_path):
            os.rename(old_rgb_path, new_rgb_path)
        
        # Build old and new paths for corresponding pose file
        old_pose_name = rgb_file_name.replace('_rgb.png', '_pose.json')
        old_pose_path = os.path.join(pose_path, old_pose_name)
        new_pose_name = new_base_name + '.json'
        new_pose_path = os.path.join(pose_path, new_pose_name)
        
        # Rename pose file
        if os.path.exists(old_pose_path):
            os.rename(old_pose_path, new_pose_path)
        
        # Build old and new paths for corresponding depth file
        old_depth_name = rgb_file_name.replace('_rgb.png', '_depth.png')
        old_depth_path = os.path.join(depth_path, old_depth_name)
        new_depth_name = new_base_name + '.png'
        new_depth_path = os.path.join(depth_path, new_depth_name)
        
        # Rename depth file
        if os.path.exists(old_depth_path):
            os.rename(old_depth_path, new_depth_path)

    print(f"Rename complete! Processed {len(rgb_files)} file groups in folder '{scan_path}'.")

# Specify folder path
dataset_path = '/home/yzm/dataset/S2D3D/'  # Replace with your folder path
txt_file_path = '/home/yzm/Workspace/simplerecon_v2/data_splits/S2D3D/S2D3D_test.txt' 

# Read all folder paths from the .txt file
with open(txt_file_path, 'r') as file:
    folder_paths = file.readlines()

# Remove newline characters from each line
folder_paths = [folder.strip() for folder in folder_paths]

# Iterate through each folder path
for folder_path in folder_paths:
    scan_path = os.path.join(dataset_path, folder_path)
    rename_images(scan_path)