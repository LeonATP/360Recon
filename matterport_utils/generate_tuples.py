import os
import numpy as np

def read_camera_position(file_path):
    with open(file_path, 'r') as file:
        # Read the last three numbers, assuming they are in the first line of the file
        position = list(map(float, file.readline().strip().split()[-3:]))
    return np.array(position)

def calculate_euclidean_distance(pos1, pos2):
    return np.linalg.norm(pos1 - pos2)

def generate_image_list_with_distances(folder_path):
    files = os.listdir(folder_path)
    
    # Filter all frame_XXX.txt files
    txt_files = [f for f in files if f.endswith('.txt')]
    txt_files.sort()
    
    # Extract camera positions and corresponding indices
    camera_positions = []
    for txt_file in txt_files:
        file_path = os.path.join(folder_path, txt_file)
        camera_position = read_camera_position(file_path)
        file_index = int(txt_file.split('_')[1].split('.')[0])
        camera_positions.append((file_index, camera_position))
    
    # Generate result list
    result_list = []
    for i, (index, position) in enumerate(camera_positions):
        distances = []
        for j, (other_index, other_position) in enumerate(camera_positions):
            if i != j:
                distance = calculate_euclidean_distance(position, other_position)
                distances.append((other_index, distance))
        
        # Find the two closest frames
        distances.sort(key=lambda x: x[1])
        #closest_two_indices = [str(distances[0][0]).zfill(3), str(distances[1][0]).zfill(3)]
        closest_two_indices = [str(distances[0][0]).zfill(3),str(distances[1][0]).zfill(3)]
        
        # Build a line of text
        scene_name = os.path.basename(folder_path)
        line = f"{scene_name} {str(index).zfill(3)} {' '.join(closest_two_indices)}"
        result_list.append(line)
    
    return result_list

def process_all_folders(txt_file_path,output_file_path):
    # Read all folder paths from the .txt file
    with open(txt_file_path, 'r') as file:
        folder_paths = file.readlines()

    # Remove newline characters from each line
    folder_paths = [folder.strip() for folder in folder_paths]
    
    # Open output file
    with open(output_file_path, 'w') as output_file:
        # Process each folder and generate results
        for folder_path in folder_paths:
            folder_path = os.path.join("/home/yzm/dataset/Matterport3D/",folder_path)
            if os.path.isdir(folder_path):
                result_list = generate_image_list_with_distances(folder_path)
                # Write results to file
                for line in result_list:
                    output_file.write(line + "\n")
            else:
                print(f"Invalid or non-existent folder path: '{folder_path}'")
                


# Specify the path to the .txt file containing folder paths
txt_file_path = '/home/yzm/Workspace/simplerecon_v2/data_splits/Matterport3d/matterport3d_val.txt'  # Replace with your .txt file path

# Specify output file path
output_file_path = 'data_splits/Matterport3d/val_three_view_deepvmvs.txt'  # Replace with your desired output file path

# Call function to process all folders and save results
process_all_folders(txt_file_path, output_file_path)