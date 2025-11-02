import os
import numpy as np
import json

def read_camera_position(file_path):
    with open(file_path, 'r') as file:
        # Read camera_location from JSON file
        data = json.load(file)
        position = data["camera_location"]
    return np.array(position)

def calculate_euclidean_distance(pos1, pos2):
    return np.linalg.norm(pos1 - pos2)

def calculate_all_frames_avg_nearest_distance(folder_paths):
    all_nearest_distances = []
    
    for folder_path in folder_paths:
        pose_path = os.path.join(folder_path, "pose")
        files = os.listdir(pose_path)
        
        # Filter all _pose.json files
        json_files = [f for f in files if f.endswith('.json')]
        json_files.sort()
        
        # Extract camera positions and corresponding indices
        camera_positions = []
        for json_file in json_files:
            file_path = os.path.join(pose_path, json_file)
            camera_position = read_camera_position(file_path)
            file_index = int(json_file.split('_')[1].split('.')[0])
            camera_positions.append((file_index, camera_position))
        
        # Store distance from each camera to its nearest camera
        for i, (index, position) in enumerate(camera_positions):
            nearest_distance = float('inf')  # Initialize to infinity
            for j, (other_index, other_position) in enumerate(camera_positions):
                if i != j:
                    distance = calculate_euclidean_distance(position, other_position)
                    if distance < nearest_distance:
                        nearest_distance = distance
            all_nearest_distances.append(nearest_distance)
    
    # Calculate average of nearest distances for all cameras
    avg_nearest_distance = np.mean(all_nearest_distances)
    
    return avg_nearest_distance

def process_all_folders(txt_file_path, output_file_path):
    # Read all folder paths from the .txt file
    with open(txt_file_path, 'r') as file:
        folder_paths = file.readlines()

    # Remove newline characters from each line
    folder_paths = [folder.strip() for folder in folder_paths]
    
    # Map folder paths to full paths
    full_folder_paths = [os.path.join("/home/yzm/dataset/S2D3D/", folder) for folder in folder_paths]
    
    # Calculate average nearest distance for all cameras
    avg_distance = calculate_all_frames_avg_nearest_distance(full_folder_paths)
    
    # Open output file and save results
    with open(output_file_path, 'w') as output_file:
        output_file.write(f"Average nearest distance for all frames: {avg_distance:.3f}\n")

# Specify the path to the .txt file containing folder paths
txt_file_path = '/home/yzm/Workspace/simplerecon_v2/data_splits/S2D3D/S2D3D_test.txt'  # Replace with your .txt file path

# Specify output file path
output_file_path = 'data_splits/S2D3D/test_avg_nearest_distance_all_frames.txt'  # Replace with your desired output file path

# Call function to process all folders and save results
process_all_folders(txt_file_path, output_file_path)
