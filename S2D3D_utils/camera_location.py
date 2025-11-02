import os
import json

def extract_camera_location(json_file):
    """
    Extract camera_location from JSON file.
    """
    try:
        with open(json_file, 'r') as file:
            data = json.load(file)
            camera_location = data.get("camera_location")
            if camera_location:
                # Return x, y, z coordinates from camera_location
                return camera_location
            else:
                print(f"No camera_location found in {json_file}")
                return None
    except Exception as e:
        print(f"Error reading {json_file}: {e}")
        return None

def process_json_folder(folder_path):
    """
    Process all JSON files in the folder, extract x, y, z coordinates from all camera_location.
    """
    # Get all json files in the folder
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    
    # Initialize extremes to positive and negative infinity
    xmin, ymin, zmin = float('inf'), float('inf'), float('inf')
    xmax, ymax, zmax = float('-inf'), float('-inf'), float('-inf')

    for json_file in json_files:
        json_path = os.path.join(folder_path, json_file)
        camera_location = extract_camera_location(json_path)
        
        if camera_location:
            x, y, z = camera_location
            # Update min and max values for x, y, z
            xmin = min(xmin, x)
            ymin = min(ymin, y)
            zmin = min(zmin, z)
            xmax = max(xmax, x)
            ymax = max(ymax, y)
            zmax = max(zmax, z)
    
    return xmin, ymin, zmin, xmax, ymax, zmax

if __name__ == "__main__":
    folder_path = "/home/yzm/dataset/S2D3D/area_5b/pose"  # Replace with your folder path
    xmin, ymin, zmin, xmax, ymax, zmax = process_json_folder(folder_path)
    
    print(f"Minimum and Maximum values of camera_location across all JSON files:")
    print(f"xmin: {xmin}, xmax: {xmax}")
    print(f"ymin: {ymin}, ymax: {ymax}")
    print(f"zmin: {zmin}, zmax: {zmax}")
