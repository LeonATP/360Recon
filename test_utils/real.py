# Generate txt file
def generate_frame_file():
    frames = 39  # Number of frames
    prefix = "EValley2"
    
    with open("frame_data.txt", "w") as f:
        for i in range(frames):
            # Current frame number
            current_frame = str(i).zfill(3)
            # Previous frame number
            prev_frame = str((i - 1) % frames).zfill(3)  # Get previous frame in a cycle
            # Next frame number
            next_frame = str((i + 1) % frames).zfill(3)  # Get next frame in a cycle
            
            # Format and write data
            f.write(f"{prefix} {current_frame} {prev_frame} {next_frame}\n")

# Execute file generation
generate_frame_file()
