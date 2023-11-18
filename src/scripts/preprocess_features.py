import os
import re
import sys
import shutil

def copy_files_to_directories(base_dir, target_root):
    if not os.path.exists(target_root):
        os.makedirs(target_root)
    
    # Dictionary to keep track of the last index for each folder
    last_index = {}

    for dirpath, _, filenames in os.walk(base_dir):
        for filename in filenames:
            # Match the three numbers before the extension
            match = re.search(r'(\d+)_(\d+)(?=\.([^.]+$))', filename)
            if match:
                # Create the target directory name based on the second number
                target_dir_name = match.group(2)
                target_dir_path = os.path.join(target_root, target_dir_name)
                
                # Create target directory if it does not exist
                if not os.path.exists(target_dir_path):
                    os.makedirs(target_dir_path)
                    last_index[target_dir_name] = 0
                
                # Get the next file index for the target directory
                last_index[target_dir_name] += 1
                new_filename = f"{last_index[target_dir_name]}{os.path.splitext(filename)[-1]}"
                
                # Define the source and target paths
                source_path = os.path.join(dirpath, filename)
                target_path = os.path.join(target_dir_path, new_filename)
                
                # Copy the file
                shutil.copy(source_path, target_path)
                print(f"Copied: {source_path} -> {target_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python preprocess_features.py <base_directory> <target_directory>")
    else:
        base_directory = sys.argv[1]
        target_directory = sys.argv[2]
        copy_files_to_directories(base_directory, target_directory)
