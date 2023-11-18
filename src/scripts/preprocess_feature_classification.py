import os
import shutil
import sys

def copy_files_with_char(source_dir, target_dir, char, position):
    """
    Copy files from source_dir to target_dir if they contain the character 'char' 
    at the 'position' after the '_' character in their name. The directory structure 
    of the source directory is preserved in the target directory.
    """
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            # Find the position of '_' in the filename
            underscore_index = file.find('_')

            # Check if the character at the given position after '_' is the specified character
            if underscore_index != -1 and underscore_index + position < len(file) and file[underscore_index + position] == char:
                source_file = os.path.join(root, file)
                # Calculate the relative path to maintain the directory structure
                relative_path = os.path.relpath(root, source_dir)
                target_file_dir = os.path.join(target_dir, relative_path)

                if not os.path.exists(target_file_dir):
                    os.makedirs(target_file_dir)

                target_file = os.path.join(target_file_dir, file)

                # Copy the file to the target directory
                shutil.copy2(source_file, target_file)
                print(f"Copied: {source_file} to {target_file}")


character = '1'

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python preprocess_feature_classification.py <base_directory> <target_directory> <position_of_1>")
    else:
        source_directory = sys.argv[1]
        target_directory = sys.argv[2]
        position_after_underscore = int(sys.argv[3])
        copy_files_with_char(source_directory, target_directory, character, position_after_underscore)
