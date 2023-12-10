import os
import re
import sys
import shutil
import random
from datetime import datetime

DATA_COUNT_FOR_ONE_CAT = 500

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
                #target_dir_name = match.group(2)
                # Uzyskaj ścieżkę do folderu, w którym znajduje się plik
                #target_dir_cat = os.path.dirname(filename)
                #print(target_dir_cat)
                # Uzyskaj nazwę folderu
                target_dir_cat = os.path.basename(dirpath)
                print(target_dir_cat)
                #target_dir_cat = match.group(1)
                target_dir_subcat = match.group(2)
                #target_dir_path = os.path.join(target_root, target_dir_name)
                target_dir_path = os.path.join(target_root, target_dir_cat)
                target_dir_path = os.path.join(target_dir_path, target_dir_subcat)
                
                # Create target directory if it does not exist
                if not os.path.exists(target_dir_path):
                    os.makedirs(target_dir_path)
                    last_index[target_dir_path] = 0
                
                # Get the next file index for the target directory
                last_index[target_dir_path] += 1
                new_filename = f"{last_index[target_dir_path]}{os.path.splitext(filename)[-1]}"
                
                # Define the source and target paths
                source_path = os.path.join(dirpath, filename)
                target_path = os.path.join(target_dir_path, new_filename)
                
                # Copy the file
                shutil.copy(source_path, target_path)
                print(f"Copied: {source_path} -> {target_path}")

def chooseOneFrom1xx(category_dir):
    choose_dir_path = 'nieistniejacyplik'
    while not os.path.isdir(choose_dir_path):
        flags = '1' + str(random.randint(0, 1)) + str(random.randint(0, 1))
        choose_dir_path = os.path.join(category_dir, flags)
    lst = os.listdir(choose_dir_path)
    id = random.randint(1, len(lst))
    filejpg =  os.path.join(choose_dir_path, str(id) + '.jpg')
    filejpeg = os.path.join(choose_dir_path, str(id) + '.jpeg')
    if os.path.isfile(filejpg):
        return filejpg, '.jpg'
    return filejpeg, '.jpeg'

def chooseOneFromx1x(category_dir):
    choose_dir_path = 'nieistniejacyplik'
    while not os.path.isdir(choose_dir_path):
        flags = str(random.randint(0, 1)) + '1' + str(random.randint(0, 1))
        choose_dir_path = os.path.join(category_dir, flags)
    lst = os.listdir(choose_dir_path)
    id = random.randint(1, len(lst))
    filejpg =  os.path.join(choose_dir_path, str(id) + '.jpg')
    filejpeg = os.path.join(choose_dir_path, str(id) + '.jpeg')
    if os.path.isfile(filejpg):
        return filejpg, '.jpg'
    return filejpeg, '.jpeg'

def chooseOneFromxx1(category_dir):
    choose_dir_path = 'nieistniejacyplik'
    while not os.path.isdir(choose_dir_path):
        flags = str(random.randint(0, 1)) + str(random.randint(0, 1)) + '1'
        choose_dir_path = os.path.join(category_dir, flags)
    lst = os.listdir(choose_dir_path)
    id = random.randint(1, len(lst))
    filejpg =  os.path.join(choose_dir_path, str(id) + '.jpg')
    filejpeg = os.path.join(choose_dir_path, str(id) + '.jpeg')
    if os.path.isfile(filejpg):
        return filejpg , '.jpg'
    return filejpeg, '.jpeg'

def createFinalDataset(tmpDatasetDir, target_root):
    for category in os.listdir(tmpDatasetDir):
        source_category_dir = os.path.join(tmpDatasetDir, category)
        target_category_dir = os.path.join(target_root, category)
        if not os.path.exists(target_category_dir):
            os.makedirs(target_category_dir)
        random.seed(datetime.now().timestamp())
        for i in range(DATA_COUNT_FOR_ONE_CAT):
            target_dir = os.path.join(target_category_dir, str(i))
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            source_path_1xx, ext1xx = chooseOneFrom1xx(source_category_dir)
            source_path_x1x, extx1x = chooseOneFromx1x(source_category_dir)
            source_path_xx1, extxx1 = chooseOneFromxx1(source_category_dir)

            target_path_1xx = os.path.join(target_dir, '1xx' + ext1xx)
            target_path_x1x = os.path.join(target_dir, 'x1x' + extx1x)
            target_path_xx1 = os.path.join(target_dir, 'xx1' + extxx1)

            shutil.copy(source_path_1xx, target_path_1xx)
            shutil.copy(source_path_x1x, target_path_x1x)
            shutil.copy(source_path_xx1, target_path_xx1)





if __name__ == "__main__":
    random.seed(datetime.now().timestamp())
    if len(sys.argv) < 3:
        print("Usage: python preprocess_features.py <base_directory> <target_directory>")
    else:
        base_directory = sys.argv[1]
        tmp_dir = './tmpdirfordatabase'
        target_directory = sys.argv[2]
        copy_files_to_directories(base_directory, tmp_dir)
        createFinalDataset(tmp_dir, target_directory)
        shutil.rmtree(tmp_dir)