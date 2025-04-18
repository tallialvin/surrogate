import os
import shutil
import re
import random

# Variables for easy modification
dataset_prefix = 'cs3'
source_folder = '/home/emu/Documents/surrogate/dataset/'+str(dataset_prefix)
base_destination = '/home/emu/Documents/surrogate/dataset'

first_set_count = 2000
second_set_count = 1000
random_set_count = 500

# Function to extract number from filename
def get_number(filename):
    match = re.search(r'data-set-(\d+)\.pkl', filename)
    return int(match.group(1)) if match else float('inf')

# Get all files and sort them by their number
all_files = sorted(os.listdir(source_folder), key=get_number)

def copy_files(files, destination_suffix):
    destination_folder = os.path.join(base_destination, f"{dataset_prefix}_{destination_suffix}")
    os.makedirs(destination_folder, exist_ok=True)
    for file_name in files:
        shutil.copy2(os.path.join(source_folder, file_name), os.path.join(destination_folder, file_name))
    print(f"Copied {len(files)} datasets to {destination_folder}")

# Copy the first set of files
first_set = all_files[:first_set_count]
copy_files(first_set, str(first_set_count))

# Copy the second set of files
second_set = all_files[first_set_count:first_set_count + second_set_count]
copy_files(second_set, str(second_set_count))

# Copy random files from the remaining set
remaining_files = all_files[first_set_count + second_set_count:]
selected_files = random.sample(remaining_files, min(random_set_count, len(remaining_files)))
copy_files(selected_files, str(random_set_count))

print(f"Total files processed: {first_set_count + second_set_count + len(selected_files)}")
