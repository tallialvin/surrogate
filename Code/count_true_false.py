#count_true_false.py
import os
import glob
import pickle
import re
from collections import defaultdict

def load_data_set(dataset_path, file_number):
    file_path = os.path.join(dataset_path, f'data-set-{file_number}.pkl')
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

def get_available_files(dirpath):
    file_pattern = os.path.join(dirpath, 'data-set-*.pkl')
    files = glob.glob(file_pattern)
    
    file_numbers = []
    for file in files:
        filename = os.path.basename(file)
        match = re.search(r'(\d+)', filename)
        if match:
            number_str = match.group(1)
            try:
                file_numbers.append(int(number_str))
            except ValueError:
                print(f"Warning: Could not convert '{number_str}' to an integer.")
    
    return sorted(file_numbers)

def count_feasibility(dataset_dir):
    available_files = get_available_files(dataset_dir)
    true_count = 0
    false_count = 0
    total_files = 0

    for file_number in available_files:
        data = load_data_set(dataset_dir, file_number)
        if data is not None:
            feasibility = data['feasibility']
            if feasibility:
                true_count += 1
            else:
                false_count += 1
            total_files += 1

    return true_count, false_count, total_files

if __name__ == '__main__':
    dataset_dir = '/home/emu/Documents/surrogate/dataset/cs3_2000'
    
    true_count, false_count, total_files = count_feasibility(dataset_dir)
    
    print(f"Total files processed: {total_files}")
    print(f"Number of True values: {true_count}")
    print(f"Number of False values: {false_count}")
    print(f"Percentage of True: {(true_count / total_files) * 100:.2f}%")
    print(f"Percentage of False: {(false_count / total_files) * 100:.2f}%")
