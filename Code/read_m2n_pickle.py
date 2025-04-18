import os
import pickle
import re
import networkx as nx
import mesh2net as m2n

def get_available_file_numbers(datasetdir, case_num):
    dirpath = os.path.join(datasetdir, f'cs{case_num}')
    
    if not os.path.exists(dirpath):
        print(f"Directory not found: {dirpath}")
        return []
    
    file_numbers = []
    for filename in os.listdir(dirpath):
        match = re.search(r'data-set-(\d+)\.pkl', filename)
        if match:
            file_numbers.append(int(match.group(1)))
    
    return sorted(file_numbers)

def read_selected_data(datasetdir, case_num, file_number):
    # Construct the file path
    dirpath = os.path.join(datasetdir, f'cs{case_num}')
    file_path = os.path.join(dirpath, f'data-set-{file_number}.pkl')

    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None

    # Read the pickle file
    with open(file_path, 'rb') as f:
        data_set = pickle.load(f)

    # Extract only the desired fields
    selected_data = {
        'seq': data_set['seq'],
        'target': data_set['target'],
        'feasibility': data_set['feasibility'],
        'G' : data_set['G']
    }

    # print(f"Selected data read from {file_path}")
    return selected_data

def get_sorted_stl_list(mesh_dir):
    stl_files = [f for f in os.listdir(mesh_dir) if f.endswith('.stl')]
    sorted_stl_list = sorted(stl_files, key=lambda x: int(x.split('_')[0]))
    return [os.path.join(mesh_dir, f) for f in sorted_stl_list]

def get_mesh_directory(base_mesh_dir, case_num):
    # List all directories in the base mesh directory
    all_dirs = [d for d in os.listdir(base_mesh_dir) if os.path.isdir(os.path.join(base_mesh_dir, d))]
    
    # Filter directories that start with the correct case number and don't end with "_wo"
    matching_dirs = [d for d in all_dirs if d.startswith(f"cs{case_num}_") and not d.endswith("_wo")]
    
    if not matching_dirs:
        print(f"No matching mesh directory found for case {case_num}")
        return None
    
    # If multiple matching directories are found, use the first one
    mesh_dir = os.path.join(base_mesh_dir, matching_dirs[0])
    print(f"Using mesh directory: {mesh_dir}")
    return mesh_dir

# Example usage
case_num = 8
pointdir = '/home/emu/Documents/surrogate/dataset/points/'
pickledir = '/home/emu/Documents/surrogate/dataset/pickle/m2n/'
base_mesh_dir = '/home/emu/Documents/surrogate/dataset/meshes/'



# Get the correct mesh directory based on case_num
mesh_dir = get_mesh_directory(base_mesh_dir, case_num)
if mesh_dir is None:
    print("Exiting due to missing mesh directory")
    exit()

sorted_stl_list = get_sorted_stl_list(mesh_dir)
# print("Sorted STL files:")
# for stl_file in sorted_stl_list:
#     # print(stl_file)

available_numbers = get_available_file_numbers(pickledir, case_num)
# print(f"Available file numbers: {available_numbers}")

for file_number in available_numbers:
    selected_data = read_selected_data(pickledir, case_num, file_number)
    
    if selected_data:
        print(f"\nData for file number {file_number}:")
        print("Sequence:", selected_data['seq'])
        print("Target:", selected_data['target'])
        print("Feasibility:", selected_data['feasibility'])

        # Combine target and obstacle graphs
        combined_graph = selected_data['G']

        # Visualizations
        # m2n.vis_comb(po_ob, pe_ob, po_t, pe_t)
        # m2n.vis_net_comb(pe_ob, G_ob, pe_t, g_t)
        m2n.vis_comb_net(combined_graph)