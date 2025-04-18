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
        'feasibility': data_set['feasibility']
    }

    print(f"Selected data read from {file_path}")
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
pickledir = '/home/emu/Documents/surrogate/dataset/pickle/data-set/'
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

#cs8
# look = [7, 26, 29, 43, 51, 54, 60 ]
#look = [ 7,15,26,29,43,51,54,59,60,61]
# #cs0
look = [35, 34, 33, 26, 11, 10, 3]

for file_number in available_numbers:
    
    if file_number in look:
        
        selected_data = read_selected_data(pickledir, case_num, file_number)
        
        if selected_data:
            print(f"\nData for file number {file_number}:")
            print("Sequence:", selected_data['seq'])
            print("Target:", selected_data['target'])
            print("Feasibility:", selected_data['feasibility'])

            # Process the target object
            obj_stlpath = selected_data['target']
            print(obj_stlpath)
            po_t, pe_t, g_t = m2n.process_stl_file(obj_stlpath, dataset_path=pointdir)

            # m2n.vis_mesh_graph(obj_stlpath, pe_t, g_t)

            po_ob = []
            pe_ob = []
            graphs = []

            # Process each STL file in the sequence
            for sq in selected_data['seq']:
                stl_path = sorted_stl_list[sq-1]  # sq is 1-indexed, list is 0-indexed
                # print(stl_path)
                p_o, e_o, g_o = m2n.process_stl_file(stl_path, dataset_path=pointdir)
                po_ob.append(p_o)
                pe_ob.append(e_o)
                graphs.append(g_o)

            # Compose graphs with offset
            graphs = m2n.compose_graphs_with_offset(graphs)

            print(f"number graphs : {nx.number_connected_components(graphs)}")

            # Connect graphs using MST
            G_ob = m2n.connect_graphs_mst(graphs, pe_ob)

            # Combine target and obstacle graphs
            combined_graph = m2n.comb_tar_obs(g_t, G_ob)

            # Visualizations
            # m2n.vis_comb(po_ob, pe_ob, po_t, pe_t, file_number, selected_data['target'])
            # m2n.vis_net_comb(pe_ob, G_ob, pe_t, g_t)
            # m2n.vis_comb_net(combined_graph, file_number, selected_data['target'])