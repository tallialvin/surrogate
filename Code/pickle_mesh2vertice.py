import os
import sys
import pickle
import re
import networkx as nx
import mesh2vertice as m2v
import numpy as np

if sys.version_info >= (3, 9) and np.lib.NumpyVersion(np.__version__) >= '1.20.0':
    # Fix VTK's deprecated numpy reference
    np.bool = bool  # type: ignore
    np.object = object  # type: ignore
    np.int = int  # type: ignore

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




def save_data_set(datasetdir, case_num,  file_number, seq, target, feasibility, G):
    # Create the directory path

    dirpath = os.path.join(datasetdir)
    os.makedirs(dirpath, exist_ok=True)
    
    # Define the file name
    file_path = os.path.join(dirpath, f'data-set-{file_number}.pkl')

    # Prepare the dataset
    data_set = {
        'seq': seq,
        'target': target,
        'feasibility': feasibility,
        'G': G
    }

    # Save the dataset to a pickle file
    with open(file_path, 'wb') as f:
        pickle.dump(data_set, f)

    print(f"Data set saved to {file_path}")

case_num = 3
pointdir = '/home/emu/Documents/surrogate/dataset/points/'
pickle_dir = "/home/emu/Documents/surrogate/dataset/pickle"
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

for file_number in available_numbers:
    if file_number > 0:
        try:
            selected_data = read_selected_data(pickledir, case_num, file_number)
            new_dir = os.path.join(pickle_dir,'m2v',f'cs{case_num}')
            if selected_data:
                print(f"\nProcessing data for file number {file_number}:")
                print("Sequence:", selected_data['seq'])
                print("Target:", selected_data['target'])
                print("Feasibility:", selected_data['feasibility'])

############################################
                ratio = 0.001
############################################

                # Process the target object
                # Process the target object
                stl_file = os.path.basename(selected_data["target"])
                obj_stlpath = os.path.join(mesh_dir, stl_file)

                # Extract points and calculate sample size for target object
                po_t = m2v.extract_trimesh_1(obj_stlpath)  # Returns TrackedArray
                num_samples_t = max(1, int(len(po_t) * ratio))  # Ensure at least 1 sample
                po_t = m2v.farthest_point_sampling(np.asarray(po_t), num_samples_t)  # TrackedArray works directly

                # Process each STL file in the sequence
                po_ob = []
                for sq in selected_data['seq']:
                    stl_path = sorted_stl_list[sq-1]  # sq is 1-indexed, list is 0-indexed
                    p_o = m2v.extract_trimesh_1(stl_path)
                    po_ob.append(p_o)  # No need for np.asarray()

                # Combine all points and sample
                combined_points = np.vstack(po_ob)  # Automatically handles TrackedArray
                total_points = len(combined_points)
                num_samples_ob = max(1, int(total_points * ratio))  # Ensure at least 1 sample
                po_ob = m2v.farthest_point_sampling(combined_points, num_samples_ob)

                # combined_cloud = np.vstack((po_t, po_ob))
                graph_type = [{'class': 'target'} for _ in range(len(po_t))] + \
                        [{'class': 'obstacle'} for _ in range(len(po_ob))]
                # input(graph_type)
                # Obtaining base id (lowest point in the combined cloud)
                # base_id = np.argmin(combined_cloud[:, 2])

                # Create graphs with adjusted node IDs using k-NN
                # G1 = m2v.knn_to_graph(po_t, graph_type[:len(po_t)], k=5, graph_threshold=1)
                # G2 = m2v.knn_to_graph(po_ob, graph_type[len(po_t):], k=5, graph_threshold=1, 
                #                 node_id_offset=len(po_t))
                G1 = m2v.knn_to_graph(po_t)
                G2 = m2v.knn_to_graph(po_ob)
                

                nx.set_node_attributes(G1, 'object', 'graph_type')
                nx.set_node_attributes(G2, 'obstacle', 'graph_type')

                # # For G1
                # G1_nodes = list(G1.nodes(data=True))  # Returns [(node_id, {attr_dict}), ...]
                # print("G1 Node Attributes:", G1_nodes[0][1].keys())  # Show keys of the first node's attributes

                # # For G2
                # G2_nodes = list(G2.nodes(data=True))
                # print("G2 Node Attributes:", G2_nodes[0][1].keys())
                
                # Compose the graphs
                # G = nx.compose(G1, G2)
                G = m2v.combine_graphs_with_offset(G1, G2)
                
                # m2v.vis_comb_net(G)
                
                save_data_set(new_dir, case_num, file_number, selected_data['seq'], selected_data['target'], selected_data['feasibility'], G)
                print(f"Successfully processed and saved data for file number {file_number}")

        except Exception as e:
            print(f"Error processing file number {file_number}: {str(e)}")
            print(f"Skipping file number {file_number} and continuing with the next file.")
            continue
