# dataset.py
import os
import glob
import re
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
import pickle

from shortpath2 import knn_to_graph
from downsampling import edge_ds_2, vol, farthest_point_sampling
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import networkx as nx
import mesh2net as m2n

import logging
import sys

class TqdmToLogger(tqdm):
    def __init__(self, *args, logger=None, level=logging.INFO, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logger
        self.level = level

    def display(self, msg=None, pos=None):
        if self.logger:
            self.logger.log(self.level, self.format_meter())
        else:
            super().display(msg, pos)


class GraphDataset(Dataset):
    def __init__(self, directory, case_num, folder):
        print("i am in GraphDataSet")
        self.directory = directory
        self.case_num = case_num
        self.folder = folder
        self.file_numbers = self.get_available_files()
        print(f"Found {len(self.file_numbers)} pickle files.")

        # Get the correct mesh directory based on case_num
        base_mesh_dir = os.path.join(self.directory, 'meshes')
        self.mesh_dir = get_mesh_directory(base_mesh_dir, case_num)
        if self.mesh_dir is None:
            print("Exiting due to missing mesh directory")
            exit()
        
        self.sorted_stl_list = get_sorted_stl_list(self.mesh_dir)
        
        self.data_list = []
        timeout_duration = 10

        logger = logging.getLogger('file_processing')

        with TqdmToLogger(total=len(self.file_numbers), desc="Processing files", unit="file", logger=logger) as pbar:
            while self.file_numbers:
                file_number = self.file_numbers.pop(0)
                file_path = os.path.join(self.directory, 'pickle', 'm2n', self.folder, f"data-set-{file_number}.pkl")

                if not os.path.exists(file_path):
                    pbar.update(1)
                    continue

                pbar.set_description(f"Processing file {file_number}")
                
                # Your existing ThreadPoolExecutor code here
                
                pbar.update(1)
        
        print(f"\nLoaded {len(self.data_list)} graphs.")
        sys.stdout.flush()
        sys.stderr.flush()
    def get_available_files(self):
        file_pattern = os.path.join(self.directory, 'pickle', 'm2n', self.folder, 'data-set-*.pkl')
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

    def process_file(self, file_number):

        data = self.load_data_set(file_number)

        if data is not None:

            # Combine target and obstacle graphs
            
            # Visualizations
            # m2n.vis_comb(po_ob, pe_ob, po_t, pe_t)
            # m2n.vis_net_comb(pe_ob, G_ob, pe_t, g_t)
            # m2n.vis_comb_net(combined_graph)

            G = data['G']

            # m2n.vis_comb_net(G,'','')
                      
            # Create edge_index tensor
            edge_index = torch.tensor(list(G.edges()), dtype=torch.long).t().contiguous()

            # Create feature tensor with 4 attributes (3 position + 1 class)
            x = torch.tensor([[
                                G.nodes[n]['x'],      # x coordinate
                                G.nodes[n]['y'],      # y coordinate
                                G.nodes[n]['z'],      # z coordinate
                                1 if G.nodes[n]['graph_type'] == 'target' else 0
                                ] 
                                for n in G.nodes()], 
                                dtype=torch.float)

            # Ensure y is a single label for the entire graph (0 or 1)
            feasibility = data.get('feasibility', False)  # Assuming feasibility is defined in your data
            y = torch.tensor([1 if feasibility else 0], dtype=torch.long)  # Single label for graph


            return Data(x=x, edge_index=edge_index, y=y).to("cuda:0")


        else:
            print(f"No data found for file number: {file_number}")
            return None
    
    def load_data_set(self, file_number):

        file_path = os.path.join(self.directory, 'pickle', 'm2n', self.folder, f'data-set-{file_number}.pkl')
        
        # Check if the file exists
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} does not exist.")
            return None

        # Load the dataset from the pickle file
        with open(file_path, 'rb') as f:
            data_set = pickle.load(f)

        # print(f"Data set loaded from {file_path}")

        return data_set

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

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