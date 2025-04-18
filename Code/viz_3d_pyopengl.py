import networkx as nx
from shortpath2 import array_to_graph
import numpy as np
import glob
import os
import open3d as o3d
import matplotlib.pyplot as plt
import pickle  # Ensure you import pickle for loading datasets
import re
import math

import plotly.graph_objects as go


from downsampling import farthest_point_sampling, edge_ds_1, edge_ds_2, edge_ds_3, edge_ds_4, edge_ds_2_with_hpr, vol
from shortpath2 import knn_to_graph, knn_mst_graph

def load_data_set(directory, file_number):

        file_path = os.path.join(directory, f'data-set-{file_number}.pkl')

        # Check if the file exists
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} does not exist.")
            return None

        # Load the dataset from the pickle file
        with open(file_path, 'rb') as f:
            data_set = pickle.load(f)

        return data_set


def process_file(dataset_dir, file_number):
    dataset = dataset_dir
    
    data = load_data_set(dataset, file_number)

    if data is not None:
        seq = data['seq']
        target = data['target']
        feasibility = data['feasibility']

        # Safely process obstacles_pcd
        obstacles_pcd = data['obstacles_pcd']
        if isinstance(obstacles_pcd, tuple):
            # Convert tuple to NumPy array if necessary
            obstacles_pcd = np.array(obstacles_pcd[0])

        # Safely process obj_pcd
        obj_pcd = data['obj_pcd']
        if isinstance(obj_pcd, tuple):
            # Convert tuple to NumPy array if necessary
            obj_pcd = np.array(obj_pcd[0])

        print(f"File Number: {file_number}")
        print("Sequence:", seq)
        print("Target:", target)
        print("Feasibility:", feasibility)

        try:
            print("Obstacles PCD shape:", obstacles_pcd.shape)
        except AttributeError:
            print("Error: obstacles_pcd does not have a shape. Type:", type(obstacles_pcd))

        try:
            print("Object PCD shape:", obj_pcd.shape)
        except AttributeError:
            print("Error: obj_pcd does not have a shape. Type:", type(obj_pcd))
      
        ratio = 0.01

        downsample_cloud_1 = farthest_point_sampling(obj_pcd, obj_pcd.shape[0]*ratio)
        downsample_cloud_2 = farthest_point_sampling(obstacles_pcd, obstacles_pcd.shape[0]*ratio)

        # Combine downsampled point clouds
        combined_cloud = np.vstack((downsample_cloud_1, downsample_cloud_2))
        
        # Create other_data with class information
        other_data = [{'class': 'object'} for _ in range(len(downsample_cloud_1))] + \
                     [{'class': 'obstacles'} for _ in range(len(downsample_cloud_2))]

        # Obtaining base id (lowest point in the combined cloud)
        # base_id = np.argmin(combined_cloud[:, 2])

        # Create graphs with adjusted node IDs using k-NN
        G1 = knn_mst_graph(downsample_cloud_1, other_data[:len(downsample_cloud_1)], k=5, graph_threshold=1)
        G2 = knn_mst_graph(downsample_cloud_2, other_data[len(downsample_cloud_1):], k=5, graph_threshold=1, 
                        node_id_offset=len(downsample_cloud_1))

        # Compose the graphs
        G = nx.compose(G1, G2)

        # Extract nodes and edges for visualization
        nodes = np.array([G.nodes[n]['position'] for n in G.nodes()])
        edges = np.array([(G.nodes[u]['position'], G.nodes[v]['position']) for (u, v) in G.edges()])

        # # Create the figure
        # fig = go.Figure()

        # # Add nodes
        # node_x, node_y, node_z = nodes.T
        # fig.add_trace(go.Scatter3d(
        #     x=node_x, y=node_y, z=node_z,
        #     mode='markers',
        #     marker=dict(
        #         size=5,
        #         color=[('red' if G.nodes[n]['other_data']['class'] == 'object' else 'blue') for n in G.nodes()],
        #         opacity=0.6
        #     ),
        #     hoverinfo='text',
        #     text=[f"Node {n}: {G.nodes[n]['other_data']['class']}" for n in G.nodes()]
        # ))

        # # Add edges
        # edge_x, edge_y, edge_z = [], [], []
        # edge_colors = []
        # for u, v in G.edges():
        #     edge_x.extend([nodes[u][0], nodes[v][0], None])
        #     edge_y.extend([nodes[u][1], nodes[v][1], None])
        #     edge_z.extend([nodes[u][2], nodes[v][2], None])
        #     if G.nodes[u]['other_data']['class'] == 'object' and G.nodes[v]['other_data']['class'] == 'object':
        #         edge_colors.extend(['red', 'red', 'red'])
        #     elif G.nodes[u]['other_data']['class'] == 'obstacles' and G.nodes[v]['other_data']['class'] == 'obstacles':
        #         edge_colors.extend(['blue', 'blue', 'blue'])
        #     else:
        #         edge_colors.extend(['blue', 'blue', 'blue'])

        # fig.add_trace(go.Scatter3d(
        #     x=edge_x, y=edge_y, z=edge_z,
        #     mode='lines',
        #     line=dict(color=edge_colors, width=2),
        #     hoverinfo='none'
        # ))

        # # Update layout
        # fig.update_layout(
        #     title=f"Combined Point Cloud Graph - File {file_number}, {target}",
        #     scene=dict(
        #         xaxis=dict(visible=False),
        #         yaxis=dict(visible=False),
        #         zaxis=dict(visible=False),
        #         bgcolor='rgba(0,0,0,0)'
        #     ),
        #     showlegend=False
        # )

        # # Show the plot
        # fig.show()


        # Create Open3D point cloud object
        final_pcd = o3d.geometry.PointCloud()
        final_pcd.points = o3d.utility.Vector3dVector(combined_cloud)

        # Set colors based on class
        colors = np.array([[1, 0, 0] if data['class'] == 'object' else [0, 0, 1] for data in other_data])
        final_pcd.colors = o3d.utility.Vector3dVector(colors)

        # Visualize the point cloud
        o3d.visualization.draw_geometries([final_pcd])

        input("next")





def get_available_files(dirpath):
    """ Get a list of available dataset file numbers from the specified directory. """
    file_pattern = os.path.join(dirpath, 'data-set-*.pkl')
    files = glob.glob(file_pattern)
    
    file_numbers = []
    for file in files:
        # Extract the number from the filename using regex
        filename = os.path.basename(file)
        match = re.search(r'(\d+)', filename)  # Look for digits in the filename
        if match:
            number_str = match.group(1)  # Get the first match of digits
            try:
                file_numbers.append(int(number_str))  # Convert to integer
            except ValueError:
                print(f"Warning: Could not convert '{number_str}' to an integer.")
    
    return sorted(file_numbers)


if __name__ == '__main__':
    
    #cs1
    look = [7, 26, 29, 43, 51, 54, 60 ]
    #cs0
    #look = [35, 34, 33, 26, 11, 10, 3]


    dataset_dir = '/home/emu/Documents/surrogate/dataset/cs8'
    
    # Get available file numbers from the directory
    available_files = get_available_files(dataset_dir)

    print("Found : ",len(available_files))

    # Process each available file number
    for file_number in available_files:
        if (file_number in look):
            process_file(dataset_dir, file_number)
