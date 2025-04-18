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
import pyvista as pv
np.bool = bool
import matplotlib.colors as mcolors


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

def vis_o3d(G, edges, nodes, other_data, combined_cloud, target):

        # Visualization of the point cloud
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_axis_off()

        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.set_facecolor('none')
        ax.grid(False)  # Remove grid lines

        # Color points based on their class
        colors = ['blue' if data['class'] == 'object' else 'red' for data in other_data]

        ax.scatter(*combined_cloud.T, c=colors, alpha=0.6, s=50)

        ax.set_title(f"Combined Point Cloud - File {file_number}, {target}")

        plt.show()

        # Visualization of the graph
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_axis_off()

        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.set_facecolor('none')
        ax.grid(False)  # Remove grid lines

        # Color nodes based on their class
        node_colors = ['blue' if G.nodes[n]['other_data']['class'] == 'object' else 'red' for n in G.nodes()]

        # Color edges based on the groups they connect
        edge_colors = []
        for u, v in G.edges():
            if G.nodes[u]['other_data']['class'] == 'object' and G.nodes[v]['other_data']['class'] == 'object':
                edge_colors.append('blue')
            elif G.nodes[u]['other_data']['class'] == 'obstacles' and G.nodes[v]['other_data']['class'] == 'obstacles':
                edge_colors.append('red')
            else:
                edge_colors.append('red')  # For edges between different classes

        ax.scatter(*nodes.T, c=node_colors, alpha=0.6, s=50)

        # Plot edges with their respective colors
        for edge, color in zip(edges, edge_colors):
            ax.plot(*edge.T, color=color, alpha=0.2)

        ax.set_title(f"Combined Point Cloud Graph - File {file_number}, {target}")

        plt.show()

def vis_pv_ply(G, edges, nodes, other_data, combined_cloud, target, file_number):
    # Define colors
    object_color = np.array(mcolors.to_rgb('red'))
    obstacle_color = np.array(mcolors.to_rgb('blue'))

    # Create a new plotter for the graph visualization
    plotter = pv.Plotter(window_size=[1000, 1000])
    plotter.background_color = 'white'

    # Visualization of the graph
    graph_points = pv.PolyData(nodes)
    node_colors = np.array([object_color if G.nodes[n]['other_data']['class'] == 'object' else obstacle_color for n in G.nodes()])
    
    # Add colors to the PolyData
    graph_points['colors'] = node_colors

    # Use 'sphere_glyph' to make points appear as circles
    spheres = graph_points.glyph(scale=False, geom=pv.Sphere(radius=0.001), orient=False)
    
    plotter.add_mesh(spheres, scalars='colors', rgb=True, opacity=1)

    # Add title with smaller font size
    # plotter.add_text(f"Combined Point Cloud Graph - File {file_number}, {target}", font_size=10)

    # plotter.show()

    
    # save_directory = os.path.expanduser("~/Pictures/PyVista_Outputs/for_check-1")
    # os.makedirs(save_directory, exist_ok=True)
    # base_name = os.path.basename(target)  # Get the file name from the path
    # base_name = os.path.splitext(base_name)[0]  # Remove the extension
    # file_name = f"point_cloud_{file_number}_{base_name}.pdf"
    # full_path = os.path.join(save_directory, file_name)
    # plotter.save_graphic(full_path)




def vis_pv_net(G, edges, nodes, other_data, combined_cloud, target, file_number):
    # Define colors
    object_color = np.array(mcolors.to_rgb('red'))
    obstacle_color = np.array(mcolors.to_rgb('blue'))

    # Create a new plotter for the graph visualization
    plotter = pv.Plotter()
    plotter.background_color = 'white'

    # Visualization of the graph
    graph_points = pv.PolyData(nodes)
    node_colors = np.array([object_color if G.nodes[n]['other_data']['class'] == 'object' else obstacle_color for n in G.nodes()])
    
    # Add colors to the PolyData
    graph_points['colors'] = node_colors

    # Use 'sphere_glyph' to make points appear as circles
    spheres = graph_points.glyph(scale=False, geom=pv.Sphere(radius=0.001), orient=False)
    
    plotter.add_mesh(spheres, scalars='colors', rgb=True, opacity=1)

    # Add edges
    lines = []
    line_colors = []
    for u, v in G.edges():
        lines.extend([2, u, v])
        if G.nodes[u]['other_data']['class'] == 'object' and G.nodes[v]['other_data']['class'] == 'object':
            line_colors.append(object_color)
        elif G.nodes[u]['other_data']['class'] == 'obstacles' and G.nodes[v]['other_data']['class'] == 'obstacles':
            line_colors.append(obstacle_color)
        else:
            line_colors.append(np.array(mcolors.to_rgb('purple')))  # For edges between different classes

    lines = np.array(lines)
    line_colors = np.array(line_colors)

    line_poly = pv.PolyData()
    line_poly.points = nodes
    line_poly.lines = lines

    plotter.add_mesh(line_poly, scalars=line_colors, rgb=True, opacity=1, line_width=1)

    # Set up the save directory
    save_directory = os.path.expanduser("~/Pictures/PyVista_Outputs/for_check-1")
    os.makedirs(save_directory, exist_ok=True)

    # Prepare the file name
    base_name = os.path.basename(target)  # Get the file name from the path
    base_name = os.path.splitext(base_name)[0]  # Remove the extension
    file_name = f"network_{file_number}_{base_name}.png"
    full_path = os.path.join(save_directory, file_name)

    # Save the image as PNG
    plotter.show(auto_close=False)  # Render the scene
    plotter.screenshot(full_path, transparent_background=False, return_img=False)

    # Close the plotter
    plotter.close()

    print(f"Network visualization saved as {full_path}")

    # Add title with smaller font size
    # plotter.add_text(f"Combined Point Cloud Graph - File {file_number}, {target}", font_size=10)

    # plotter.show()
    # save_directory = os.path.expanduser("~/Pictures/PyVista_Outputs/for_check-1")
    # os.makedirs(save_directory, exist_ok=True)
    # base_name = os.path.basename(target)  # Get the file name from the path
    # base_name = os.path.splitext(base_name)[0]  # Remove the extension
    # file_name = f"network_{file_number}_{base_name}.pdf"
    # full_path = os.path.join(save_directory, file_name)
    # plotter.save_graphic(full_path)

    







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

        if obj_pcd.shape[0] < 800 :
            return
        
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

        def custom_parameter(num_points, volume, scale=0.01, offset=0.02):
            # Calculate a ratio using logarithms
            ratio = np.log(volume) / np.log(num_points)
            
            # # Apply scaling and offset
            # sigmoid = 1 / (1 + np.exp(-scale * (ratio - offset)))

            # # Clip the result to ensure it's between 0 and 1
            # return sigmoid
            parameter = scale * ratio + offset
            return np.clip(parameter, 0, 1)


        volume = vol(obstacles_pcd)
      
        ratio = 0.001
        initial_voxel_size=0.001
        # edge_voxel_size=0.1*(math.log(math.log(obstacles_pcd.shape[0]*ratio)*volume))
        # edge_voxel_size=0.1*(math.log(math.pow(obstacles_pcd.shape[0], 1/3 )*volume*200))
        # edge_voxel_size=0.0008*(math.log((obstacles_pcd.shape[0]/volume)))
        edge_voxel_size=custom_parameter(obstacles_pcd.shape[0], volume)/1.1
        curvature_threshold = 0.2
        print(volume, edge_voxel_size, obstacles_pcd.shape[0],obj_pcd.shape[0])
        
        
        
        downsample_cloud_1 = farthest_point_sampling(obj_pcd, obj_pcd.shape[0]*ratio)
        downsample_cloud_2 = farthest_point_sampling(obstacles_pcd, obstacles_pcd.shape[0]*ratio)
        # downsample_cloud_1 = edge_ds(downsample_cloud_1)
        # downsample_cloud_2 = edge_ds(downsample_cloud_2)
        
        # downsample_cloud_1 = edge_ds_2(obj_pcd, initial_voxel_size=initial_voxel_size, edge_voxel_size=edge_voxel_size, curvature_threshold=curvature_threshold, max_points=obj_pcd.shape[0]*ratio)
        # print("\nobject downsample : ", len(downsample_cloud_1))
        
        # downsample_cloud_2 = edge_ds_2(obstacles_pcd, initial_voxel_size=initial_voxel_size, edge_voxel_size=edge_voxel_size, curvature_threshold=curvature_threshold, max_points=obstacles_pcd.shape[0]*ratio)
        # print("\nobstacles downsample : ", len(downsample_cloud_2))

        # downsample_cloud_1 = edge_ds_1(obj_pcd, edge_voxel_size, curvature_threshold)
        # print("\nobject downsample : ", len(downsample_cloud_1))
        
        # downsample_cloud_2 = edge_ds_1(obstacles_pcd,edge_voxel_size, curvature_threshold)
        # print("\nobstacles downsample : ", len(downsample_cloud_2))

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

        # vis_o3d(G, edges, nodes, other_data, combined_cloud, target)
        vis_pv_ply(G, edges, nodes, other_data, combined_cloud, target, file_number)
        vis_pv_net(G, edges, nodes, other_data, combined_cloud, target, file_number)
        # vis_comb_net(G, col_target='red', col_obstacle='blue')


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
   

    #cs8
    # look = [7, 26, 29, 43, 51, 54, 60 ]
    # look = [ 7,15,26,29,43,51,54,59,60,61]
    #cs0
    look = [35, 34, 33, 26, 11, 10, 3]

    dataset_dir = '/home/emu/Documents/surrogate/dataset/pickle/data-set/cs0'
    
    # Get available file numbers from the directory
    available_files = get_available_files(dataset_dir)

    print("Found : ",len(available_files))

    # Process each available file number
    for file_number in available_files:
        if file_number in look:
            process_file(dataset_dir, file_number)
