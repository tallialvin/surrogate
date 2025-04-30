import pyvista as pv
import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import pickle
import matplotlib.colors as mcolors
np.bool = bool
# pv.OFF_SCREEN = True


def read_mesh(filepath):
    return pv.read(filepath)

def save_graph(graph, output_path):

    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Verbose saving with error handling
        try:
            with open(output_path, 'wb') as f:
                pickle.dump(graph, f)
            print(f"Graph saved successfully to: {output_path}")
        except Exception as save_error:
            print(f"Save error: {save_error}")

            # Fallback: try absolute path
            abs_path = os.path.abspath(output_path)
            try:
                with open(abs_path, 'wb') as f:
                    pickle.dump(graph, f)
                print(f"Saved using absolute path: {abs_path}")
            except Exception as abs_save_error:
                print(f"Absolute path save failed: {abs_save_error}")

    except Exception as dir_error:
        print(f"Directory creation error: {dir_error}")

def load_graph(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def save_point_cloud(points, output_path):
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create pv PolyData point cloud
        point_cloud = pv.PolyData(points)
        
        # Verbose saving with error handling
        try:
            point_cloud.save(output_path, binary=True)
            print(f"Point cloud saved successfully to: {output_path}")
        except Exception as save_error:
            print(f"Save error: {save_error}")
            
            # Fallback: try absolute path
            abs_path = os.path.abspath(output_path)
            try:
                point_cloud.save(abs_path, binary=True)
                print(f"Saved using absolute path: {abs_path}")
            except Exception as abs_save_error:
                print(f"Absolute path save failed: {abs_save_error}")
    
    except Exception as dir_error:
        print(f"Directory creation error: {dir_error}")


def extract_edges(mesh, feature_angle=10):
    return mesh.extract_feature_edges(
        feature_angle=feature_angle,
        boundary_edges=True,
        non_manifold_edges=True,
        feature_edges=True,
        manifold_edges=True
    )

def calculate_edge_lengths(edges):
    edge_lengths = []
    for i in range(edges.n_cells):
        edge = edges.extract_cells(i)
        start_point = edge.points[0]
        end_point = edge.points[1]
        length = np.linalg.norm(end_point - start_point)
        edge_lengths.append(length)
    return np.array(edge_lengths)

def edges_to_point_cloud(edges, edge_lengths, base_points=2, length_scale=0.001, max_points=100):
    points = []
    for i in range(edges.n_cells):
        edge = edges.extract_cells(i)
        start_point = edge.points[0]
        end_point = edge.points[1]
        edge_length = edge_lengths[i]
        
        num_points = max(2, min(max_points, base_points + int(edge_length / length_scale)))
        
        for j in range(num_points):
            t = j / (num_points - 1)
            point = start_point * (1 - t) + end_point * t
            points.append(point)
    return np.array(points)

def voxel_downsample(points, voxel_size=0.001):

    voxel_grid = {}
    for point in points:
        # Quantize point coordinates to voxel grid
        voxel_key = tuple(np.floor(point / voxel_size).astype(int))
        
        if voxel_key not in voxel_grid:
            voxel_grid[voxel_key] = [point]
        else:
            voxel_grid[voxel_key].append(point)
    
    # Use centroid of points in each voxel
    downsampled_points = [np.mean(voxel_points, axis=0) for voxel_points in voxel_grid.values()]
    
    return np.array(downsampled_points)

def create_node_data(point):
    return {'x': point[0], 'y': point[1], 'z': point[2]}

def knn_mst_graph(arr, k=3):
    # K-Nearest Neighbors graph
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(arr)
    distances, indices = nbrs.kneighbors(arr)
    
    # Create complete graph for MST
    G_complete = nx.Graph()
    for i, point1 in enumerate(arr):
        G_complete.add_node(i, **create_node_data(point1))
        for j, point2 in enumerate(arr[i+1:], start=i+1):
            distance = np.linalg.norm(point1 - point2)
            G_complete.add_edge(i, j, weight=distance)
    
    # Compute Minimum Spanning Tree
    mst = nx.minimum_spanning_tree(G_complete)
    
    # Create K-NN graph
    G_knn = nx.Graph()
    for i, point in enumerate(arr):
        G_knn.add_node(i, **create_node_data(point))
    
    # Add K-NN edges
    for i, (dist, idx) in enumerate(zip(distances, indices)):
        for j, d in zip(idx[1:], dist[1:]):  # Skip first index (self)
            G_knn.add_edge(i, j, weight=d)
    
    # Combine K-NN and MST
    G_combined = nx.Graph(G_knn)
    G_combined.add_edges_from(mst.edges(data=True))
    
    return G_combined

def compose_graphs_with_offset(graphs):
    combined_graph = nx.Graph()
    offset = 0
    for graph in graphs:
        # Relabel nodes with an offset
        relabeled_graph = nx.relabel_nodes(graph, 
                                        {n: n+offset for n in graph.nodes()})
        combined_graph = nx.compose(combined_graph, relabeled_graph)
        offset += len(graph.nodes())
    return combined_graph

def connect_graphs_mst(combined_graph, points_list, k=3):
    # Combine all points
    all_points = np.vstack(points_list)
    
    # Find inter-graph connections using KNN
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(all_points)
    distances, indices = nbrs.kneighbors(all_points)
    
    # Track graph boundaries
    graph_boundaries = [0]
    for points in points_list:
        graph_boundaries.append(graph_boundaries[-1] + len(points))
    
    # Add inter-graph edges
    for i in range(len(all_points)):
        for j in indices[i][1:k+1]:  # Skip self
            # Check if points are from different graphs
            graph_i = next(idx-1 for idx in graph_boundaries if idx > i) - 1
            graph_j = next(idx-1 for idx in graph_boundaries if idx > j) - 1
            
            if graph_i != graph_j:
                distance = np.linalg.norm(all_points[i] - all_points[j])
                combined_graph.add_edge(i, j, weight=distance)
    
    # Function to connect components iteratively
    def connect_components():
        components = list(nx.connected_components(combined_graph))
        
        while len(components) > 1:
            min_dist = float('inf')
            bridge_nodes = None
            bridge_component_indices = None
            
            for i in range(len(components)):
                for j in range(i + 1, len(components)):
                    # Find closest points between two components
                    min_component_dist = float('inf')
                    min_component_nodes = None
                    
                    for node_i in components[i]:
                        for node_j in components[j]:
                            dist = np.linalg.norm(all_points[node_i] - all_points[node_j])
                            if dist < min_component_dist:
                                min_component_dist = dist
                                min_component_nodes = (node_i, node_j)
                    
                    # Update overall minimum distance
                    if min_component_dist < min_dist:
                        min_dist = min_component_dist
                        bridge_nodes = min_component_nodes
                        bridge_component_indices = (i, j)
            
            # Add bridge edge between closest points of disconnected components
            if bridge_nodes:
                combined_graph.add_edge(bridge_nodes[0], bridge_nodes[1], weight=min_dist)
            
            # Recompute components after adding the edge
            components = list(nx.connected_components(combined_graph))
            # print(f"number graphs : {nx.number_connected_components(combined_graph)}")
        
        return combined_graph

    # Call the function to connect components iteratively
    combined_graph = connect_components()

    return combined_graph


def connect_graphs_mst_2(combined_graph, points_list, k=3):
    # Combine all points
    all_points = np.vstack(points_list)
    
    # Find inter-graph connections using KNN
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(all_points)
    distances, indices = nbrs.kneighbors(all_points)
    
    # Track graph boundaries
    graph_boundaries = [0]
    for points in points_list:
        graph_boundaries.append(graph_boundaries[-1] + len(points))
    
    # Add inter-graph edges
    for i in range(len(all_points)):
        for j in indices[i][1:k+1]:  # Skip self
            # Check if points are from different graphs
            graph_i = next(idx-1 for idx in graph_boundaries if idx > i) - 1
            graph_j = next(idx-1 for idx in graph_boundaries if idx > j) - 1
            
            if graph_i != graph_j:
                distance = np.linalg.norm(all_points[i] - all_points[j])
                combined_graph.add_edge(i, j, weight=distance)
    
    # Function to connect components iteratively using CoM representatives
    def connect_components():
        components = list(nx.connected_components(combined_graph))
        while len(components) > 1:
            # For each component, find its CoM and the closest node to CoM
            rep_nodes = []
            for comp in components:
                indices = np.array(list(comp))
                points = all_points[indices]
                com = points.mean(axis=0)
                dists = np.linalg.norm(points - com, axis=1)
                closest_idx = np.argmin(dists)
                rep_node = indices[closest_idx]
                rep_nodes.append(rep_node)
            
            # Find the closest pair of representative nodes between components
            min_dist = float('inf')
            bridge_nodes = None
            for i in range(len(rep_nodes)):
                for j in range(i + 1, len(rep_nodes)):
                    dist = np.linalg.norm(all_points[rep_nodes[i]] - all_points[rep_nodes[j]])
                    if dist < min_dist:
                        min_dist = dist
                        bridge_nodes = (rep_nodes[i], rep_nodes[j])
            
            # Add edge between the closest representatives
            if bridge_nodes:
                combined_graph.add_edge(bridge_nodes[0], bridge_nodes[1], weight=min_dist)
            
            # Update components
            components = list(nx.connected_components(combined_graph))
        return combined_graph

    # Call the function to connect components iteratively
    combined_graph = connect_components()

    return combined_graph


def comb_tar_obs(g_t, G_ob):
    # print(f"number obstacle sub-network : {nx.number_connected_components(G_ob)}")

    # Add a new attribute 'graph_type' to all nodes in g_t
    nx.set_node_attributes(g_t, 'target', 'graph_type')

    # Verify the new attribute
    g_t_attributes = set()
    for node in g_t.nodes():
        g_t_attributes.update(g_t.nodes[node].keys())

    # print(f"Updated attributes in g_t: {', '.join(g_t_attributes)}")

    # Add a new attribute 'graph_type' to all nodes in G_ob
    nx.set_node_attributes(G_ob, 'obstacle', 'graph_type')

    # Verify the new attribute
    G_ob_attributes = set()
    for node in G_ob.nodes():
        G_ob_attributes.update(G_ob.nodes[node].keys())

    # print(f"Updated attributes in G_ob: {', '.join(G_ob_attributes)}")

    # Use disjoint_union to combine the graphs
    combined_graph = nx.disjoint_union(g_t, G_ob)

    # Verify the attributes in the combined graph
    combined_attributes = set()
    for node in combined_graph.nodes():
        combined_attributes.update(combined_graph.nodes[node].keys())

    # print(f"Attributes in combined graph: {', '.join(combined_attributes)}")

    # # Distinguish nodes based on their 'graph_type' attribute
    # target_nodes = [node for node, attr in combined_graph.nodes(data=True) if attr.get('graph_type') == 'target']
    # obstacle_nodes = [node for node, attr in combined_graph.nodes(data=True) if attr.get('graph_type') == 'obstacle']

    # print(f"Number of target nodes: {len(target_nodes)}")
    # print(f"Number of obstacle nodes: {len(obstacle_nodes)}")

    # # Print a few nodes from each type to verify
    # print("\nSample target nodes:")
    # print(target_nodes[:5])
    # print("\nSample obstacle nodes:")
    # print(obstacle_nodes[:5])
    # print(f"number combined sub-network : {nx.number_connected_components(combined_graph)}")
    return combined_graph


def comb_tar_obs_1(g_t, G_ob):
    # print(f"number obstacle sub-network : {nx.number_connected_components(G_ob)}")

    # Use disjoint_union to combine the graphs
    combined_graph = nx.disjoint_union(g_t, G_ob)

    # Keep only the 'x', 'y', and 'z' attributes for each node
    for node in combined_graph.nodes():
        x = combined_graph.nodes[node].get('x', None)
        y = combined_graph.nodes[node].get('y', None)
        z = combined_graph.nodes[node].get('z', None)
        combined_graph.nodes[node].clear()
        if x is not None and y is not None and z is not None:
            combined_graph.nodes[node]['x'] = x
            combined_graph.nodes[node]['y'] = y
            combined_graph.nodes[node]['z'] = z

    # print(f"number combined sub-network : {nx.number_connected_components(combined_graph)}")
    return combined_graph

def vis_mesh_graph(filepath, points, graph):
    mesh = read_mesh(filepath)
    plotter = pv.Plotter()
    point_cloud = pv.PolyData(points)
    plotter.add_points(point_cloud, color='red', point_size=5)
    
    lines = []
    for edge in graph.edges():
        lines.extend([2, edge[0], edge[1]])
    lines = np.array(lines)
    
    line_poly = pv.PolyData()
    line_poly.points = points
    line_poly.lines = lines
    
    plotter.add_mesh(line_poly, color='blue', line_width=1)
    plotter.add_mesh(mesh, color='white', opacity=0.1)
    plotter.view_isometric()
    plotter.set_background('white')
    plotter.show()

# Example usage in process_stl_file
def process_stl_file(filepath, base_points=2, length_scale=0.001, max_points=100, k=3, voxel_size=0.001, dataset_path="/dataset/points"):
    # Prepare output paths
    base_filename = os.path.basename(filepath).replace('.stl', '')
    output_dir = os.path.join(dataset_path, os.path.basename(os.path.dirname(filepath)))
    points_output_path = os.path.join(output_dir, f"{base_filename}_full.ply")
    edge_points_output_path = os.path.join(output_dir, f"{base_filename}_edge.ply")
    graph_output_path = os.path.join(output_dir, f"{base_filename}_graph.pkl")

    # Check if point cloud and graph files already exist
    if (os.path.exists(points_output_path) and 
        os.path.exists(edge_points_output_path) and 
        os.path.exists(graph_output_path)):
        # print(f"Loading existing point clouds and graph for {filepath}")
        point_cloud = pv.read(points_output_path)
        downsampled_cloud = pv.read(edge_points_output_path)
        
        # Load the graph
        with open(graph_output_path, 'rb') as f:
            graph = pickle.load(f)
        
        return point_cloud.points, downsampled_cloud.points, graph

    # If files do not exist, proceed with processing
    print(filepath)
    mesh = pv.read(filepath)
    edges = extract_edges(mesh)
    edge_lengths = calculate_edge_lengths(edges)

    filename = os.path.basename(filepath)
    print(f"File: {filename}")

    point_cloud = edges_to_point_cloud(
        edges, 
        edge_lengths, 
        base_points, 
        length_scale, 
        max_points
    )
    
    # Initialize variables
    ratio = 1.0
    downsampled_cloud = point_cloud
    initial_points = len(point_cloud)
    print(f"Original point cloud size: {initial_points}")
    threshold = 0.02

    if initial_points < 2500:
        threshold = 0.05

    # Downsampling loop
    while ratio > threshold:
        downsampled_cloud = voxel_downsample(point_cloud, voxel_size)
        ratio = len(downsampled_cloud) / initial_points
        voxel_size += 0.0005
        
        # Optional: Break if voxel size becomes too large
        if voxel_size > 1.0:
            break

    # Apply voxel downsampling
    points = downsampled_cloud
    
    print(f"Downsampled point cloud size: {len(points)}, {100*len(points)/initial_points:.2f}%")

    graph = knn_mst_graph(points, k=k)
    print(f"Number of edges: {graph.number_of_edges()}")

    # vis_mesh_graph(filepath, points, graph)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save point clouds
    save_point_cloud(point_cloud, points_output_path)
    save_point_cloud(points, edge_points_output_path)

    # Save the graph using the new save_graph function
    save_graph(graph, graph_output_path)

    return point_cloud, points, graph

def process_directory(directory, base_points=2, length_scale=0.001, max_points=100, k=3, voxel_size=0.001):
    for filename in os.listdir(directory):
        if filename.endswith(".stl"):
            filepath = os.path.join(directory, filename)
            process_stl_file(filepath, base_points, length_scale, max_points, k, voxel_size)

def vis_comb(po_ob, pe_ob, po_t, pe_t, file_number, target, opacity=0.0, col_t='red', col_ob='blue'):
    # Original point cloud
    # combined_points_original = np.vstack(po_ob)
    # target_points_original = np.vstack(po_t)

    # Downsampled point cloud
    combined_points_downsampled = np.vstack(pe_ob)
    target_points_downsampled = np.vstack(pe_t)

    # Create PyVista PolyData objects
    # pcd_combined_original = pv.PolyData(combined_points_original)
    # pcd_target_original = pv.PolyData(target_points_original)
    pcd_combined_downsampled = pv.PolyData(combined_points_downsampled)
    pcd_target_downsampled = pv.PolyData(target_points_downsampled)

    # Create plotter
    plotter = pv.Plotter(window_size=[1000, 1000])

    # Create sphere glyph
    sphere = pv.Sphere(radius=0.0006)

    # Add meshes with different colors and opacity using spheres
    # combined_original_spheres = pcd_combined_original.glyph(scale=False, geom=sphere, orient=False)
    # target_original_spheres = pcd_target_original.glyph(scale=False, geom=sphere, orient=False)
    combined_downsampled_spheres = pcd_combined_downsampled.glyph(scale=False, geom=sphere, orient=False)
    target_downsampled_spheres = pcd_target_downsampled.glyph(scale=False, geom=sphere, orient=False)

    # plotter.add_mesh(combined_original_spheres, color=col_ob, opacity=opacity)
    # plotter.add_mesh(target_original_spheres, color=col_t, opacity=opacity)
    plotter.add_mesh(combined_downsampled_spheres, color=col_ob)
    plotter.add_mesh(target_downsampled_spheres, color=col_t)

    # plotter.add_text(f"Combined Point Cloud Graph - File {file_number}, {target}", font_size=10)
    
    # Set background color to white
    plotter.set_background('white')
    
    # Show the plot
    plotter.show()
    # Set background color to white
    # plotter.set_background('white')
    
    # base_name = os.path.basename(target)  # Get the file name from the path
    # base_name = os.path.splitext(base_name)[0]  # Remove the extension


    # save_directory = os.path.expanduser("~/Pictures/PyVista_Outputs")
    # os.makedirs(save_directory, exist_ok=True)
    # file_name = f"point_cloud_{file_number}_{base_name}.pdf"
    # full_path = os.path.join(save_directory, file_name)
    # plotter.save_graphic(full_path)

def vis_unit(po_ob, pe_ob):
    # Original point cloud
    combined_points_original = np.vstack(po_ob)

    # Downsampled point cloud
    combined_points_downsampled = np.vstack(pe_ob)

    # Create PyVista PolyData objects
    pcd_combined_original = pv.PolyData(combined_points_original)
    pcd_combined_downsampled = pv.PolyData(combined_points_downsampled)

    # Create plotter
    plotter = pv.Plotter()

    # Add meshes with different colors and opacity
    plotter.add_mesh(pcd_combined_original, color='blue', opacity=0.3, point_size=5)  # Transparent blue for original combined points
    plotter.add_mesh(pcd_combined_downsampled, color='blue', point_size=5)            # Solid blue for downsampled combined points

    # Show the plot
    plotter.show()

def vis_net(points, graph):
    plotter = pv.Plotter()
    point_cloud = pv.PolyData(points)
    plotter.add_points(point_cloud, color='blue', point_size=5)
    
    lines = []
    for edge in graph.edges():
        lines.extend([2, edge[0], edge[1]])
    lines = np.array(lines)
    
    line_poly = pv.PolyData()
    line_poly.points = points
    line_poly.lines = lines
    
    plotter.add_mesh(line_poly, color='blue', line_width=1)
    plotter.view_isometric()
    plotter.set_background('white')
    plotter.show()

def vis_net_comb(points1, graph1, points2, graph2, 
                 col_net1='blue', 
                 col_net2='red'):
    """
    Visualize two networks with comparison options
    
    Parameters:
    - points1, points2: point coordinates for networks
    - graph1, graph2: NetworkX graph objects
    - col_net1, col_net2: colors for networks
    """
    # Create plotter

    points1 = np.vstack(points1)
    points2 = np.vstack(points2)

    plotter = pv.Plotter()
    
    # First network (transparent)
    point_cloud1 = pv.PolyData(points1)
    plotter.add_points(point_cloud1, color=col_net1, point_size=5)
    
    lines1 = []
    for edge in graph1.edges():
        lines1.extend([2, edge[0], edge[1]])
    lines1 = np.array(lines1)
    
    line_poly1 = pv.PolyData()
    line_poly1.points = points1
    line_poly1.lines = lines1
    
    plotter.add_mesh(line_poly1, color=col_net1, line_width=1)
    
    # Second network (solid)
    point_cloud2 = pv.PolyData(points2)
    plotter.add_points(point_cloud2, color=col_net2, point_size=5)
    
    lines2 = []
    for edge in graph2.edges():
        lines2.extend([2, edge[0], edge[1]])
    lines2 = np.array(lines2)
    
    line_poly2 = pv.PolyData()
    line_poly2.points = points2
    line_poly2.lines = lines2
    
    plotter.add_mesh(line_poly2, color=col_net2, line_width=1)
    
    # Visualization settings
    plotter.view_isometric()
    plotter.set_background('white')
    plotter.show()




def vis_comb_net(combined_graph, col_target='red', col_obstacle='blue'):
    """
    Visualize a combined network with nodes as spheres and edges colored by graph_type
    
    Parameters:
    - combined_graph: NetworkX graph object with 'x', 'y', 'z', and 'graph_type' attributes
    - col_target: color for 'target' nodes and edges (default: red)
    - col_obstacle: color for 'obstacle' nodes and edges (default: blue)
    """
    # Create plotter
    plotter = pv.Plotter(window_size=[1000, 1000])
    
    # Extract points and create point cloud
    points = np.array([[data['x'], data['y'], data['z']] for _, data in combined_graph.nodes(data=True)])
    point_cloud = pv.PolyData(points)
    
    # Convert color names to RGB
    rgb_target = np.array(mcolors.to_rgb(col_target))
    rgb_obstacle = np.array(mcolors.to_rgb(col_obstacle))
    
    # Create colors for nodes based on graph_type
    node_colors = np.array([rgb_target if data['graph_type'] == 'target' else rgb_obstacle 
                            for _, data in combined_graph.nodes(data=True)])
    
    # Add colors to the PolyData
    point_cloud['colors'] = node_colors
    
    # Create spheres for nodes
    spheres = point_cloud.glyph(scale=False, geom=pv.Sphere(radius=0.0004), orient=False)
    
    # Add spheres to plotter
    plotter.add_mesh(spheres, scalars='colors', rgb=True)
    
    # Create lines with colors
    lines = []
    line_colors = []
    for edge in combined_graph.edges():
        lines.extend([2, edge[0], edge[1]])
        # Determine edge color based on nodes' graph_type
        start_type = combined_graph.nodes[edge[0]]['graph_type']
        end_type = combined_graph.nodes[edge[1]]['graph_type']
        if start_type == end_type:
            line_colors.append(rgb_target if start_type == 'target' else rgb_obstacle)
        else:
            # For edges between different types, use an intermediate color
            line_colors.append(np.array(mcolors.to_rgb('purple')))  # You can change this to any color you prefer
    
    lines = np.array(lines)
    line_colors = np.array(line_colors)
    
    line_poly = pv.PolyData()
    line_poly.points = points
    line_poly.lines = lines
    
    # Add lines to plotter
    plotter.add_mesh(line_poly, scalars=line_colors, rgb=True)
    # plotter.add_text(f"Combined Point Cloud Graph - File {file_number}, {target}", font_size=10)
    
    # Visualization settings
    plotter.view_isometric()
    plotter.set_background('white')
    plotter.show()
    # # # Set background color to white
    # # # plotter.set_background('white')
    
    # base_name = os.path.basename(target)  # Get the file name from the path
    # base_name = os.path.splitext(base_name)[0]  # Remove the extension

    # # Instead of showing the plot, save it as a PDF
    # save_directory = os.path.expanduser("~/Pictures/PyVista_Outputs/m2n")
    # os.makedirs(save_directory, exist_ok=True)
    # file_name = f"network_{file_number}_{base_name}.pdf"
    # full_path = os.path.join(save_directory, file_name)
    # plotter.save_graphic(full_path)



# if __name__ == "__main__":

#     # mesh_directory = "/home/emu/Documents/new_had/had/docker/dataset/meshes/cs0_wrc18_rubber_band_drive_unit_ext"
#     # mesh_directory = "/home/emu/Documents/new_had/had/docker/dataset/meshes/cs1_CU-X405C2_compressed_aligned"
#     mesh_directory = "/home/emu/Documents/new_had/had/docker/dataset/meshes/cs2_TV_ASSY-55MZ1800_V11"
#     # mesh_directory = "/home/emu/Documents/new_had/had/docker/dataset/meshes/cs3_CU-J223D-V5"
#     process_directory(
#         mesh_directory, 
#         base_points=2,      
#         length_scale=0.001, 
#         max_points=50,      
#         k=3,                
#         voxel_size=0.001    # Adjust this to control downsampling
#     )
#     print("Processing complete.")