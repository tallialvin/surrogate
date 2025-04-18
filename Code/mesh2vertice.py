import pyvista as pv
import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import matplotlib.colors as mcolors
import trimesh


# Define custom node data structure

def create_node_data(point):
    return {'x': point[0], 'y': point[1], 'z': point[2]}

def knn_to_graph_BACKUP(arr, k=5, graph_threshold=np.inf, node_id_offset=0):
    # Compute k-Nearest Neighbors
    # print(len(arr))

    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(arr)
    distances, indices = nbrs.kneighbors(arr)
    
    # Create a NetworkX graph
    G = nx.Graph()
    
    # Add nodes with their 3D coordinates and other data
    for i, point in enumerate(arr):
        node_id = i + node_id_offset
        node_data = create_node_data(point, other_data[i] if other_data else None)
        G.add_node(node_id, **node_data)
    
    # Add edges based on k-NN results
    for i, (dist, idx) in enumerate(zip(distances, indices)):
        for j, d in zip(idx[1:], dist[1:]):  # Skip the first neighbor (self)
            if d <= graph_threshold:
                G.add_edge(i + node_id_offset, j + node_id_offset, weight=d)

    return G


def knn_to_graph(arr, k =5):
    # Compute k-Nearest Neighbors
    

    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(arr)
    distances, indices = nbrs.kneighbors(arr)
    
    # Create a NetworkX graph
    G = nx.Graph()
    
    # Create K-NN graph
    G = nx.Graph()
    for i, point in enumerate(arr):
        G.add_node(i, **create_node_data(point))
    
    # Add K-NN edges
    for i, (dist, idx) in enumerate(zip(distances, indices)):
        for j, d in zip(idx[1:], dist[1:]):  # Skip first index (self)
            G.add_edge(i, j, weight=d)
    
    return G

def farthest_point_sampling(point_cloud, num_samples, return_indices=False):
    
    # Ensure num_samples is an integer
    while num_samples <= 1:
        num_samples *= 10
    
    num_samples += 5 
    num_samples = int(num_samples)
    # print(num_samples)
    # Check if point_cloud is a PointCloud instance or a numpy array
    # if isinstance(point_cloud, np.ndarray):
    #     points = point_cloud  # Use the numpy array directly
    # else:
    #     points = point_cloud.points  # Access points if it's a PointCloud instance
    points = point_cloud
    # Randomly select the first point
    sampled_indices = np.zeros(num_samples, dtype=int)
    sampled_indices[0] = np.random.randint(0, len(points))

    # Create an array to store distances
    distances = np.full(len(points), np.inf)

    # Loop to select remaining points
    for i in range(1, num_samples):
        current_point = points[sampled_indices[i - 1]]
        
        # Calculate distances from the current point to all other points
        dists = np.linalg.norm(points - current_point, axis=1)
        
        # Update distances with minimum distance found
        distances = np.minimum(distances, dists)

        # Select the next farthest point based on max distance
        sampled_indices[i] = np.argmax(distances)

    # Get the sampled points from the original point cloud using indices
    sampled_points = points[sampled_indices]
    
    if return_indices:
        return sampled_indices, sampled_points
    else:
        return sampled_points
    

# Function to extract surface points from a mesh file
def extract_surface_points(mesh_file):
    mesh = pv.read(mesh_file)
    surface = mesh.extract_surface()
    print(type(surface.points))
    return surface.points

def extract_trimesh(mesh_file):
    # Load the mesh file
    mesh = trimesh.load_mesh(mesh_file)
    
    # If the mesh is not already a triangular mesh, triangulate it
    if not mesh.is_triangular:
        mesh = mesh.triangulate()
    
    # Extract the surface if it's not already a surface mesh
    if not mesh.is_watertight:
        mesh = mesh.extract_surface()
    
    # Create a new Trimesh object with the surface data
    surface_trimesh = trimesh.Trimesh(
        vertices=mesh.vertices,
        faces=mesh.faces,
        face_normals=mesh.face_normals
    )
    
    return surface_trimesh

def extract_trimesh_1(mesh_file):

    mesh = trimesh.load(mesh_file)

    # Extract all vertices (points on the surface of the mesh)
    vertices = mesh.vertices  # This is a NumPy array of all vertex coordinates
    return vertices


def process_stl_file(filepath, ratio = 0.001):
    pcd = extract_trimesh_1(filepath)
    # pcd = farthest_point_sampling(pcd, int(pcd.shape[0]*ratio))
    return pcd

def vis_comb_net(combined_graph, col_target='red', col_obstacle='blue'):
    """
    Visualize a combined network with nodes and edges colored by graph_type
    
    Parameters:
    - combined_graph: NetworkX graph with nodes containing x,y,z coordinates
    - col_target: color for 'object' nodes/edges (default: red)
    - col_obstacle: color for 'obstacle' nodes/edges (default: blue)
    """
    # Create plotter
    plotter = pv.Plotter()
    
    # Extract points from x,y,z coordinates
    points = np.array([[data['x'], data['y'], data['z']] 
                      for _, data in combined_graph.nodes(data=True)])
    point_cloud = pv.PolyData(points)
    
    # Convert colors to RGB
    rgb_target = np.array(mcolors.to_rgb(col_target))
    rgb_obstacle = np.array(mcolors.to_rgb(col_obstacle))
    
    # Create node colors
    node_colors = np.array([rgb_target if data['graph_type'] == 'object' else rgb_obstacle 
                          for _, data in combined_graph.nodes(data=True)])
    
    # Add points to plotter
    plotter.add_mesh(point_cloud, scalars=node_colors, rgb=True, point_size=5)
    
    # Create edges with matching colors
    if combined_graph.edges():
        lines = []
        line_colors = []
        for u, v in combined_graph.edges():
            lines.extend([2, u, v])
            u_type = combined_graph.nodes[u]['graph_type']
            line_colors.append(rgb_target if u_type == 'object' else rgb_obstacle)
        
        # Create line mesh
        line_poly = pv.PolyData()
        line_poly.points = points
        line_poly.lines = np.array(lines)
        plotter.add_mesh(line_poly, scalars=np.array(line_colors), rgb=True)
    
    # Finalize visualization
    plotter.view_isometric()
    plotter.set_background('white')
    plotter.show()



def vis_comb_net_1(combined_graph, col_target='red', col_obstacle='blue'):
    """
    Visualize a combined network with nodes and edges colored by graph_type
    
    Parameters:
    - combined_graph: NetworkX graph with nodes containing x,y,z coordinates
    - col_target: color for 'object' nodes/edges (name or hex/rgb tuple)
    - col_obstacle: color for 'obstacle' nodes/edges (name or hex/rgb tuple)
    """
    # Create plotter
    plotter = pv.Plotter()
    
    # Extract points from x,y,z coordinates
    points = []
    for _, data in combined_graph.nodes(data=True):
        points.append([data['x'], data['y'], data['z']])
    points = np.array(points)
    
    # Add points to plotter with colors
    node_colors = []
    for _, data in combined_graph.nodes(data=True):
        if data['graph_type'] == 'object':
            node_colors.append(mcolors.to_rgb(col_target))
        else:
            node_colors.append(mcolors.to_rgb(col_obstacle))
    node_colors = np.array(node_colors)
    
    point_cloud = pv.PolyData(points)
    plotter.add_mesh(point_cloud, scalars=node_colors, rgb=True, point_size=5)
    
    # Create lines
    lines = []
    line_colors = []
    for u, v in combined_graph.edges():
        lines.extend([2, u, v])
        u_type = combined_graph.nodes[u]['graph_type']
        v_type = combined_graph.nodes[v]['graph_type']
        
        if u_type == v_type:
            line_colors.append(mcolors.to_rgb(col_target) if u_type == 'object' else mcolors.to_rgb(col_obstacle))
        else:
            line_colors.append(mcolors.to_rgb('purple'))
    
    if lines:  # Only if edges exist
        lines = np.array(lines)
        line_colors = np.array(line_colors)
        line_poly = pv.PolyData()
        line_poly.points = points
        line_poly.lines = lines
        plotter.add_mesh(line_poly, scalars=line_colors, rgb=True)
    
    # Finalize visualization
    plotter.view_isometric()
    plotter.set_background('white')
    plotter.show()

def combine_graphs_with_offset(G1, G2):
    """Combine two graphs while keeping them visually separated"""
    # Offset node IDs of G2 to avoid collisions
    offset = len(G1.nodes())
    G2_offset = nx.relabel_nodes(G2, {n: n+offset for n in G2.nodes()})
    
    # Combine graphs
    combined = nx.compose(G1, G2_offset)
    
    # Add original graph type as node attribute
    for n in combined.nodes():
        if n < offset:
            combined.nodes[n]['source_graph'] = 'G1'
        else:
            combined.nodes[n]['source_graph'] = 'G2'
    
    return combined