#shortpath2.py
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import Delaunay

# Define custom node data structure
def create_node_data(position, other_data):
    return {'position': position, 'other_data': other_data}

# Modified array_to_graph function with visualization classes A and B
def array_to_graph(arr, base_id, kpairs, knn, nbrs_threshold,
                   nbrs_threshold_step, graph_threshold=np.inf,
                   return_step=False, other_data=None):
    
    G = nx.Graph()
    idx_base = np.arange(arr.shape[0], dtype=int)
    # print(arr.shape[0])
    idx = np.arange(arr.shape[0], dtype=int)
    nbrs = NearestNeighbors(n_neighbors=knn, metric='euclidean', leaf_size=15).fit(arr)
    distances, indices = nbrs.kneighbors(arr)
    indices = indices.astype(int)
    current_idx = [base_id]
    processed_idx = [base_id]
    step_register = np.full(arr.shape[0], np.nan)
    current_step = 0
    step_register[base_id] = current_step

    # Adding the base node with its data
    G.add_node(base_id, **create_node_data(arr[base_id], other_data[base_id] if other_data else None))

    while idx.shape[0] > 0:
        current_step += 1
        if len(current_idx) > 0:
            nn = indices[current_idx]
            dd = distances[current_idx]
            mask1 = np.in1d(nn, processed_idx, invert=True).reshape(nn.shape)
            nntemp = []
            for i, (n, d, g) in enumerate(zip(nn, dd, current_idx)):
                nn_idx = n[mask1[i]][0:kpairs+1]
                dd_idx = d[mask1[i]][0:kpairs+1]
                nntemp.append(nn_idx)
                add_nodes(G, g, nn_idx, dd_idx, arr, other_data, graph_threshold)
            current_idx = np.unique([t2 for t1 in nntemp for t2 in t1])

        elif len(current_idx) == 0:
            idx2 = indices[idx]
            dist2 = distances[idx]
            mask1 = np.in1d(idx2, processed_idx).reshape(idx2.shape)
            mask2 = dist2 < nbrs_threshold
            mask = np.logical_and(mask1, mask2)
            temp_idx = np.unique(np.where(mask)[0])
            current_idx = idx[temp_idx]
            nn = indices[current_idx]
            dd = distances[current_idx]
            mask = np.in1d(nn, processed_idx, invert=True).reshape(nn.shape)
            nntemp = []
            for i, (n, d, g) in enumerate(zip(nn, dd, current_idx)):
                nn_idx = n[mask[i]][0:kpairs+1]
                dd_idx = d[mask[i]][0:kpairs+1]
                add_nodes(G, g, nn_idx, dd_idx, arr, other_data, graph_threshold)
                nn_idx = n[~mask[i]][0:kpairs+1]
                dd_idx = d[~mask[i]][0:kpairs+1]
                add_nodes(G, g, nn_idx, dd_idx, arr, other_data, graph_threshold)
            if len(current_idx) == 0:
                nbrs_threshold += nbrs_threshold_step

        processed_idx = np.append(processed_idx, current_idx)
        processed_idx = np.unique(processed_idx).astype(int)
        idx = idx_base[np.in1d(idx_base, processed_idx, invert=True)]
        current_idx = np.array(current_idx).astype(int)
        step_register[current_idx] = current_step

    if return_step is True:
        return G, step_register
    else:
        return G

def add_nodes(G, base_node, indices, distance, arr, other_data, threshold):
    if base_node not in G.nodes:
        G.add_node(base_node, **create_node_data(arr[base_node], other_data[base_node] if other_data else None))
    
    base_class = G.nodes[base_node]['other_data']['class']
    for c in range(len(indices)):
        if indices[c] not in G:
            G.add_node(indices[c], **create_node_data(arr[indices[c]], other_data[indices[c]] if other_data else None))
        
        # Connect nodes of the same class, regardless of distance
        if G.nodes[indices[c]]['other_data']['class'] == base_class:
            G.add_weighted_edges_from([(base_node, indices[c], distance[c])])

def delaunay_to_graph(arr, other_data=None, graph_threshold=np.inf, node_id_offset=0):
    """
    Compute the 3D Delaunay triangulation of points and convert it to a NetworkX graph.

    Parameters:
    arr (ndarray): A 2D array of shape (n_points, 3) representing the 3D coordinates of points.
    other_data (list): A list of dictionaries containing additional data for each point.
    graph_threshold (float): Maximum edge length to include in the graph.
    node_id_offset (int): Offset to add to node IDs.

    Returns:
    G (networkx.Graph): A NetworkX graph where nodes are points and edges are Delaunay edges.
    """
    # Perform Delaunay triangulation
    tri = Delaunay(arr)
    
    # Create a NetworkX graph
    G = nx.Graph()
    
    # Add nodes with their 3D coordinates and other data
    for i, point in enumerate(arr):
        node_id = i + node_id_offset
        node_data = create_node_data(point, other_data[i] if other_data else None)
        G.add_node(node_id, **node_data)
    
    # Add edges based on the simplices (tetrahedra)
    for simplex in tri.simplices:
        for i in range(len(simplex)):
            for j in range(i + 1, len(simplex)):
                # Calculate Euclidean distance for edge weight
                weight = np.linalg.norm(arr[simplex[i]] - arr[simplex[j]])
                if weight <= graph_threshold:
                    G.add_edge(simplex[i] + node_id_offset, simplex[j] + node_id_offset, weight=weight)
    
    return G

def knn_to_graph(arr, other_data=None, k=5, graph_threshold=np.inf, node_id_offset=0):
    """
    Compute the k-Nearest Neighbors graph from points and convert it to a NetworkX graph.

    Parameters:
    arr (ndarray): A 2D array of shape (n_points, 3) representing the 3D coordinates of points.
    other_data (list): A list of dictionaries containing additional data for each point.
    k (int): Number of nearest neighbors to consider for each point.
    graph_threshold (float): Maximum edge length to include in the graph.
    node_id_offset (int): Offset to add to node IDs.

    Returns:
    G (networkx.Graph): A NetworkX graph where nodes are points and edges connect k-nearest neighbors.
    """
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
    
    #  # Connect disconnected components
    # components = list(nx.connected_components(G))
    # if len(components) > 1:
    #     for i in range(len(components) - 1):
    #         comp1 = list(components[i])[0]
    #         comp2 = list(components[i+1])[0]
    #         point1 = G.nodes[comp1]['position']
    #         point2 = G.nodes[comp2]['position']
    #         distance = np.linalg.norm(np.array(point1) - np.array(point2))
    #         G.add_edge(comp1, comp2, weight=distance)
    

    return G

def knn_mst_graph(arr, other_data=None, k=5, graph_threshold=np.inf, node_id_offset=0):
    """
    Compute a graph using k-Nearest Neighbors and Minimum Spanning Tree to ensure all points are connected.

    Parameters:
    arr (ndarray): A 2D array of shape (n_points, 3) representing the 3D coordinates of points.
    other_data (list): A list of dictionaries containing additional data for each point.
    k (int): Number of nearest neighbors to consider for each point.
    graph_threshold (float): Maximum edge length to include in the k-NN graph.
    node_id_offset (int): Offset to add to node IDs.

    Returns:
    G (networkx.Graph): A NetworkX graph where all nodes are connected.
    """
    # Compute k-Nearest Neighbors
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(arr)
    distances, indices = nbrs.kneighbors(arr)
    
    # Create a NetworkX graph for k-NN
    G_knn = nx.Graph()
    
    # Add nodes with their 3D coordinates and other data
    for i, point in enumerate(arr):
        node_id = i + node_id_offset
        node_data = create_node_data(point, other_data[i] if other_data else None)
        G_knn.add_node(node_id, **node_data)
    
    # Add edges based on k-NN results
    for i, (dist, idx) in enumerate(zip(distances, indices)):
        for j, d in zip(idx[1:], dist[1:]):  # Skip the first neighbor (self)
            if d <= graph_threshold:
                G_knn.add_edge(i + node_id_offset, j + node_id_offset, weight=d)
    
    # Create a complete graph for MST
    G_complete = nx.Graph()
    for i, point1 in enumerate(arr):
        node_id1 = i + node_id_offset
        G_complete.add_node(node_id1, **create_node_data(point1, other_data[i] if other_data else None))
        for j, point2 in enumerate(arr[i+1:], start=i+1):
            node_id2 = j + node_id_offset
            distance = np.linalg.norm(np.array(point1) - np.array(point2))
            G_complete.add_edge(node_id1, node_id2, weight=distance)
    
    # Compute Minimum Spanning Tree
    mst = nx.minimum_spanning_tree(G_complete)
    
    # Combine k-NN and MST graphs
    G_combined = nx.Graph(G_knn)
    G_combined.add_edges_from(mst.edges(data=True))
    
    return G_combined
