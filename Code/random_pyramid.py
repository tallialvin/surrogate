import pyvista as pv
import numpy as np
np.bool = bool
# Create a random pyramid
# Create a random pyramid
# pyramid = pv.Pyramid()
# pyramid.points += np.random.rand(*pyramid.points.shape) * 0.1

# # Create a plotter
# plotter = pv.Plotter()

# # Add the pyramid surface with transparency
# plotter.add_mesh(pyramid, color='white', opacity=0.2)

# # Add the edges separately with full opacity
# edges = pyramid.extract_all_edges()
# plotter.add_mesh(edges, color='blue', line_width=5)

# # Color all vertices red
# vertices = pv.PolyData(pyramid.points)
# plotter.add_mesh(vertices, color='red', point_size=15, render_points_as_spheres=True)

# # Show the plot
# plotter.show()


# # Create a random pyramid
# pyramid = pv.Pyramid()
# pyramid.points += np.random.rand(*pyramid.points.shape) * 0.1

# # Extract edges
# edges = pyramid.extract_all_edges()

# # Calculate edge lengths
# edge_lengths = []
# for i in range(edges.n_cells):
#     edge = edges.extract_cells(i)
#     start_point = edge.points[0]
#     end_point = edge.points[1]
#     length = np.linalg.norm(end_point - start_point)
#     edge_lengths.append(length)

# edge_lengths = np.array(edge_lengths)

# # Parameters for point sampling
# base_points = 5  # Minimum number of points per edge
# length_scale = 0.1  # Length per additional point
# max_points = 20  # Maximum number of points per edge

# # Sample points along edges
# points = []
# for i in range(edges.n_cells):
#     edge = edges.extract_cells(i)
#     start_point = edge.points[0]
#     end_point = edge.points[1]
#     edge_length = edge_lengths[i]
    
#     num_points = max(2, min(max_points, base_points + int(edge_length / length_scale)))
    
#     for j in range(num_points):
#         t = j / (num_points - 1)
#         point = start_point * (1 - t) + end_point * t
#         points.append(point)

# # Create a PolyData object from the sampled points
# edge_points = pv.PolyData(np.array(points))

# # Create a plotter
# plotter = pv.Plotter()

# # Add the pyramid surface with transparency
# plotter.add_mesh(pyramid, color='white', opacity=0.2)

# # Add the sampled edge points
# plotter.add_mesh(edge_points, color='red', point_size=13, render_points_as_spheres=True)
# plotter.add_mesh(edges, color='blue', line_width=2, opacity=0.4)

# # Color all vertices red
# vertices = pv.PolyData(pyramid.points)
# plotter.add_mesh(vertices, color='red', point_size=13, render_points_as_spheres=True)

# # Show the plot
# plotter.show()

# Create a hexagonal prism
# Manually create a hexagonal prism vertices
vertices = np.array([
    # Bottom hexagon vertices
    [1, 0, 0], [0.5, 0.866, 0], [-0.5, 0.866, 0], 
    [-1, 0, 0], [-0.5, -0.866, 0], [0.5, -0.866, 0],
    
    # Top hexagon vertices (same pattern, but with height)
    [1, 0, 1], [0.5, 0.866, 1], [-0.5, 0.866, 1], 
    [-1, 0, 1], [-0.5, -0.866, 1], [0.5, -0.866, 1]
])

# Create faces for the hexagonal prism
faces = [
    6, 0, 1, 2, 3, 4, 5,  # Bottom face
    6, 6, 7, 8, 9, 10, 11,  # Top face
    4, 0, 6, 7, 1,  # Side faces
    4, 1, 7, 8, 2,
    4, 2, 8, 9, 3,
    4, 3, 9, 10, 4,
    4, 4, 10, 11, 5,
    4, 5, 11, 6, 0
]

def create_hexagonal_diamond():
    # Hexagonal base vertices (bottom)
    bottom = np.array([
        [1, 0, 0],      # Right vertex
        [0.5, 0.866, 0],  # Top-right vertex
        [-0.5, 0.866, 0], # Top-left vertex
        [-1, 0, 0],     # Left vertex
        [-0.5, -0.866, 0],# Bottom-left vertex
        [0.5, -0.866, 0]  # Bottom-right vertex
    ])

    # Apex vertex (top point of diamond)
    apex = np.array([[0, 0, 1.5]])

    # Bottom base vertex (opposite of apex)
    bottom_center = np.array([[0, 0, -0.5]])

    # Combine vertices
    vertices = np.vstack([bottom, apex, bottom_center])

    # Define faces (hexagonal base, triangular sides, bottom cap)
    faces = [
        # Hexagonal base (6 vertices)
        6, 0, 1, 2, 3, 4, 5,
        
        # Triangular sides to apex
        3, 0, 6, 1,
        3, 1, 6, 2,
        3, 2, 6, 3,
        3, 3, 6, 4,
        3, 4, 6, 5,
        3, 5, 6, 0,
        
        # Bottom cap
        6, 0, 5, 4, 3, 2, 1
    ]

    return pv.PolyData(vertices, faces)

# prism = pv.PolyData(vertices, faces)

prism = create_hexagonal_diamond()

prism.points += np.random.rand(*prism.points.shape) * 0.1

# Extract edges
edges = prism.extract_all_edges()

# Calculate edge lengths
edge_lengths = []
for i in range(edges.n_cells):
    edge = edges.extract_cells(i)
    start_point = edge.points[0]
    end_point = edge.points[1]
    length = np.linalg.norm(end_point - start_point)
    edge_lengths.append(length)

edge_lengths = np.array(edge_lengths)

# Parameters for point sampling
base_points = 5  # Minimum number of points per edge
length_scale = 0.1  # Length per additional point
max_points = 20  # Maximum number of points per edge

# Sample points along edges
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

# Create a PolyData object from the sampled points
edge_points = pv.PolyData(np.array(points))

# Create a plotter
plotter = pv.Plotter()

# Add the prism surface with transparency
plotter.add_mesh(prism, color='white', opacity=0.2)

# Add the sampled edge points
plotter.add_mesh(edge_points, color='red', point_size=13, render_points_as_spheres=True)
plotter.add_mesh(edges, color='blue', line_width=2, opacity=0.4)

# Color all vertices red
vertices = pv.PolyData(prism.points)
plotter.add_mesh(vertices, color='red', point_size=13, render_points_as_spheres=True)

# Show the plot
plotter.show()
