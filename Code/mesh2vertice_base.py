import numpy as np
import sys

if sys.version_info >= (3, 9) and np.lib.NumpyVersion(np.__version__) >= '1.20.0':
    # Fix VTK's deprecated numpy reference
    np.bool = bool  # type: ignore
    np.object = object  # type: ignore
    np.int = int  # type: ignore

import trimesh
import pyvista as pv

# Load the mesh
mesh = trimesh.load('/home/emu/Documents/surrogate/dataset/meshes/cs0_wrc18_rubber_band_drive_unit_ext/13_MBGA30-2_ignore.stl')

# Extract all vertices (points on the surface of the mesh)
vertices = mesh.vertices  # This is a NumPy array of all vertex coordinates

print(f"Number of vertices: {len(vertices)}")
print(f"First 5 vertices:\n{vertices[:5]}")
print(type(vertices))

# Convert to PyVista for visualization
pv_mesh = pv.PolyData(mesh.vertices, faces=np.hstack([np.full((len(mesh.faces), 1), 3), mesh.faces]))
points_cloud = pv.PolyData(vertices)  # Create a point cloud for visualization
print(type(points_cloud))
# Visualize the mesh and its vertices
plotter = pv.Plotter()
plotter.add_mesh(pv_mesh, style='wireframe', color='gray', label='Mesh Surface')
plotter.add_points(points_cloud, color='red', point_size=10, render_points_as_spheres=True, label='Surface Vertices')
plotter.add_legend()
plotter.show()

# Optionally save the vertices to a CSV file
# np.savetxt('mesh_vertices.csv', vertices, delimiter=',', header='x,y,z')
