import numpy as np
import sys

if sys.version_info >= (3, 9) and np.lib.NumpyVersion(np.__version__) >= '1.20.0':
    # Fix VTK's deprecated numpy reference
    np.bool = bool  # type: ignore
    np.object = object  # type: ignore
    np.int = int  # type: ignore

import pyvista as pv
import trimesh  # Optional for trimesh integration

# Load mesh (example using trimesh + PyVista conversion)
tmesh = trimesh.load('/home/emu/Documents/surrogate/dataset/meshes/cs1_CU-X405C2_compressed_aligned/2_01_B092746_manual_danger.stl')  # Replace with your mesh file


# Convert to PyVista mesh (CORRECTED)
mesh = pv.PolyData(
    tmesh.vertices,
    faces=np.hstack([
        np.full((len(tmesh.faces), 1), 3),  # VTK triangle cell format
        tmesh.faces
    ])
)


# Example: Extract points with Z > 0
extracted = mesh.extract_points(
    mesh.points[:, 2] > 0,
    adjacent_cells=False,
    include_cells=False
)

# Visualize both meshes
plotter = pv.Plotter()
plotter.add_mesh(mesh, color='tan', opacity=0.3)
plotter.add_points(extracted, color='red', point_size=10)
plotter.show()
