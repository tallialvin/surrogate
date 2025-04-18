import networkx as nx
import numpy as np
import glob
import os
import open3d as o3d
import matplotlib.pyplot as plt
import pickle  # Ensure you import pickle for loading datasets
import re
import math

def load_data_set(directory, file_number):
    # Find the file that starts with the given number
    for filename in os.listdir(directory):
        if filename.startswith(f"{file_number}_") and filename.endswith(".stl"):
            file_path = os.path.join(directory, filename)
            break
    else:
        print(f"Error: No STL file found for number {file_number}.")
        return None

    # Check if the file exists (this is redundant but kept for safety)
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist.")
        return None

    # Load the STL mesh using Open3D
    try:
        mesh = o3d.io.read_triangle_mesh(file_path)
        return mesh
    except Exception as e:
        print(f"Error loading mesh from {file_path}: {str(e)}")
        return None


def process_file(dataset_dir, file_number):
    dataset = dataset_dir
    
    mesh = load_data_set(dataset, file_number)

# メッシュから点群を生成
pcd = stl_mesh.sample_points_uniformly(number_of_points=100000)  # 点の数を調整

# 法線推定
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# 曲率推定（エッジ部分を検出するための指標）
kdtree = o3d.geometry.KDTreeFlann(pcd)
curvatures = []
for i in range(len(pcd.points)):
    [_, idx, _] = kdtree.search_knn_vector_3d(pcd.points[i], 30)
    neighbors = np.asarray(pcd.points)[idx, :]
    cov = np.cov(neighbors.T)
    eigvals, _ = np.linalg.eigh(cov)
    curvature = eigvals[0] / np.sum(eigvals)  # 最小固有値を用いた曲率指標
    curvatures.append(curvature)
curvatures = np.array(curvatures)
# 曲率しきい値でエッジを検出
curvature_threshold = 0.05
edge_indices = np.where(curvatures > curvature_threshold)[0]
non_edge_indices = np.where(curvatures <= curvature_threshold)[0]
# エッジ部分とそれ以外を分ける
edge_points = np.asarray(pcd.points)[edge_indices]
non_edge_points = np.asarray(pcd.points)[non_edge_indices]
# 非エッジ部分をボクセルダウンサンプリング
non_edge_pcd = o3d.geometry.PointCloud()
non_edge_pcd.points = o3d.utility.Vector3dVector(non_edge_points)
downsampled_non_edge_pcd = non_edge_pcd.voxel_down_sample(voxel_size=0.05)
# エッジ部分とダウンサンプリングされた非エッジ部分を統合
final_pcd = downsampled_non_edge_pcd + o3d.geometry.PointCloud(o3d.utility.Vector3dVector(edge_points))
# 結果の保存と可視化
o3d.io.write_point_cloud("downsampled_with_edges.ply", final_pcd)
o3d.visualization.draw_geometries([final_pcd])




def get_available_files(dirpath):
    """ Get a list of available dataset file numbers from the specified directory. """
    file_pattern = os.path.join(dirpath, '*.stl')  # Changed to match .stl files
    files = glob.glob(file_pattern)
    
    file_numbers = []
    for file in files:
        filename = os.path.basename(file)
        # Extract the number before the first underscore
        match = re.match(r'^(\d+)_', filename)
        if match:
            number_str = match.group(1)
            try:
                file_numbers.append(int(number_str))
            except ValueError:
                print(f"Warning: Could not convert '{number_str}' to an integer.")
    
    return sorted(file_numbers)


if __name__ == '__main__':
   
    dataset_dir = '/home/emu/Documents/surrogate/dataset/cs0'
    
    # Get available file numbers from the directory
    available_files = get_available_files(dataset_dir)

    print("Found : ",len(available_files))

    # Process each available file number
    for file_number in available_files:
        process_file(dataset_dir, file_number)