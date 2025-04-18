import open3d as o3d
import numpy as np
# STLファイルを読み込む
stl_mesh = o3d.io.read_triangle_mesh("/home/emu/Desktop/new_had/had/docker/dataset/meshes/cs1_CU-X405C2_compressed_aligned/1_01_CHASSIS-X_graspable_base.stl")
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