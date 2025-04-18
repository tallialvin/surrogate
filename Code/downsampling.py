# Copyright (c) 2018-2019, Matheus Boni Vicari, pc2graph
# All rights reserved.
#
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

__author__ = "Matheus Boni Vicari"
__copyright__ = "Copyright 2018-2019"
__credits__ = ["Matheus Boni Vicari"]
__license__ = "GPL3"
__version__ = "1.0.1"
__maintainer__ = "Matheus Boni Vicari"
__email__ = "matheus.boni.vicari@gmail.com"
__status__ = "Development"


import numpy as np
from scipy.spatial.distance import cdist
import open3d as o3d
from scipy.spatial import ConvexHull




def downsample_cloud(point_cloud, downsample_size, return_indices=False,
                     return_neighbors=False):

    """
    Downsamples a point cloud by voxelizing it and selecting points closest
    to the median coordinate of all points inside each voxel. The remaining
    points can be stored and returned as a dictrionary for later
    use in upsampling back to original input data.

    Parameters
    ----------
    point_cloud : numpy.ndarray
        Three-dimensional (m x n) array of a point cloud, where the
        coordinates are represented in the columns (n) and the points are
        represented in the rows (m).
    downsample_size : float
        Size of the voxels used to sample points into groups and select the
        most central point from. Note that this will not be the final points
        distance from each other, but an approximation.
    return_indices : bool
        Option to return results as downsampled array (False) or the
        indices of downsampled points from original point cloud (True).
    return_neighbors : bool
        Option to return original neighbors of downsampled points (True) or
        not (False). This information can be used to upsample back the
        downsampled indices.

    """

    # Voxelizing input point cloud by truncating coordinates based on
    # downsample_size.
    voxels_ids = (point_cloud / downsample_size).astype(int)
    voxels = {}

    # Looping over each point voxel index. Adds each point index to its
    # voxel key (vid).
    for i, vid in enumerate(voxels_ids):
        if tuple(vid) in voxels:
            voxels[tuple(vid)].append(i)
        else:
            voxels[tuple(vid)] = [i]

    # If return_neighbors is set to True, initialize neighbors_ids dictionary.
    if return_neighbors:
        neighbors_ids = {}

    # Initializing point cloud downsampled indices as array of zeros with
    # length equal to number of voxels.
    pc_downsample_ids = np.zeros(len(voxels.keys()), dtype=int)
    # Looping over each pair of voxel indices and point indices.
    for i, (vid, pids) in enumerate(voxels.items()):
        # Calculating median coordinates of points inside current voxel.
        median_coord = np.median(point_cloud[pids], axis=0)
        # Calculating distance of every point inside current voxel to
        # their median.
        dist = cdist(point_cloud[pids], median_coord.reshape([1, 3]))
        # Sorting indices by distance and selecting closest point as
        # representative of current voxel's center. Assign selected point's
        # index to current index of pc_downsample_ids.
        sort_ids = np.argsort(dist.T)
        pids = np.array(pids).flatten()
        pc_downsample_ids[i] = pids[sort_ids[0][0]]
        # If set to return neighbors indices, assign all remaining points
        # indices to selected center index in neighbors_ids.
        if return_neighbors:
            neighbors_ids[pc_downsample_ids[i]] = pids[sort_ids[0]]

    if return_indices:
        if return_neighbors:
            return pc_downsample_ids, neighbors_ids
        else:
            return pc_downsample_ids
    else:
        if return_neighbors:
            return point_cloud[pc_downsample_ids], neighbors_ids
        else:
            return point_cloud[pc_downsample_ids]


def upsample_cloud(upsample_ids, neighbors_dict):

    """
    Upsample cloud based on downsampling information from 'downsample_cloud'.
    This function will loop over each 'upsample_ids' and retrieve its
    original neighboring points stored in 'neighbors_dict'.

    Parameters
    ----------
    upsample_ids : list
        List of indices in 'neighbors_dict' to upsample.
    neighbors_dict : dict
        Neighbors information provided by 'downsample_cloud' containing
        all the original neighboring points to each point in the downsampled
        cloud.

    Returns
    -------
    upsampled_indices : numpy.ndarray
        Upsampled points from original point cloud.

    """

    # Looping over each index in upsample_ids and retrieving its
    # original neighbors indices.
    ids = [neighbors_dict[i] for i in upsample_ids if i in neighbors_dict]

    return np.unique([i for j in ids for i in j])


def farthest_point_sampling(point_cloud, num_samples, return_indices=False):
    """
    Perform Farthest Point Sampling on the point cloud.

    Parameters
    ----------
    point_cloud : PointCloud or numpy.ndarray
        An instance of the PointCloud class containing the points or a numpy array of points.
    num_samples : int
        The number of points to sample from the point cloud.
    return_indices : bool
        Option to return indices of downsampled points from original point cloud (True)
        or the downsampled array (False).

    Returns
    -------
    numpy.ndarray or tuple
        If return_indices is False, returns a downsampled array of points.
        If return_indices is True, returns a tuple containing:
        - An array of indices of downsampled points.
        - The downsampled array of points.
    """

    # Ensure num_samples is an integer
    while num_samples <= 1:
        num_samples *= 10
    
    num_samples += 5 
    num_samples = int(num_samples)

    # Check if point_cloud is a PointCloud instance or a numpy array
    if isinstance(point_cloud, np.ndarray):
        points = point_cloud  # Use the numpy array directly
    else:
        points = point_cloud.points  # Access points if it's a PointCloud instance

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
    

def no_sampling(point_cloud, return_indices=False):
    """
    Return all points from the point cloud without sampling.

    Parameters
    ----------
    point_cloud : PointCloud or numpy.ndarray
        An instance of the PointCloud class containing the points or a numpy array of points.
    return_indices : bool
        Option to return indices of points from original point cloud (True)
        or just the points (False).

    Returns
    -------
    numpy.ndarray or tuple
        If return_indices is False, returns the array of points.
        If return_indices is True, returns a tuple containing:
        - An array of indices of all points.
        - The array of all points.
    """

    # Check if point_cloud is a PointCloud instance or a numpy array
    if isinstance(point_cloud, np.ndarray):
        points = point_cloud  # Use the numpy array directly
    else:
        points = point_cloud.points  # Access points if it's a PointCloud instance

    if return_indices:
        indices = np.arange(len(points))
        return indices, points
    else:
        return points
    

def edge_ds(point_cloud):

    if isinstance(point_cloud, np.ndarray):
        points = point_cloud  # Use the numpy array directly
    else:
        points = point_cloud.points  # Access points if it's a PointCloud instance

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # pcd = point_cloud  # 点の数を調整
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
    
    final_numpy_array = np.asarray(final_pcd.points)

    return final_numpy_array

def edge_ds_1(point_cloud, voxel_size, curvature_threshold = 0.05):

    if isinstance(point_cloud, np.ndarray):
        points = point_cloud  # Use the numpy array directly
    else:
        points = point_cloud.points  # Access points if it's a PointCloud instance

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

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
    
    edge_indices = np.where(curvatures > curvature_threshold)[0]
    non_edge_indices = np.where(curvatures <= curvature_threshold)[0]
    # エッジ部分とそれ以外を分ける
    edge_points = np.asarray(pcd.points)[edge_indices]
    non_edge_points = np.asarray(pcd.points)[non_edge_indices]
    # 非エッジ部分をボクセルダウンサンプリング
    non_edge_pcd = o3d.geometry.PointCloud()
    non_edge_pcd.points = o3d.utility.Vector3dVector(non_edge_points)
    downsampled_non_edge_pcd = non_edge_pcd.voxel_down_sample(voxel_size)
    # エッジ部分とダウンサンプリングされた非エッジ部分を統合
    final_pcd = downsampled_non_edge_pcd + o3d.geometry.PointCloud(o3d.utility.Vector3dVector(edge_points))
    
    final_numpy_array = np.asarray(final_pcd.points)

    return final_numpy_array

def vol(point_cloud):
    if isinstance(point_cloud, np.ndarray):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
    else:
        pcd = point_cloud

    original_points = np.asarray(pcd.points)

    # Create a ConvexHull object
    hull = ConvexHull(original_points)

    # Calculate the volume
    volume = hull.volume

    return volume


def edge_ds_2(point_cloud, initial_voxel_size=0.02, edge_voxel_size=0.05, curvature_threshold=0.2, min_neighbors=10, min_points=5, max_points=1000):
    # Convert to Open3D PointCloud if it's a numpy array
    
    
    # print("######################max points", max_points)
    if isinstance(point_cloud, np.ndarray):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
    else:
        pcd = point_cloud

    original_points = np.asarray(pcd.points)
    
    # If original point cloud has 6 or fewer points, return all of them
    if len(original_points) <= min_points:
        return original_points

    # Initial voxel downsampling
    pcd_down = pcd.voxel_down_sample(voxel_size=initial_voxel_size)
    points = np.asarray(pcd_down.points)

    # print("# points after voxel downsample : ", len(points))

    
     # Estimate normals
    pcd_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=30))

    # Curvature estimation
    kdtree = o3d.geometry.KDTreeFlann(pcd_down)
    curvatures = []
    for i in range(len(points)):
        [k, idx, _] = kdtree.search_knn_vector_3d(points[i], 30)
        if k < min_neighbors:
            curvatures.append(0)  # Not enough neighbors, assume flat surface
            continue
        neighbors = points[idx, :]
        neighbors = neighbors - np.mean(neighbors, axis=0)  # Center the neighborhood
        cov = np.cov(neighbors.T)
        try:
            eigvals = np.linalg.eigvalsh(cov)
            if np.sum(eigvals) != 0:
                curvature = eigvals[0] / np.sum(eigvals)
            else:
                curvature = 0
        except np.linalg.LinAlgError:
            curvature = 0  # In case of LinAlgError, assume flat surface
        curvatures.append(curvature)
    curvatures = np.array(curvatures)

    # Edge detection
    edge_indices = np.where(curvatures > curvature_threshold)[0]
    non_edge_indices = np.where(curvatures <= curvature_threshold)[0]

    # Separate edge and non-edge points
    edge_points = points[edge_indices]
    non_edge_points = points[non_edge_indices]

    # Voxel downsample non-edge points
    non_edge_pcd = o3d.geometry.PointCloud()
    non_edge_pcd.points = o3d.utility.Vector3dVector(non_edge_points)
    if edge_voxel_size!=0.0:
        downsampled_non_edge_pcd = non_edge_pcd.voxel_down_sample(voxel_size=edge_voxel_size)
    else:
        downsampled_non_edge_pcd = non_edge_pcd

    # Combine edge points and downsampled non-edge points
    final_pcd = downsampled_non_edge_pcd + o3d.geometry.PointCloud(o3d.utility.Vector3dVector(edge_points))

    final_points = np.asarray(final_pcd.points)

    # If final points are fewer than min_points, add more points from the original
    if len(final_points) < min_points:
        # print("#")
        final_points = farthest_point_sampling(original_points, min(len(original_points), max_points*0.001))
        # print("# points after edge-aware downsampling:", len(final_points))
        return final_points

    

    # If the number of points is still more than max_points, recursively call the function
    # if len(final_points) > max_points:
    #     # if max_points < min_points:
    #     #     max_points = min_points
    #     # input("here")
    #     # print("Recursively calling edge_ds_2 with increased voxel sizes")
    #     return edge_ds_2(final_points, 
    #                      initial_voxel_size=initial_voxel_size*1.1, 
    #                      edge_voxel_size=edge_voxel_size*1.1, 
    #                      curvature_threshold=curvature_threshold,
    #                      min_neighbors=min_neighbors,
    #                      min_points=min_points,
    #                      max_points=max_points)

    return final_points

def edge_ds_3(point_cloud, initial_voxel_size=0.02, edge_voxel_size=0.05, curvature_threshold=0.2,
              min_neighbors=10, min_points=5, max_points=1000):
    # Convert to Open3D PointCloud if it's a numpy array
    if isinstance(point_cloud, np.ndarray):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
    else:
        pcd = point_cloud

    original_points = np.asarray(pcd.points)

    if len(original_points) < min_points:
        final_points = farthest_point_sampling(original_points, len(original_points) * 0.001)
        # print("# points after edge-aware downsampling:", len(final_points))
        return final_points

    # If original point cloud has 6 or fewer points, return all of them
    if len(original_points) <= min_points:
        return original_points

    # Initial voxel downsampling
    pcd_down = pcd.voxel_down_sample(voxel_size=initial_voxel_size)
    points = np.asarray(pcd_down.points)

    # print("# points after voxel downsample : ", len(points))

    # Estimate normals
    pcd_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Curvature estimation
    kdtree = o3d.geometry.KDTreeFlann(pcd_down)
    curvatures = []
    
    for i in range(len(points)):
        [k, idx, _] = kdtree.search_knn_vector_3d(points[i], 30)
        if k < min_neighbors:
            curvatures.append(0)  # Not enough neighbors, assume flat surface
            continue
        neighbors = points[idx, :]
        neighbors -= np.mean(neighbors, axis=0)  # Center the neighborhood
        cov = np.cov(neighbors.T)
        
        try:
            eigvals = np.linalg.eigvalsh(cov)
            curvature = eigvals[0] / np.sum(eigvals) if np.sum(eigvals) != 0 else 0
        except np.linalg.LinAlgError:
            curvature = 0  # In case of LinAlgError

        curvatures.append(curvature)

    curvatures = np.array(curvatures)

    # Edge detection
    edge_indices = np.where(curvatures > curvature_threshold)[0]
    
    # Separate edge points
    edge_points = points[edge_indices]

    # Compute convex hull to keep only outer points
    hull, _ = pcd_down.compute_convex_hull()  # Updated line to compute convex hull
    hull_vertices = np.asarray(hull.vertices)

    # Combine edge points and hull vertices while avoiding duplicates
    final_points_set = set(map(tuple, hull_vertices)) | set(map(tuple, edge_points))
    
    final_points = np.array(list(final_points_set))

    # If final points are fewer than min_points, add more from the original
    if len(final_points) < min_points:
        final_points = farthest_point_sampling(original_points, len(original_points) * 0.001)
        # print("# points after edge-aware downsampling:", len(final_points))
        return final_points

    # If the number of points is still more than max_points, recursively call the function
    if len(final_points) > max_points:
        # print("Recursively calling edge_ds_2 with increased voxel sizes")
        return edge_ds_2(final_points,
                         initial_voxel_size=initial_voxel_size * 1.5,
                         edge_voxel_size=edge_voxel_size * 1.5,
                         curvature_threshold=curvature_threshold,
                         min_neighbors=min_neighbors,
                         min_points=min_points,
                         max_points=max_points)

    return final_points

def edge_ds_4(point_cloud, initial_voxel_size=0.02, edge_voxel_size=0.05, curvature_threshold=0.2,
              min_neighbors=10, min_points=5, max_points=1000):
    # Convert to Open3D PointCloud if it's a numpy array
    if isinstance(point_cloud, np.ndarray):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
    else:
        pcd = point_cloud

    original_points = np.asarray(pcd.points)

    if len(original_points) < min_points:
        final_points = farthest_point_sampling(original_points, len(original_points) * 0.001)
        # print("# points after edge-aware downsampling:", len(final_points))
        return final_points

    # If original point cloud has 6 or fewer points, return all of them
    if len(original_points) <= min_points:
        return original_points

    # Initial voxel downsampling
    pcd_down = pcd.voxel_down_sample(voxel_size=initial_voxel_size)
    points = np.asarray(pcd_down.points)

    # print("# points after voxel downsample : ", len(points))

    # Estimate normals
    pcd_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Curvature estimation
    kdtree = o3d.geometry.KDTreeFlann(pcd_down)
    curvatures = []
    
    for i in range(len(points)):
        [k, idx, _] = kdtree.search_knn_vector_3d(points[i], 30)
        if k < min_neighbors:
            curvatures.append(0)  # Not enough neighbors, assume flat surface
            continue
        neighbors = points[idx, :]
        neighbors -= np.mean(neighbors, axis=0)  # Center the neighborhood
        cov = np.cov(neighbors.T)
        
        try:
            eigvals = np.linalg.eigvalsh(cov)
            curvature = eigvals[0] / np.sum(eigvals) if np.sum(eigvals) != 0 else 0
        except np.linalg.LinAlgError:
            curvature = 0  # In case of LinAlgError

        curvatures.append(curvature)

    curvatures = np.array(curvatures)

    # Edge detection
    edge_indices = np.where(curvatures > curvature_threshold)[0]
    
    # Separate edge points
    edge_points = points[edge_indices]

    # Compute convex hull to keep only outer points
    hull, _ = pcd_down.compute_convex_hull()  # Updated line to compute convex hull
    hull_vertices = np.asarray(hull.vertices)

    # Combine edge points and hull vertices while avoiding duplicates
    final_points_set = set(map(tuple, hull_vertices)) | set(map(tuple, edge_points))
    
    final_points = np.array(list(final_points_set))

    # If final points are fewer than min_points, add more from the original
    if len(final_points) < min_points:
        final_points = farthest_point_sampling(original_points, len(original_points) * 0.001)
        # print("# points after edge-aware downsampling:", len(final_points))
        return final_points

    # If the number of points is still more than max_points, recursively call the function
    # if len(final_points) > max_points:
    #     # print("Recursively calling edge_ds_2 with increased voxel sizes")
    #     return edge_ds_2(final_points,
    #                      initial_voxel_size=initial_voxel_size * 1.5,
    #                      edge_voxel_size=edge_voxel_size * 1.5,
    #                      curvature_threshold=curvature_threshold,
    #                      min_neighbors=min_neighbors,
    #                      min_points=min_points,
    #                      max_points=max_points)

    return final_points


def edge_ds_2_with_hpr(point_cloud, initial_voxel_size=0.02, edge_voxel_size=0.05, curvature_threshold=0.2, min_neighbors=10, min_points=5, max_points=1000, camera_distance=None, num_views=8, hpr_radius=None):
    # Convert to Open3D PointCloud if it's a numpy array
    if isinstance(point_cloud, np.ndarray):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
    else:
        pcd = point_cloud

    original_points = np.asarray(pcd.points)
    
    # If original point cloud has 6 or fewer points, return all of them
    if len(original_points) <= min_points:
        return original_points

    # Initial voxel downsampling
    pcd_down = pcd.voxel_down_sample(voxel_size=initial_voxel_size)
    points = np.asarray(pcd_down.points)

    # Estimate normals
    pcd_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Curvature estimation
    kdtree = o3d.geometry.KDTreeFlann(pcd_down)
    curvatures = []
    for i in range(len(points)):
        [k, idx, _] = kdtree.search_knn_vector_3d(points[i], 30)
        if k < min_neighbors:
            curvatures.append(0)  # Not enough neighbors, assume flat surface
            continue
        neighbors = points[idx, :]
        neighbors = neighbors - np.mean(neighbors, axis=0)  # Center the neighborhood
        cov = np.cov(neighbors.T)
        try:
            eigvals = np.linalg.eigvalsh(cov)
            if np.sum(eigvals) != 0:
                curvature = eigvals[0] / np.sum(eigvals)
            else:
                curvature = 0
        except np.linalg.LinAlgError:
            curvature = 0  # In case of LinAlgError, assume flat surface
        curvatures.append(curvature)
    curvatures = np.array(curvatures)

    # Edge detection
    edge_indices = np.where(curvatures > curvature_threshold)[0]
    non_edge_indices = np.where(curvatures <= curvature_threshold)[0]

    # Separate edge and non-edge points
    edge_points = points[edge_indices]
    non_edge_points = points[non_edge_indices]

    # Voxel downsample non-edge points
    non_edge_pcd = o3d.geometry.PointCloud()
    non_edge_pcd.points = o3d.utility.Vector3dVector(non_edge_points)
    downsampled_non_edge_pcd = non_edge_pcd.voxel_down_sample(voxel_size=edge_voxel_size)

    # Combine edge points and downsampled non-edge points
    final_pcd = downsampled_non_edge_pcd + o3d.geometry.PointCloud(o3d.utility.Vector3dVector(edge_points))

    # Apply Hidden Point Removal for 360-degree view
    if camera_distance is not None:
        if hpr_radius is None:
            # If radius is not provided, estimate it based on the point cloud size
            hpr_radius = np.linalg.norm(np.asarray(final_pcd.get_max_bound()) - np.asarray(final_pcd.get_min_bound())) * 100

        # Calculate the center of the point cloud
        center = final_pcd.get_center()

        # Generate camera positions around the point cloud
        angles = np.linspace(0, 2*np.pi, num_views, endpoint=False)
        visible_indices = set()

        for angle in angles:
            camera_position = center + camera_distance * np.array([np.cos(angle), np.sin(angle), 0])
            _, pt_map = final_pcd.hidden_point_removal(camera_position, hpr_radius)
            visible_indices.update(pt_map)

        final_pcd = final_pcd.select_by_index(list(visible_indices))

    final_points = np.asarray(final_pcd.points)

    # If final points are fewer than min_points, add more points from the original
    if len(final_points) < min_points:
        print("here 1")
        final_points = farthest_point_sampling(original_points, min(len(original_points), max_points))

    # Limit the number of points to max_points
    # if len(final_points) > max_points:
    #     print("here 2")
    #     final_points = farthest_point_sampling(final_points, max_points)

    return final_points
