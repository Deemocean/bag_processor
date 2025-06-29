#!/usr/bin/env python3
"""
Extract ground manifold from race track point cloud data.
Uses RANSAC plane segmentation followed by region growing to extract ground points.
"""

import argparse
import numpy as np
import open3d as o3d
from pathlib import Path
import time
from tqdm import tqdm


def extract_ground_ransac(pcd, distance_threshold=0.1, ransac_n=3, num_iterations=1000):
    """
    Extract ground plane using RANSAC algorithm.
    
    Args:
        pcd: Open3D point cloud object
        distance_threshold: Maximum distance from point to plane for inlier consideration
        ransac_n: Number of points to sample for generating hypothetical plane
        num_iterations: Number of RANSAC iterations
    
    Returns:
        ground_pcd: Point cloud containing only ground points
        non_ground_pcd: Point cloud containing non-ground points
        plane_model: [a, b, c, d] plane equation coefficients
    """
    print(f"Running RANSAC with distance threshold: {distance_threshold}")
    
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations
    )
    
    ground_pcd = pcd.select_by_index(inliers)
    non_ground_pcd = pcd.select_by_index(inliers, invert=True)
    
    print(f"Plane equation: {plane_model[0]:.3f}x + {plane_model[1]:.3f}y + {plane_model[2]:.3f}z + {plane_model[3]:.3f} = 0")
    print(f"Ground points: {len(inliers)} / {len(pcd.points)} ({100*len(inliers)/len(pcd.points):.1f}%)")
    
    return ground_pcd, non_ground_pcd, plane_model


def extract_ground_progressive(pcd, initial_distance=0.5, max_iterations=20, angle_threshold=15.0, growth_factor=1.1):
    """
    Progressive morphological filter for ground extraction.
    Optimized for smooth race tracks with elevation changes.
    
    Args:
        pcd: Open3D point cloud object
        initial_distance: Initial distance threshold
        max_iterations: Maximum number of iterations
        angle_threshold: Maximum angle difference (degrees) for ground surface
        growth_factor: Factor to increase distance threshold each iteration
    
    Returns:
        ground_pcd: Extracted ground points
        non_ground_pcd: Non-ground points
    """
    points = np.asarray(pcd.points)
    
    # Sort points by height (z-coordinate)
    sorted_indices = np.argsort(points[:, 2])
    
    # Initialize with lowest points as seeds - use more seeds for race tracks
    seed_percentage = 0.05  # 5% of lowest points as initial seeds
    seed_count = max(100, int(seed_percentage * len(points)))
    ground_indices = sorted_indices[:seed_count]
    
    distance_threshold = initial_distance
    prev_ground_count = len(ground_indices)
    
    for iteration in tqdm(range(max_iterations), desc="Progressive morphological filter"):
        # Fit plane to current ground points
        ground_points = points[ground_indices]
        
        if len(ground_points) < 3:
            break
        
        # For race tracks, use local plane fitting in spatial cells
        # Divide space into grid cells and fit local planes
        xy_points = ground_points[:, :2]
        
        # Determine grid size based on point cloud extent
        xy_min = xy_points.min(axis=0)
        xy_max = xy_points.max(axis=0)
        grid_size = 10.0  # 10m x 10m cells
        
        grid_dims = ((xy_max - xy_min) / grid_size).astype(int) + 1
        
        # Create mask for all points
        new_ground_mask = np.zeros(len(points), dtype=bool)
        
        # Process each grid cell
        total_cells = grid_dims[0] * grid_dims[1]
        with tqdm(total=total_cells, desc=f"Iteration {iteration}: Processing grid cells", leave=False) as pbar:
            for i in range(grid_dims[0]):
                for j in range(grid_dims[1]):
                    pbar.update(1)
                    # Define cell bounds
                    cell_min = xy_min + np.array([i, j]) * grid_size
                    cell_max = cell_min + grid_size
                    
                    # Find points in this cell
                    in_cell = ((points[:, 0] >= cell_min[0]) & (points[:, 0] < cell_max[0]) &
                              (points[:, 1] >= cell_min[1]) & (points[:, 1] < cell_max[1]))
                    
                    # Find ground points in this cell
                    cell_ground_mask = np.zeros(len(points), dtype=bool)
                    cell_ground_mask[ground_indices] = True
                    cell_ground_points = points[in_cell & cell_ground_mask]
                    
                    if len(cell_ground_points) < 3:
                        continue
                    
                    # Fit local plane
                    centroid = np.mean(cell_ground_points, axis=0)
                    centered = cell_ground_points - centroid
                    cov = np.cov(centered.T)
                    _, eigenvectors = np.linalg.eigh(cov)
                    normal = eigenvectors[:, 0]  # Smallest eigenvalue corresponds to normal
                    
                    # Ensure normal points up
                    if normal[2] < 0:
                        normal = -normal
                    
                    # Check angle constraint for this local region
                    angle = np.degrees(np.arccos(np.clip(normal[2], -1, 1)))
                    if angle > angle_threshold:
                        continue
                    
                    # Compute distances to local plane for points in cell
                    d = -np.dot(normal, centroid)
                    cell_points = points[in_cell]
                    cell_indices = np.where(in_cell)[0]
                    
                    if len(cell_points) > 0:
                        distances = np.abs(np.dot(cell_points, normal) + d)
                        local_ground = distances < distance_threshold
                        new_ground_mask[cell_indices[local_ground]] = True
        
        # Find new ground points
        new_ground_indices = np.where(new_ground_mask)[0]
        
        # Check convergence
        if len(new_ground_indices) <= prev_ground_count * 1.01:  # Less than 1% growth
            print(f"Converged at iteration {iteration}")
            break
        
        ground_indices = new_ground_indices
        prev_ground_count = len(ground_indices)
        distance_threshold *= growth_factor
        
        print(f"Iteration {iteration}: {len(ground_indices)} ground points, threshold: {distance_threshold:.3f}m")
    
    # Create output point clouds
    ground_mask = np.zeros(len(points), dtype=bool)
    ground_mask[ground_indices] = True
    
    ground_pcd = pcd.select_by_index(np.where(ground_mask)[0])
    non_ground_pcd = pcd.select_by_index(np.where(~ground_mask)[0])
    
    print(f"Progressive filter: {len(ground_indices)} / {len(points)} ground points ({100*len(ground_indices)/len(points):.1f}%)")
    
    return ground_pcd, non_ground_pcd


def extract_ground_voxel(pcd, voxel_size=0.5, height_threshold=0.3):
    """
    Voxel-based ground extraction.
    Divides space into voxels and finds lowest points in each voxel.
    
    Args:
        pcd: Open3D point cloud object
        voxel_size: Size of voxels for spatial division
        height_threshold: Height difference threshold from lowest point in voxel
    
    Returns:
        ground_pcd: Extracted ground points
        non_ground_pcd: Non-ground points
    """
    points = np.asarray(pcd.points)
    
    # Compute voxel indices
    min_bound = points.min(axis=0)
    voxel_indices = ((points - min_bound) / voxel_size).astype(int)
    
    # Find unique voxels
    unique_voxels, inverse_indices = np.unique(
        voxel_indices, axis=0, return_inverse=True
    )
    
    ground_mask = np.zeros(len(points), dtype=bool)
    
    # Process each voxel
    for i in range(len(unique_voxels)):
        voxel_mask = inverse_indices == i
        voxel_points = points[voxel_mask]
        
        if len(voxel_points) > 0:
            min_z = voxel_points[:, 2].min()
            
            # Mark points close to minimum height as ground
            voxel_indices_in_original = np.where(voxel_mask)[0]
            height_diff = voxel_points[:, 2] - min_z
            ground_in_voxel = height_diff < height_threshold
            
            ground_mask[voxel_indices_in_original[ground_in_voxel]] = True
    
    ground_pcd = pcd.select_by_index(np.where(ground_mask)[0])
    non_ground_pcd = pcd.select_by_index(np.where(~ground_mask)[0])
    
    print(f"Voxel filter: {ground_mask.sum()} / {len(points)} ground points ({100*ground_mask.sum()/len(points):.1f}%)")
    
    return ground_pcd, non_ground_pcd


def extract_ground_hybrid_fast(pcd, voxel_size=5.0, height_percentile=5, local_radius=3.0, slope_threshold=30.0):
    """
    Fast hybrid method optimized for race tracks.
    Uses 2D grid projection and local height analysis.
    
    Args:
        pcd: Open3D point cloud object
        voxel_size: Size of 2D grid cells for segmentation
        height_percentile: Percentile of heights to consider as ground seeds in each cell
        local_radius: Radius for local surface analysis
        slope_threshold: Maximum allowed slope in degrees
    
    Returns:
        ground_pcd: Extracted ground points
        non_ground_pcd: Non-ground points
    """
    points = np.asarray(pcd.points)
    
    # Step 1: 2D grid-based segmentation (ignore Z for grid)
    print("Step 1: 2D grid segmentation...")
    xy_points = points[:, :2]
    min_xy = xy_points.min(axis=0)
    max_xy = xy_points.max(axis=0)
    
    # Create 2D grid indices
    grid_indices = ((xy_points - min_xy) / voxel_size).astype(int)
    
    # Find lowest points in each 2D cell
    initial_ground_mask = np.zeros(len(points), dtype=bool)
    
    # Group by 2D grid cell
    unique_cells = np.unique(grid_indices, axis=0)
    
    for cell in tqdm(unique_cells, desc="Processing grid cells"):
        # Find all points in this 2D cell
        cell_mask = (grid_indices[:, 0] == cell[0]) & (grid_indices[:, 1] == cell[1])
        cell_indices = np.where(cell_mask)[0]
        
        if len(cell_indices) > 0:
            cell_heights = points[cell_indices, 2]
            
            # Use percentile to be robust to outliers
            height_threshold = np.percentile(cell_heights, height_percentile)
            
            # Mark points below threshold as potential ground
            ground_in_cell = cell_indices[cell_heights <= height_threshold + 0.3]  # 30cm tolerance
            initial_ground_mask[ground_in_cell] = True
    
    ground_indices = np.where(initial_ground_mask)[0]
    print(f"Initial segmentation: {len(ground_indices)} ground points")
    
    # Step 2: Region growing with strict height constraints
    print("Step 2: Constrained region growing...")
    ground_mask = initial_ground_mask.copy()
    
    # Build KD-tree for 2D search (project to XY plane)
    pcd_2d = o3d.geometry.PointCloud()
    pcd_2d.points = o3d.utility.Vector3dVector(np.column_stack([xy_points, np.zeros(len(points))]))
    pcd_tree_2d = o3d.geometry.KDTreeFlann(pcd_2d)
    
    # Single pass region growing
    new_ground_indices = []
    
    for idx in tqdm(ground_indices, desc="Region growing"):
        # Search in 2D (XY plane)
        [k, nn_indices, _] = pcd_tree_2d.search_radius_vector_3d(
            pcd_2d.points[idx], local_radius
        )
        
        if k > 5:
            # Get heights of neighbors
            neighbor_heights = points[nn_indices, 2]
            current_height = points[idx, 2]
            
            # Compute local statistics
            local_median = np.median(neighbor_heights[ground_mask[nn_indices]])
            
            # Add neighbors that are close to local ground height
            for nn_idx in nn_indices:
                if not ground_mask[nn_idx]:
                    height_diff = abs(points[nn_idx, 2] - local_median)
                    if height_diff < 0.5:  # Within 50cm of local ground
                        new_ground_indices.append(nn_idx)
    
    # Update ground mask
    if new_ground_indices:
        ground_mask[new_ground_indices] = True
    
    print(f"Added {len(new_ground_indices)} points through region growing")
    
    # Step 3: Remove isolated points and fill small gaps
    print("Step 3: Morphological refinement...")
    
    # Create 2D binary image for morphological operations
    grid_resolution = 1.0  # 1m resolution
    grid_size = ((max_xy - min_xy) / grid_resolution).astype(int) + 1
    
    # Project to 2D grid
    ground_image = np.zeros(grid_size, dtype=bool)
    ground_points_2d = xy_points[ground_mask]
    grid_coords = ((ground_points_2d - min_xy) / grid_resolution).astype(int)
    
    for coord in grid_coords:
        if 0 <= coord[0] < grid_size[0] and 0 <= coord[1] < grid_size[1]:
            ground_image[coord[0], coord[1]] = True
    
    # Morphological operations
    from scipy import ndimage
    struct = ndimage.generate_binary_structure(2, 1)  # 4-connectivity
    
    # Remove small isolated regions
    ground_image = ndimage.binary_opening(ground_image, structure=struct, iterations=1)
    
    # Fill small holes
    ground_image = ndimage.binary_closing(ground_image, structure=struct, iterations=2)
    
    # Map back to 3D points
    final_ground_mask = np.zeros(len(points), dtype=bool)
    
    all_grid_coords = ((xy_points - min_xy) / grid_resolution).astype(int)
    
    for i, coord in enumerate(all_grid_coords):
        if (0 <= coord[0] < grid_size[0] and 
            0 <= coord[1] < grid_size[1] and 
            ground_image[coord[0], coord[1]]):
            
            # Additional height check
            nearby_ground = ground_indices[
                np.linalg.norm(xy_points[ground_indices] - xy_points[i], axis=1) < local_radius * 2
            ]
            
            if len(nearby_ground) > 0:
                expected_height = np.median(points[nearby_ground, 2])
                if abs(points[i, 2] - expected_height) < 1.0:  # Within 1m of expected height
                    final_ground_mask[i] = True
    
    # Create output point clouds
    ground_pcd = pcd.select_by_index(np.where(final_ground_mask)[0])
    non_ground_pcd = pcd.select_by_index(np.where(~final_ground_mask)[0])
    
    print(f"Fast hybrid filter: {final_ground_mask.sum()} / {len(points)} ground points ({100*final_ground_mask.sum()/len(points):.1f}%)")
    
    return ground_pcd, non_ground_pcd


def extract_ground_hybrid(pcd, voxel_size=2.0, height_percentile=10, local_radius=5.0, slope_threshold=30.0):
    """
    Hybrid method combining voxel-based and local surface fitting.
    Optimized for race tracks with smooth elevation changes.
    
    Args:
        pcd: Open3D point cloud object
        voxel_size: Size of voxels for initial segmentation
        height_percentile: Percentile of heights to consider as ground seeds in each voxel
        local_radius: Radius for local surface fitting
        slope_threshold: Maximum allowed slope in degrees
    
    Returns:
        ground_pcd: Extracted ground points
        non_ground_pcd: Non-ground points
    """
    points = np.asarray(pcd.points)
    
    # Step 1: Voxel-based initial segmentation
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)
    voxel_indices = ((points - min_bound) / voxel_size).astype(int)
    
    # Find unique voxels
    unique_voxels, inverse_indices = np.unique(
        voxel_indices, axis=0, return_inverse=True
    )
    
    initial_ground_mask = np.zeros(len(points), dtype=bool)
    
    # Process each voxel to find potential ground seeds
    for i in tqdm(range(len(unique_voxels)), desc="Initial voxel segmentation"):
        voxel_mask = inverse_indices == i
        voxel_points = points[voxel_mask]
        
        if len(voxel_points) > 5:
            # Use percentile instead of minimum to be robust to outliers
            height_threshold = np.percentile(voxel_points[:, 2], height_percentile)
            
            voxel_indices_in_original = np.where(voxel_mask)[0]
            ground_in_voxel = voxel_points[:, 2] <= height_threshold
            
            initial_ground_mask[voxel_indices_in_original[ground_in_voxel]] = True
    
    print(f"Initial voxel segmentation: {initial_ground_mask.sum()} potential ground points")
    
    # Step 2: Region growing with local surface fitting
    ground_indices = np.where(initial_ground_mask)[0]
    
    # Build KD-tree for efficient neighbor search
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    
    # Iterative region growing
    max_iterations = 10
    for iteration in range(max_iterations):
        new_ground_indices = set(ground_indices)
        
        # For each ground point, check its neighbors
        for idx in tqdm(ground_indices, desc=f"Region growing iteration {iteration}", leave=False):
            [k, nn_indices, _] = pcd_tree.search_radius_vector_3d(
                pcd.points[idx], local_radius
            )
            
            if k < 10:  # Need enough neighbors for robust fitting
                continue
            
            # Fit local surface to neighborhood
            neighbor_points = points[nn_indices]
            
            # Use RANSAC for robust plane fitting
            local_pcd = o3d.geometry.PointCloud()
            local_pcd.points = o3d.utility.Vector3dVector(neighbor_points)
            
            try:
                plane_model, inliers = local_pcd.segment_plane(
                    distance_threshold=0.1,
                    ransac_n=3,
                    num_iterations=100
                )
                
                # Check plane slope
                normal = plane_model[:3]
                if normal[2] < 0:
                    normal = -normal
                
                angle = np.degrees(np.arccos(np.clip(normal[2], -1, 1)))
                
                if angle <= slope_threshold:
                    # Add inliers as ground points
                    ground_neighbors = np.array(nn_indices)[inliers]
                    new_ground_indices.update(ground_neighbors)
                    
            except:
                continue
        
        # Check convergence
        old_size = len(ground_indices)
        ground_indices = np.array(list(new_ground_indices))
        new_size = len(ground_indices)
        
        print(f"Iteration {iteration}: {new_size} ground points (+{new_size - old_size})")
        
        if new_size <= old_size * 1.01:  # Less than 1% growth
            break
    
    # Step 3: Fill holes using morphological operations
    # Create 2D grid projection
    xy_points = points[:, :2]
    grid_resolution = 0.5
    
    xy_min = xy_points.min(axis=0)
    xy_max = xy_points.max(axis=0)
    
    grid_size = ((xy_max - xy_min) / grid_resolution).astype(int) + 1
    
    # Create binary image
    ground_image = np.zeros(grid_size, dtype=bool)
    ground_xy = points[ground_indices, :2]
    grid_coords = ((ground_xy - xy_min) / grid_resolution).astype(int)
    
    for coord in grid_coords:
        if 0 <= coord[0] < grid_size[0] and 0 <= coord[1] < grid_size[1]:
            ground_image[coord[0], coord[1]] = True
    
    # Morphological closing to fill small holes
    from scipy import ndimage
    struct = ndimage.generate_binary_structure(2, 2)
    closed_image = ndimage.binary_closing(ground_image, structure=struct, iterations=2)
    filled_image = ndimage.binary_fill_holes(closed_image)
    
    # Map back to 3D points
    final_ground_mask = np.zeros(len(points), dtype=bool)
    final_ground_mask[ground_indices] = True
    
    # Add points that fall within filled regions
    all_grid_coords = ((xy_points - xy_min) / grid_resolution).astype(int)
    
    for i, coord in tqdm(enumerate(all_grid_coords), total=len(all_grid_coords), desc="Filling holes"):
        if (0 <= coord[0] < grid_size[0] and 
            0 <= coord[1] < grid_size[1] and 
            filled_image[coord[0], coord[1]]):
            
            # Check if point is close to estimated ground height
            nearby_ground = ground_indices[
                np.linalg.norm(xy_points[ground_indices] - xy_points[i], axis=1) < local_radius
            ]
            
            if len(nearby_ground) > 0:
                nearby_heights = points[nearby_ground, 2]
                expected_height = np.median(nearby_heights)
                
                if abs(points[i, 2] - expected_height) < 0.5:  # Within 0.5m of expected height
                    final_ground_mask[i] = True
    
    # Create output point clouds
    ground_pcd = pcd.select_by_index(np.where(final_ground_mask)[0])
    non_ground_pcd = pcd.select_by_index(np.where(~final_ground_mask)[0])
    
    print(f"Hybrid filter: {final_ground_mask.sum()} / {len(points)} ground points ({100*final_ground_mask.sum()/len(points):.1f}%)")
    
    return ground_pcd, non_ground_pcd


def extract_ground_racetrack(pcd, cell_size=5.0, height_tolerance=0.5, min_neighbors=10):
    """
    Specialized method for race track ground extraction.
    Uses 2D grid cells and local height filtering.
    
    Args:
        pcd: Open3D point cloud object
        cell_size: Size of 2D grid cells in meters
        height_tolerance: Maximum height difference from lowest point in cell
        min_neighbors: Minimum neighbors for a point to be considered ground
    
    Returns:
        ground_pcd: Extracted ground points
        non_ground_pcd: Non-ground points
    """
    points = np.asarray(pcd.points)
    
    print("Race track ground extraction...")
    
    # Step 1: Create 2D grid and find lowest points
    xy_points = points[:, :2]
    min_xy = xy_points.min(axis=0)
    max_xy = xy_points.max(axis=0)
    
    # Create grid
    grid_indices = ((xy_points - min_xy) / cell_size).astype(int)
    unique_cells = np.unique(grid_indices, axis=0)
    
    ground_mask = np.zeros(len(points), dtype=bool)
    
    # Process each cell
    for cell in tqdm(unique_cells, desc="Processing cells"):
        cell_mask = (grid_indices[:, 0] == cell[0]) & (grid_indices[:, 1] == cell[1])
        cell_indices = np.where(cell_mask)[0]
        
        if len(cell_indices) >= min_neighbors:
            cell_heights = points[cell_indices, 2]
            
            # Find minimum height (actual ground level)
            min_height = np.min(cell_heights)
            
            # Mark points within tolerance of minimum
            ground_in_cell = cell_indices[cell_heights <= min_height + height_tolerance]
            ground_mask[ground_in_cell] = True
    
    print(f"Initial extraction: {ground_mask.sum()} ground points")
    
    # Step 2: Fill gaps between cells
    # Build KD-tree for neighbor search
    ground_points = points[ground_mask]
    if len(ground_points) > 0:
        ground_pcd_temp = o3d.geometry.PointCloud()
        ground_pcd_temp.points = o3d.utility.Vector3dVector(ground_points)
        ground_tree = o3d.geometry.KDTreeFlann(ground_pcd_temp)
        
        # Check non-ground points near ground points
        non_ground_indices = np.where(~ground_mask)[0]
        points_to_add = []
        
        for idx in tqdm(non_ground_indices[::10], desc="Filling gaps"):  # Sample every 10th point for speed
            point = points[idx]
            
            # Find nearest ground point
            [k, nn_indices, nn_distances] = ground_tree.search_knn_vector_3d(point, 1)
            
            if k > 0 and nn_distances[0] < cell_size:
                nearest_ground_height = ground_points[nn_indices[0], 2]
                height_diff = abs(point[2] - nearest_ground_height)
                
                if height_diff < height_tolerance:
                    points_to_add.append(idx)
        
        # Add gap-filling points
        if points_to_add:
            ground_mask[points_to_add] = True
            print(f"Added {len(points_to_add)} gap-filling points")
    
    # Create output point clouds
    ground_pcd = pcd.select_by_index(np.where(ground_mask)[0])
    non_ground_pcd = pcd.select_by_index(np.where(~ground_mask)[0])
    
    print(f"Race track filter: {ground_mask.sum()} / {len(points)} ground points ({100*ground_mask.sum()/len(points):.1f}%)")
    
    return ground_pcd, non_ground_pcd


def refine_ground_statistical(ground_pcd, nb_neighbors=30, std_ratio=2.0):
    """
    Refine ground extraction using statistical outlier removal.
    
    Args:
        ground_pcd: Initial ground point cloud
        nb_neighbors: Number of neighbors for statistics
        std_ratio: Standard deviation ratio threshold
    
    Returns:
        refined_pcd: Refined ground point cloud
    """
    print(f"Refining with statistical filter (neighbors={nb_neighbors}, std_ratio={std_ratio})")
    
    refined_pcd, _ = ground_pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio
    )
    
    print(f"Refined: {len(refined_pcd.points)} / {len(ground_pcd.points)} points kept")
    
    return refined_pcd


def visualize_results(original_pcd, ground_pcd, non_ground_pcd):
    """
    Visualize the ground extraction results.
    
    Args:
        original_pcd: Original point cloud
        ground_pcd: Extracted ground points (colored green)
        non_ground_pcd: Non-ground points (colored red)
    """
    # Color the point clouds
    ground_pcd.paint_uniform_color([0.0, 1.0, 0.0])  # Green for ground
    non_ground_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # Red for non-ground
    
    # Create visualization windows
    vis1 = o3d.visualization.Visualizer()
    vis1.create_window(window_name="Original Point Cloud", width=800, height=600, left=50, top=50)
    vis1.add_geometry(original_pcd)
    
    vis2 = o3d.visualization.Visualizer()
    vis2.create_window(window_name="Ground Extraction Result", width=800, height=600, left=900, top=50)
    vis2.add_geometry(ground_pcd)
    vis2.add_geometry(non_ground_pcd)
    
    # Run visualizers
    while True:
        vis1.poll_events()
        vis1.update_renderer()
        vis2.poll_events()
        vis2.update_renderer()
        
        if not vis1.poll_events() or not vis2.poll_events():
            break
    
    vis1.destroy_window()
    vis2.destroy_window()


def main():
    parser = argparse.ArgumentParser(description="Extract ground manifold from race track point cloud")
    parser.add_argument("input_pcd", type=str, help="Input PCD file path")
    parser.add_argument("-o", "--output", type=str, help="Output ground PCD file path")
    parser.add_argument("-m", "--method", type=str, default="ransac",
                        choices=["ransac", "progressive", "voxel", "hybrid", "hybrid-fast", "racetrack"],
                        help="Ground extraction method")
    
    # RANSAC parameters
    parser.add_argument("--distance-threshold", type=float, default=0.1,
                        help="RANSAC distance threshold (m)")
    parser.add_argument("--ransac-iterations", type=int, default=1000,
                        help="Number of RANSAC iterations")
    
    # Progressive filter parameters
    parser.add_argument("--angle-threshold", type=float, default=15.0,
                        help="Maximum ground angle (degrees)")
    parser.add_argument("--grid-size", type=float, default=10.0,
                        help="Grid cell size for local plane fitting (m)")
    parser.add_argument("--growth-factor", type=float, default=1.1,
                        help="Growth factor for distance threshold")
    
    # Voxel filter parameters
    parser.add_argument("--voxel-size", type=float, default=0.5,
                        help="Voxel size for voxel-based method (m)")
    parser.add_argument("--height-threshold", type=float, default=0.3,
                        help="Height threshold for voxel method (m)")
    
    # Hybrid method parameters
    parser.add_argument("--local-radius", type=float, default=5.0,
                        help="Local radius for surface fitting in hybrid method (m)")
    parser.add_argument("--slope-threshold", type=float, default=30.0,
                        help="Maximum slope for ground surfaces in hybrid method (degrees)")
    parser.add_argument("--height-percentile", type=float, default=10,
                        help="Height percentile for initial seeds in hybrid method")
    
    # Race track method parameters
    parser.add_argument("--cell-size", type=float, default=5.0,
                        help="Cell size for race track method (m)")
    parser.add_argument("--height-tolerance", type=float, default=0.5,
                        help="Height tolerance from minimum in each cell (m)")
    parser.add_argument("--min-neighbors", type=int, default=10,
                        help="Minimum points per cell for race track method")
    
    # Refinement parameters
    parser.add_argument("--refine", action="store_true",
                        help="Apply statistical outlier removal refinement")
    parser.add_argument("--std-ratio", type=float, default=2.0,
                        help="Standard deviation ratio for outlier removal")
    
    # Visualization
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize the results")
    parser.add_argument("--no-output", action="store_true",
                        help="Don't save output, only visualize")
    
    args = parser.parse_args()
    
    # Load point cloud
    print(f"Loading point cloud from: {args.input_pcd}")
    pcd = o3d.io.read_point_cloud(args.input_pcd)
    print(f"Loaded {len(pcd.points)} points")
    
    # Extract ground based on selected method
    start_time = time.time()
    
    if args.method == "ransac":
        ground_pcd, non_ground_pcd, _ = extract_ground_ransac(
            pcd,
            distance_threshold=args.distance_threshold,
            num_iterations=args.ransac_iterations
        )
    elif args.method == "progressive":
        ground_pcd, non_ground_pcd = extract_ground_progressive(
            pcd,
            angle_threshold=args.angle_threshold
        )
    elif args.method == "voxel":
        ground_pcd, non_ground_pcd = extract_ground_voxel(
            pcd,
            voxel_size=args.voxel_size,
            height_threshold=args.height_threshold
        )
    elif args.method == "hybrid":
        ground_pcd, non_ground_pcd = extract_ground_hybrid(
            pcd,
            voxel_size=args.voxel_size,
            height_percentile=args.height_percentile,
            local_radius=args.local_radius,
            slope_threshold=args.slope_threshold
        )
    elif args.method == "hybrid-fast":
        ground_pcd, non_ground_pcd = extract_ground_hybrid_fast(
            pcd,
            voxel_size=args.voxel_size,
            height_percentile=args.height_percentile,
            local_radius=args.local_radius,
            slope_threshold=args.slope_threshold
        )
    elif args.method == "racetrack":
        ground_pcd, non_ground_pcd = extract_ground_racetrack(
            pcd,
            cell_size=args.cell_size,
            height_tolerance=args.height_tolerance,
            min_neighbors=args.min_neighbors
        )
    
    # Apply refinement if requested
    if args.refine:
        ground_pcd = refine_ground_statistical(ground_pcd, std_ratio=args.std_ratio)
    
    elapsed_time = time.time() - start_time
    print(f"Ground extraction completed in {elapsed_time:.2f} seconds")
    
    # Save output if requested
    if not args.no_output:
        output_path = args.output
        if output_path is None:
            input_path = Path(args.input_pcd)
            output_path = input_path.parent / f"{input_path.stem}_ground.pcd"
        
        print(f"Saving ground points to: {output_path}")
        o3d.io.write_point_cloud(str(output_path), ground_pcd)
        
        # Also save non-ground points
        non_ground_path = Path(output_path).parent / f"{Path(output_path).stem.replace('_ground', '')}_non_ground.pcd"
        print(f"Saving non-ground points to: {non_ground_path}")
        o3d.io.write_point_cloud(str(non_ground_path), non_ground_pcd)
    
    # Visualize if requested
    if args.visualize:
        print("Visualizing results...")
        visualize_results(pcd, ground_pcd, non_ground_pcd)


if __name__ == "__main__":
    main()