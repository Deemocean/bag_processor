#!/usr/bin/env python3
"""
Filter PCD points based on proximity to odometry trajectory from ROS2 bag.
Extracts points within a specified distance from the trajectory path.
"""

import argparse
import numpy as np
import open3d as o3d
from pathlib import Path
import sys
from tqdm import tqdm

# Add parent directory to path to import data_processor modules
sys.path.append(str(Path(__file__).parent.parent))

from data_processor.data_loader.data_loader import DataLoader


def extract_trajectory_points(bag_path, odom_topic='/odom'):
    """
    Extract trajectory points from odometry messages in bag file.
    
    Args:
        bag_path: Path to ROS2 bag file
        odom_topic: Odometry topic name
    
    Returns:
        trajectory_points: Nx3 numpy array of trajectory positions
    """
    print(f"Loading odometry data from bag: {bag_path}")
    print(f"Looking for topic: {odom_topic}")
    
    # Use DataLoader to load the bag data with specific topic
    # Create output directory for CSV files
    output_dir = f"./csv_output_{Path(bag_path).stem}"
    loader = DataLoader(bag_path=bag_path, topics=[odom_topic], output_dir=output_dir)
    database = loader.load_all()
    
    # Check if topic was loaded
    available_topics = database.topics()
    print(f"Loaded topics: {available_topics}")
    
    if not available_topics:
        # Try loading without topic filter to see what's available
        print("Topic not found. Checking all available topics in bag...")
        # For checking available topics, we need to use BagConverter directly
        from data_processor.bag_converter.bag_converter import BagConverter
        converter = BagConverter(bag_path)
        all_topics = converter.get_topic_list()
        print(f"Available topics in bag: {all_topics}")
        raise ValueError(f"Topic '{odom_topic}' not found. Available topics: {all_topics}")
    
    # Get the odometry dataframe
    odom_df = database[available_topics[0]]
    print(f"Odometry data shape: {odom_df.shape}")
    print(f"Available columns: {odom_df.columns.tolist()}")
    
    # Extract position data - try different column naming conventions
    if 'x' in odom_df.columns and 'y' in odom_df.columns and 'z' in odom_df.columns:
        trajectory_points = odom_df[['x', 'y', 'z']].values
    elif 'position_x' in odom_df.columns:
        trajectory_points = odom_df[['position_x', 'position_y', 'position_z']].values
    elif 'pose.pose.position.x' in odom_df.columns:
        trajectory_points = odom_df[['pose.pose.position.x', 'pose.pose.position.y', 'pose.pose.position.z']].values
    else:
        # Try to find any columns containing position information
        pos_cols = [col for col in odom_df.columns if 'position' in col.lower() or (col in ['x', 'y', 'z'])]
        if pos_cols:
            print(f"Found position-related columns: {pos_cols}")
        raise ValueError(f"Could not find position columns in odometry data. Available columns: {odom_df.columns.tolist()}")
    
    print(f"Extracted {len(trajectory_points)} trajectory points")
    
    # Remove any NaN values
    valid_mask = ~np.isnan(trajectory_points).any(axis=1)
    trajectory_points = trajectory_points[valid_mask]
    
    print(f"Valid trajectory points: {len(trajectory_points)}")
    
    # Print trajectory bounds for debugging
    if len(trajectory_points) > 0:
        min_bounds = trajectory_points.min(axis=0)
        max_bounds = trajectory_points.max(axis=0)
        print(f"Trajectory bounds: min={min_bounds}, max={max_bounds}")
    
    return trajectory_points


def filter_points_near_trajectory(pcd, trajectory_points, distance_threshold=10.0):
    """
    Filter point cloud to keep only points within distance threshold of trajectory.
    
    Args:
        pcd: Open3D point cloud object
        trajectory_points: Nx3 array of trajectory positions
        distance_threshold: Maximum distance from trajectory (meters)
    
    Returns:
        filtered_pcd: Point cloud containing only points near trajectory
        mask: Boolean mask indicating which points were kept
    """
    points = np.asarray(pcd.points)
    
    print(f"Filtering {len(points)} points based on trajectory proximity...")
    
    # Create trajectory point cloud for KD-tree
    traj_pcd = o3d.geometry.PointCloud()
    traj_pcd.points = o3d.utility.Vector3dVector(trajectory_points)
    
    # Build KD-tree for trajectory
    traj_tree = o3d.geometry.KDTreeFlann(traj_pcd)
    
    # Find points within threshold distance
    mask = np.zeros(len(points), dtype=bool)
    
    # Process in batches for efficiency
    batch_size = 1000
    for i in tqdm(range(0, len(points), batch_size), desc="Filtering points"):
        batch_end = min(i + batch_size, len(points))
        batch_points = points[i:batch_end]
        
        for j, point in enumerate(batch_points):
            # Find nearest trajectory point
            [k, idx, dist] = traj_tree.search_knn_vector_3d(point, 1)
            
            if k > 0 and np.sqrt(dist[0]) <= distance_threshold:
                mask[i + j] = True
    
    # Create filtered point cloud
    filtered_pcd = pcd.select_by_index(np.where(mask)[0])
    
    print(f"Kept {mask.sum()} / {len(points)} points ({100*mask.sum()/len(points):.1f}%)")
    
    return filtered_pcd, mask


def visualize_filtered_result(original_pcd, filtered_pcd, trajectory_points):
    """
    Visualize the filtering results with trajectory.
    
    Args:
        original_pcd: Original point cloud
        filtered_pcd: Filtered point cloud
        trajectory_points: Trajectory points
    """
    # Color the point clouds
    filtered_pcd.paint_uniform_color([0.0, 1.0, 0.0])  # Green for filtered points
    
    # Create trajectory visualization
    traj_pcd = o3d.geometry.PointCloud()
    traj_pcd.points = o3d.utility.Vector3dVector(trajectory_points)
    traj_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # Red for trajectory
    
    # Create line set for trajectory
    lines = []
    for i in range(len(trajectory_points) - 1):
        lines.append([i, i + 1])
    
    traj_lines = o3d.geometry.LineSet()
    traj_lines.points = o3d.utility.Vector3dVector(trajectory_points)
    traj_lines.lines = o3d.utility.Vector2iVector(lines)
    traj_lines.paint_uniform_color([1.0, 0.0, 0.0])  # Red lines
    
    # Visualize
    print("Visualizing results...")
    print("Green: Filtered points near trajectory")
    print("Red: Trajectory path")
    
    o3d.visualization.draw_geometries(
        [filtered_pcd, traj_pcd, traj_lines],
        window_name="PCD Filtered by Trajectory",
        width=1200,
        height=800
    )


def compute_trajectory_bounds(trajectory_points, distance_threshold):
    """
    Compute bounding box of trajectory with threshold padding.
    
    Args:
        trajectory_points: Nx3 array of trajectory positions
        distance_threshold: Distance to pad the bounds
    
    Returns:
        min_bound: Minimum bounds
        max_bound: Maximum bounds
    """
    min_bound = trajectory_points.min(axis=0) - distance_threshold
    max_bound = trajectory_points.max(axis=0) + distance_threshold
    
    return min_bound, max_bound


def main():
    parser = argparse.ArgumentParser(
        description="Filter PCD points based on proximity to odometry trajectory from ROS2 bag"
    )
    parser.add_argument("bag_path", type=str, help="Path to ROS2 bag file")
    parser.add_argument("pcd_path", type=str, help="Path to input PCD file")
    parser.add_argument("-o", "--output", type=str, help="Output PCD file path")
    parser.add_argument("-d", "--distance", type=float, default=10.0,
                        help="Distance threshold from trajectory (meters)")
    parser.add_argument("-t", "--topic", type=str, default="/odom",
                        help="Odometry topic name")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize the results")
    parser.add_argument("--downsample", type=float,
                        help="Downsample trajectory points (voxel size in meters)")
    parser.add_argument("--pre-filter", action="store_true",
                        help="Pre-filter by bounding box for efficiency")
    
    args = parser.parse_args()
    
    # Extract trajectory from bag
    try:
        trajectory_points = extract_trajectory_points(args.bag_path, args.topic)
    except Exception as e:
        print(f"Error loading trajectory: {e}")
        return
    
    # Optionally downsample trajectory for efficiency
    if args.downsample:
        print(f"Downsampling trajectory with voxel size {args.downsample}m...")
        traj_pcd = o3d.geometry.PointCloud()
        traj_pcd.points = o3d.utility.Vector3dVector(trajectory_points)
        traj_pcd = traj_pcd.voxel_down_sample(voxel_size=args.downsample)
        trajectory_points = np.asarray(traj_pcd.points)
        print(f"Downsampled to {len(trajectory_points)} trajectory points")
    
    # Load point cloud
    print(f"Loading point cloud from: {args.pcd_path}")
    pcd = o3d.io.read_point_cloud(args.pcd_path)
    print(f"Loaded {len(pcd.points)} points")
    
    # Pre-filter by bounding box if requested
    if args.pre_filter:
        print("Pre-filtering by trajectory bounding box...")
        min_bound, max_bound = compute_trajectory_bounds(trajectory_points, args.distance)
        
        # Create bounding box
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        pcd = pcd.crop(bbox)
        print(f"Pre-filtered to {len(pcd.points)} points within bounding box")
    
    # Filter points near trajectory
    filtered_pcd, mask = filter_points_near_trajectory(
        pcd, trajectory_points, args.distance
    )
    
    # Compute statistics
    if len(filtered_pcd.points) > 0:
        distances_to_origin = np.linalg.norm(np.asarray(filtered_pcd.points), axis=1)
        print(f"\nStatistics:")
        print(f"  Points kept: {len(filtered_pcd.points)}")
        print(f"  Min distance to origin: {distances_to_origin.min():.2f}m")
        print(f"  Max distance to origin: {distances_to_origin.max():.2f}m")
        print(f"  Mean distance to origin: {distances_to_origin.mean():.2f}m")
    
    # Save output
    if args.output:
        output_path = args.output
    else:
        input_path = Path(args.pcd_path)
        output_path = input_path.parent / f"{input_path.stem}_filtered_d{int(args.distance)}m.pcd"
    
    print(f"Saving filtered point cloud to: {output_path}")
    o3d.io.write_point_cloud(str(output_path), filtered_pcd)
    
    # Save trajectory as well
    traj_output = Path(output_path).parent / f"{Path(output_path).stem}_trajectory.pcd"
    traj_pcd = o3d.geometry.PointCloud()
    traj_pcd.points = o3d.utility.Vector3dVector(trajectory_points)
    o3d.io.write_point_cloud(str(traj_output), traj_pcd)
    print(f"Saved trajectory to: {traj_output}")
    
    # Visualize if requested
    if args.visualize:
        visualize_filtered_result(pcd, filtered_pcd, trajectory_points)


if __name__ == "__main__":
    main()