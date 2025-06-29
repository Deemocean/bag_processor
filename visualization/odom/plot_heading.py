#!/usr/bin/env python3

import os
import sys
# add project root to PYTHONPATH
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
)

import numpy as np
import matplotlib.pyplot as plt

from data_processor.data_loader.data_loader import DataLoader
from data_processor.bag_converter.bag_converter import BagConfig
from data_processor.helpers.helpers import select_time_range


def get_topics_by_nickname(db, nickname):
    """Get all topics for a given nickname."""
    topics = {}
    for topic in db.topics():
        if topic.endswith(f"_{nickname}"):
            base_name = topic.rsplit(f"_{nickname}", 1)[0]
            topics[base_name] = topic
    return topics


def quaternion_to_heading_deg(qx, qy, qz, qw):
    """Convert quaternion to heading in degrees (yaw angle projected on XY plane)."""
    # Calculate yaw (rotation around Z-axis) from quaternion
    yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    # Convert from radians to degrees
    heading_deg = np.degrees(yaw)
    return heading_deg


def unwrap_heading(heading):
    """Unwrap heading to avoid discontinuities at ±180 degrees."""
    return np.unwrap(np.radians(heading)) * 180 / np.pi


def calculate_heading_rate(timestamps_ns, heading_deg):
    """Calculate heading rate from timestamps and heading data."""
    # Find unique timestamps to avoid division by zero
    unique_times, unique_indices = np.unique(timestamps_ns, return_index=True)
    
    if len(unique_times) < 2:
        return np.array([]), np.array([])
    
    # Sort indices to maintain temporal order
    unique_indices = np.sort(unique_indices)
    time_ns_unique = unique_times
    heading_unique = heading_deg[unique_indices]
    
    # Calculate differences in nanoseconds
    dt_ns = np.diff(time_ns_unique)
    dheading = np.diff(heading_unique)
    
    # Filter out zero time intervals
    valid_mask = dt_ns > 0
    if not np.any(valid_mask):
        return np.array([]), np.array([])
    
    # Convert to seconds and calculate rate
    dt_sec = dt_ns[valid_mask] / 1e9
    dheading_valid = dheading[valid_mask]
    heading_rate = dheading_valid / dt_sec
    
    # Time points for plotting (convert to seconds from start)
    time_plot = (time_ns_unique[1:][valid_mask] - time_ns_unique[0]) / 1e9
    
    return time_plot, heading_rate


def main():
    csv_dir = "/tmp/CSVs"
    
    # Configure two bags with odometry data
    bag_configs = [
        BagConfig(
            bag_path="/media/dm0/Matrix1/recordings/opt_s_gdop_05/2025-06-29_15-09-19/2025-06-29_15-09-19_0.mcap",
            topics={
                "/state/odom_raw": "odom_raw",
                "/state/odom": "odom",
            },
            nickname="opt_s_gdop05_avg"
        ),

        BagConfig(
            bag_path="/media/dm0/Matrix1/recordings/stable_new/2025-06-29_15-01-46/2025-06-29_15-01-46_0.mcap",
            topics={
                "/state/odom_raw": "odom_raw",
                "/state/odom": "odom",
            },
            nickname="stable"
        )
    ]
    
    # Load data
    loader = DataLoader(bag_configs=bag_configs, output_dir=csv_dir)
    db = loader.load_all()
    print("Loaded topics:", db.topics())

    # Get nicknames from bag configs
    nicknames = [config.nickname for config in bag_configs]
    duration = 300 * 1e9  # 5 minutes in nanoseconds
    
    # Process data for each bag
    run_data = {}
    for nickname in nicknames:
        topics = get_topics_by_nickname(db, nickname)
        if not topics:
            print(f"No topics found for nickname: {nickname}")
            continue
            
        # Get time range for this run
        odom_topic = topics.get("odom")
        if not odom_topic:
            print(f"No odometry topic found for {nickname}")
            continue
            
        start_time = db[odom_topic]["header_t"].min()
        end_time = start_time + duration
        
        print(f"\nProcessing data for {nickname}")
        print(f"Topics for {nickname}: {topics}")
        
        # Load and filter data
        run_data[nickname] = {}
        
        for data_type in ["odom", "odom_raw"]:
            if data_type in topics:
                df = select_time_range(db[topics[data_type]], start_time, end_time)
                run_data[nickname][data_type] = df
                print(f"  {data_type} data shape: {df.shape}")
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Colors for different runs and data types
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, (nickname, data) in enumerate(run_data.items()):
        color = colors[i % len(colors)]
        
        for j, (data_type, df) in enumerate(data.items()):
            if len(df) == 0:
                continue
                
            # Check for quaternion columns
            quat_cols = ['qx', 'qy', 'qz', 'qw']
            if not all(col in df.columns for col in quat_cols):
                print(f"  {nickname} {data_type}: Missing quaternion columns")
                continue
            
            # Convert quaternion to heading
            heading = quaternion_to_heading_deg(
                df['qx'].values, df['qy'].values, 
                df['qz'].values, df['qw'].values
            )
            
            # Unwrap heading to avoid discontinuities
            heading_unwrapped = unwrap_heading(heading)
            
            # Time for heading plot
            time_sec = (df['header_t'] - df['header_t'].iloc[0]) / 1e9
            
            # Plot heading
            linestyle = '-' if data_type == 'odom' else '--'
            alpha = 1.0 if data_type == 'odom' else 0.8
            linewidth = 2 if data_type == 'odom' else 1.5
            
            ax1.plot(time_sec, heading_unwrapped, 
                    label=f'{nickname} ({data_type.replace("_", " ").title()})', 
                    color=color, linewidth=linewidth, linestyle=linestyle, alpha=alpha)
            
            # Calculate and plot heading rate
            time_rate, heading_rate = calculate_heading_rate(df['header_t'].values, heading_unwrapped)
            
            if len(heading_rate) > 0:
                print(f"  {nickname} {data_type}: {len(heading_rate)} rate points")
                print(f"    Rate range: [{heading_rate.min():.1f}, {heading_rate.max():.1f}] deg/s")
                
                # Clip extreme values for visualization
                p1, p99 = np.percentile(heading_rate, [1, 99])
                clip_range = max(abs(p1), abs(p99), 30)
                heading_rate_clipped = np.clip(heading_rate, -clip_range, clip_range)
                
                ax2.plot(time_rate, heading_rate_clipped, 
                        label=f'{nickname} ({data_type.replace("_", " ").title()})', 
                        color=color, linewidth=linewidth, linestyle=linestyle, alpha=alpha)
            else:
                print(f"  {nickname} {data_type}: No valid heading rate data")
    
    # Configure heading plot
    ax1.set_ylabel('Heading (deg)', fontsize=12)
    ax1.set_title('Heading Over Time', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Configure heading rate plot
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Heading Rate (deg/s)', fontsize=12)
    ax2.set_title('dHeading/dt', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add zero line and reference lines to rate plot
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.8)
    ax2.axhline(y=30, color='r', linestyle=':', alpha=0.5, linewidth=0.8)
    ax2.axhline(y=-30, color='r', linestyle=':', alpha=0.5, linewidth=0.8)
    ax2.axhline(y=60, color='r', linestyle='--', alpha=0.5, linewidth=0.8)
    ax2.axhline(y=-60, color='r', linestyle='--', alpha=0.5, linewidth=0.8)
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)  # Make room for legend
    plt.show()


if __name__ == "__main__":
    main()