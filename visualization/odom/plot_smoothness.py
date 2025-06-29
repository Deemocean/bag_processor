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
from scipy.interpolate import UnivariateSpline

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


def calculate_trajectory_jerk(x, y, time):
    """
    Calculate trajectory smoothness using jerk analysis (standard method).
    Jerk is the third derivative of position with respect to time.
    """
    # Ensure time is in seconds and monotonic
    dt = np.diff(time)
    if np.any(dt <= 0):
        print("Warning: Non-monotonic timestamps detected, using uniform spacing")
        dt = np.full(len(time)-1, np.mean(dt[dt > 0]))
        time = np.cumsum(np.concatenate([[0], dt]))
    
    # Calculate velocity (first derivative)
    vx = np.gradient(x, time)
    vy = np.gradient(y, time)
    
    # Calculate acceleration (second derivative)
    ax = np.gradient(vx, time)
    ay = np.gradient(vy, time)
    
    # Calculate jerk (third derivative)
    jerk_x = np.gradient(ax, time)
    jerk_y = np.gradient(ay, time)
    
    # Jerk magnitude
    jerk_magnitude = np.sqrt(jerk_x**2 + jerk_y**2)
    
    # Speed for reference
    speed = np.sqrt(vx**2 + vy**2)
    
    # Acceleration magnitude
    acceleration_magnitude = np.sqrt(ax**2 + ay**2)
    
    # Path curvature (geometric property)
    # κ = |v × a| / |v|³ where × is cross product in 2D: vx*ay - vy*ax
    cross_product = vx * ay - vy * ax
    speed_cubed = speed**3
    curvature = np.abs(cross_product) / np.where(speed_cubed > 1e-6, speed_cubed, 1e-6)
    curvature = np.nan_to_num(curvature)
    
    return {
        'jerk_x': jerk_x,
        'jerk_y': jerk_y,
        'jerk_magnitude': jerk_magnitude,
        'acceleration_magnitude': acceleration_magnitude,
        'speed': speed,
        'curvature': curvature,
        'velocity_x': vx,
        'velocity_y': vy,
        'acceleration_x': ax,
        'acceleration_y': ay
    }


def main():
    csv_dir = "/tmp/CSVs"
    
    # Configure bags with odometry data using same pattern as plot_alt.py
    bag_configs = [
        # BagConfig(
        #     bag_path="/media/dm0/Matrix1/recordings/opt_s_gdop_2_avg/2025-06-29_14-55-22/2025-06-29_14-55-22_0.mcap",
        #     topics={
        #         "/state/odom_raw": "odom_raw",
        #         "/state/odom": "odom",
        #         "/sensors/manager/gps0": "gps_a",
        #         "/sensors/manager/gps1": "gps_b",
        #         "/localization/debug/gps_dist": "gps_innovation"
        #     },
        #     nickname="opt_s_gdop2_avg"
        # ),

        BagConfig(
            bag_path="/media/dm0/Matrix1/recordings/opt_s_gdop_05/2025-06-29_15-09-19/2025-06-29_15-09-19_0.mcap",
            topics={
                "/state/odom_raw": "odom_raw",
                "/state/odom": "odom",
                "/sensors/manager/gps0": "gps_a",
                "/sensors/manager/gps1": "gps_b",
                "/localization/debug/gps_dist": "gps_innovation"
            },
            nickname="opt_s_gdop05_avg"
        ),
        BagConfig(
            bag_path="/media/dm0/Matrix1/recordings/stable_new/2025-06-29_15-01-46/2025-06-29_15-01-46_0.mcap",
            topics={
                "/state/odom_raw": "odom_raw",
                "/state/odom": "odom",
                "/sensors/manager/gps0": "gps_a",
                "/sensors/manager/gps1": "gps_b",
                "/localization/debug/gps_dist": "gps_innovation"
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
    duration = 300  # 5 minutes duration
    
    # Process data for each bag
    run_data = {}
    jerk_results = {}
    
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
        
        print(f"Processing trajectory jerk analysis for {nickname}")
        
        # Load and filter odometry data
        run_data[nickname] = {}
        
        if "odom" in topics:
            odom_data = select_time_range(db[topics["odom"]], start_time, end_time)
            run_data[nickname]["odom"] = odom_data
            
            # Extract XY trajectory
            if 'x' in odom_data.columns and 'y' in odom_data.columns:
                x = odom_data['x'].values
                y = odom_data['y'].values
                time = odom_data['header_t'].values  # Already in seconds (Unix time)
                
                # Filter out invalid data points
                valid_mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(time)
                x = x[valid_mask]
                y = y[valid_mask]
                time = time[valid_mask]
                
                if len(x) > 10:  # Need sufficient points for derivative calculation
                    # Calculate jerk and smoothness metrics
                    jerk_analysis = calculate_trajectory_jerk(x, y, time)
                    
                    jerk_results[nickname] = {
                        'x': x,
                        'y': y,
                        'time': time,
                        **jerk_analysis
                    }
                    
                    print(f"  {nickname}: Processed {len(x)} trajectory points")
                    print(f"  RMS jerk: {np.sqrt(np.mean(jerk_analysis['jerk_magnitude']**2)):.3f} m/s³")
                    print(f"  Mean speed: {np.mean(jerk_analysis['speed']):.3f} m/s")
                    print(f"  Max acceleration: {np.max(jerk_analysis['acceleration_magnitude']):.3f} m/s²")
                else:
                    print(f"  {nickname}: Insufficient data points for jerk analysis")
            else:
                print(f"  {nickname}: No x,y columns found in odometry data")

    # Create plots - XY trajectory on left, jerk analysis on right
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    
    # Left: Trajectory XY plot (spans both rows)
    ax_traj = fig.add_subplot(gs[:, 0])
    
    # Right: Jerk analysis time series plots
    ax_jerk = fig.add_subplot(gs[0, 1])
    ax_accel = fig.add_subplot(gs[1, 1])
    
    # Colors for different runs
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Find global time reference (earliest start time across all datasets)
    global_start_time = min(data['time'][0] for data in jerk_results.values())
    
    # Store time-indexed trajectory data for synchronized zooming
    time_to_traj_data = {}
    
    for i, (nickname, data) in enumerate(jerk_results.items()):
        color = colors[i % len(colors)]
        
        # Plot XY trajectory with point size based on linear normalized jerk
        # Scale jerk magnitude to reasonable point sizes (10-100)
        jerk_min = np.min(data['jerk_magnitude'])
        jerk_max = np.max(data['jerk_magnitude'])
        jerk_scaled = 10 + 90 * (data['jerk_magnitude'] - jerk_min) / (jerk_max - jerk_min + 1e-6)
        
        ax_traj.scatter(data['x'], data['y'], s=jerk_scaled, 
                       color=color, alpha=0.6, label=f'{nickname}')
        
        # Add trajectory line for context
        ax_traj.plot(data['x'], data['y'], color=color, linewidth=1, alpha=0.3)
        
        # Convert time to seconds from global start for time series plots
        time_sec = data['time'] - global_start_time
        
        # Store time-indexed trajectory data for sync
        time_to_traj_data[nickname] = {
            'time_sec': time_sec,
            'x': data['x'],
            'y': data['y'],
            'jerk_magnitude': data['jerk_magnitude'],
            'jerk_scaled': jerk_scaled,
            'acceleration_magnitude': data['acceleration_magnitude'],
            'color': color
        }
        
        # Plot jerk magnitude over time
        ax_jerk.plot(time_sec, data['jerk_magnitude'], label=f'{nickname}', 
                    color=color, linewidth=2, alpha=0.8)
        
        # Plot acceleration magnitude over time
        ax_accel.plot(time_sec, data['acceleration_magnitude'], label=f'{nickname}', 
                     color=color, linewidth=2, alpha=0.8)
    
    # Configure trajectory plot
    ax_traj.set_xlabel('X (m)', fontsize=12)
    ax_traj.set_ylabel('Y (m)', fontsize=12)
    ax_traj.set_title('Trajectory (Point Size ∝ Jerk)', fontsize=14, fontweight='bold')
    ax_traj.grid(True, alpha=0.3)
    ax_traj.legend()
    ax_traj.axis('equal')
    
    # Configure jerk plot
    ax_jerk.set_xlabel('Time (s)', fontsize=12)
    ax_jerk.set_ylabel('Jerk Magnitude (m/s³)', fontsize=12)
    ax_jerk.set_title('Trajectory Jerk', fontsize=14, fontweight='bold')
    ax_jerk.grid(True, alpha=0.3)
    ax_jerk.legend()
    ax_jerk.set_yscale('log')  # Log scale for jerk magnitude
    
    # Configure acceleration plot
    ax_accel.set_xlabel('Time (s)', fontsize=12)
    ax_accel.set_ylabel('Acceleration Magnitude (m/s²)', fontsize=12)
    ax_accel.set_title('Trajectory Acceleration', fontsize=14, fontweight='bold')
    ax_accel.grid(True, alpha=0.3)
    ax_accel.legend()
    
    # Share x-axis between time series plots
    ax_accel.sharex(ax_jerk)
    
    # Function to update trajectory plot based on time range
    def update_trajectory_view(time_min, time_max):
        # Clear trajectory plot
        ax_traj.clear()
        
        # Re-plot trajectory with time filtering
        for nickname, traj_data in time_to_traj_data.items():
            # Find indices within time range
            time_mask = (traj_data['time_sec'] >= time_min) & (traj_data['time_sec'] <= time_max)
            
            if np.any(time_mask):
                # Plot filtered trajectory with point size based on jerk
                ax_traj.scatter(traj_data['x'][time_mask], traj_data['y'][time_mask], 
                              s=traj_data['jerk_scaled'][time_mask], 
                              color=traj_data['color'], alpha=0.6, label=f'{nickname}')
                
                # Add trajectory line for context
                ax_traj.plot(traj_data['x'][time_mask], traj_data['y'][time_mask], 
                           color=traj_data['color'], linewidth=1, alpha=0.3)
        
        ax_traj.set_xlabel('X (m)', fontsize=12)
        ax_traj.set_ylabel('Y (m)', fontsize=12)
        ax_traj.set_title(f'Trajectory (Point Size ∝ Jerk) (Time: {time_min:.1f}s - {time_max:.1f}s)', fontsize=14, fontweight='bold')
        ax_traj.grid(True, alpha=0.3)
        ax_traj.legend()
        ax_traj.axis('equal')
        ax_traj.figure.canvas.draw_idle()
    
    # Connect zoom event handler
    def on_xlim_changed(ax):
        time_min, time_max = ax.get_xlim()
        update_trajectory_view(time_min, time_max)
    
    # Connect the callback to x-axis limit changes
    ax_jerk.callbacks.connect('xlim_changed', on_xlim_changed)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\n=== Trajectory Smoothness Summary (Jerk Analysis) ===")
    for nickname, data in jerk_results.items():
        print(f"\n{nickname}:")
        print(f"  RMS Jerk: {np.sqrt(np.mean(data['jerk_magnitude']**2)):.4f} m/s³")
        print(f"  Mean Jerk: {np.mean(data['jerk_magnitude']):.4f} m/s³")
        print(f"  Max Jerk: {np.max(data['jerk_magnitude']):.4f} m/s³")
        print(f"  95th percentile Jerk: {np.percentile(data['jerk_magnitude'], 95):.4f} m/s³")
        print(f"  RMS Acceleration: {np.sqrt(np.mean(data['acceleration_magnitude']**2)):.4f} m/s²")
        print(f"  Mean Speed: {np.mean(data['speed']):.4f} m/s")
        print(f"  Mean Curvature: {np.mean(data['curvature']):.6f} 1/m")
        print(f"  Total trajectory length: {np.sum(np.sqrt(np.diff(data['x'])**2 + np.diff(data['y'])**2)):.2f}m")


if __name__ == "__main__":
    main()