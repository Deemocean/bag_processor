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

from mpl_toolkits.mplot3d import Axes3D  # registers the 3D projection
import numpy as np
from pymap3d import geodetic2enu
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt

from data_processor.data_loader.data_loader import DataLoader
from data_processor.bag_converter.bag_converter import BagConfig
from data_processor.helpers.helpers import select_time_range


class Bubble:
    # hard-coded 
    _cfg = {
        'bubble0': {
            'bubble_center': [-340.0, 248.0],
            'bubble_radius':  40.0,
            'bubble_max_cov':  8.0
        },
        'bubble1': {
            'bubble_center': [-198.0,  94.0],
            'bubble_radius':  40.0,
            'bubble_max_cov':  8.0
        },
        'bubble2': {
            'bubble_center': [-105.0, -290.0],
            'bubble_radius':  40.0,
            'bubble_max_cov':  8.0
        },
        'bubble3': {
            'bubble_center': [ 181.0, -407.0],
            'bubble_radius':  40.0,
            'bubble_max_cov':  8.0
        },
    }

    def __init__(self):
        self.bubbles = []
        for cfg in self._cfg.values():
            center = np.array(cfg['bubble_center'], dtype=float)
            radius = float(cfg['bubble_radius'])
            max_cov = float(cfg['bubble_max_cov'])
            self.bubbles.append((center, radius, max_cov))

    def get_cov(self, x, y):
        pt = np.array([x, y], dtype=float)
        cov_vals = []
        for center, radius, max_cov in self.bubbles:
            d = np.linalg.norm(pt - center)
            if d < radius:
                cov_vals.append(max_cov * (1.0 - d / radius) ) 

        cov = max(cov_vals) if cov_vals else 0.0
        # addtional_cov = 2.0  # actually we were also adding the cov from GPS
        # cov += addtional_cov
        return np.diag([cov, cov, cov])
    
    def get_cov_trace(self, x, y):
        cov = self.get_cov(x, y)
        return np.trace(cov)


def load_gps_covariance(gps_df, nickname=None, gps_enu=None):
    """Convert GPS covariance from CSV list format to numpy 3x3 matrices."""
    covariances = []
    bubble = Bubble() if nickname in ["stable", "optv1_bubble", "stable_newcommits"] else None
    
    for i, cov_str in enumerate(gps_df["covariance"]):
        # Parse the list string and convert to 3x3 numpy array
        cov_list = eval(cov_str)  # Convert string representation back to list
        cov_matrix = np.array(cov_list)
        
        # Add bubble covariance for specified nicknames
        if bubble is not None and gps_enu is not None:
            x, y, z = gps_enu
            bubble_cov = bubble.get_cov(x[i], y[i])
            cov_matrix += bubble_cov
            
        covariances.append(cov_matrix)
    return np.array(covariances)


def extract_covariance_diagonals(cov_matrices):
    """Extract diagonal values from covariance matrices."""
    diagonals = np.array([np.diag(cov) for cov in cov_matrices])
    return diagonals  # Shape: (n_samples, 3) for [x_var, y_var, z_var]


def quaternion_to_heading_deg(qx, qy, qz, qw):
    """Convert quaternion to heading in degrees (yaw angle projected on XY plane)."""
    # Calculate yaw (rotation around Z-axis) from quaternion
    yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    # Convert from radians to degrees
    heading_deg = np.degrees(yaw)
    return heading_deg


def get_topics_by_nickname(db, nickname):
    """Get all topics for a given nickname."""
    topics = {}
    for topic in db.topics():
        if topic.endswith(f"_{nickname}"):
            base_name = topic.rsplit(f"_{nickname}", 1)[0]
            topics[base_name] = topic
    return topics


def main():
    csv_dir = "/tmp/CSVs"
    origin = [36.583880, -121.752955, 250.00]
    
    # Configure two bags with odometry data
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
    
    # Load data from both bags
    loader = DataLoader(bag_configs=bag_configs, output_dir=csv_dir)
    db = loader.load_all()
    print("Loaded topics:", db.topics())

    # Get nicknames from bag configs
    nicknames = [config.nickname for config in bag_configs]
    duration = 300  # 5 minutes duration
    
    # Process data for each bag
    run_data = {}
    for nickname in nicknames:
        topics = get_topics_by_nickname(db, nickname)
        if not topics:
            print(f"No topics found for nickname: {nickname}")
            continue
            
        # Get time range for this run
        odom_topic = topics.get("odom", topics.get("odom_raw"))
        if not odom_topic:
            print(f"No odometry topic found for {nickname}")
            continue
            
        start_time = db[odom_topic]["header_t"].min()
        end_time = start_time + duration
        
        # Load and filter data
        run_data[nickname] = {}
        if "odom" in topics:
            odom_data = select_time_range(db[topics["odom"]], start_time, end_time)
            run_data[nickname]["odom"] = odom_data
            # Calculate heading from quaternion if available
            if all(col in odom_data.columns for col in ["qx", "qy", "qz", "qw"]):
                heading = quaternion_to_heading_deg(
                    odom_data["qx"].values, odom_data["qy"].values, 
                    odom_data["qz"].values, odom_data["qw"].values
                )
                run_data[nickname]["heading"] = heading
            else:
                print(f"Warning: Quaternion data not found for {nickname} odom. Please regenerate CSV files.")
                
        if "odom_raw" in topics:
            odom_raw_data = select_time_range(db[topics["odom_raw"]], start_time, end_time)
            run_data[nickname]["odom_raw"] = odom_raw_data
            # Calculate heading from quaternion for raw odom too if available
            if all(col in odom_raw_data.columns for col in ["qx", "qy", "qz", "qw"]):
                heading_raw = quaternion_to_heading_deg(
                    odom_raw_data["qx"].values, odom_raw_data["qy"].values, 
                    odom_raw_data["qz"].values, odom_raw_data["qw"].values
                )
                run_data[nickname]["heading_raw"] = heading_raw
            else:
                print(f"Warning: Quaternion data not found for {nickname} odom_raw. Please regenerate CSV files.")
        if "gps_a" in topics:
            run_data[nickname]["gps_a"] = select_time_range(db[topics["gps_a"]], start_time, end_time)
        if "gps_b" in topics:
            run_data[nickname]["gps_b"] = select_time_range(db[topics["gps_b"]], start_time, end_time)
        if "gps_innovation" in topics:
            print(f"Loading GPS innovation for {nickname} from topic {topics['gps_innovation']}")
            print("before select_time_range:", db[topics["gps_innovation"]].head())
            run_data[nickname]["gps_innovation"] = select_time_range(db[topics["gps_innovation"]], start_time, end_time)
            print("after select_time_range:", run_data[nickname]["gps_innovation"].head())
        # Convert GPS to ENU and load covariance
        for gps_key in ["gps_a", "gps_b"]:
            if gps_key in run_data[nickname]:
                gps_df = run_data[nickname][gps_key]
                x, y, z = geodetic2enu(
                    gps_df["lat"].values, gps_df["lon"].values, gps_df["alt"].values, *origin
                )
                run_data[nickname][f"{gps_key}_enu"] = (x, y, z)
                
                # Load covariance matrices (with bubble covariance for stable)
                cov_matrices = load_gps_covariance(gps_df, nickname=nickname, gps_enu=(x, y, z))
                run_data[nickname][f"{gps_key}_cov"] = cov_matrices
                run_data[nickname][f"{gps_key}_diag"] = extract_covariance_diagonals(cov_matrices)
                # print(f"{nickname} {gps_key} covariance shape: {cov_matrices.shape}")
                if nickname == "stable":
                    print(f"Added bubble covariance for {nickname} {gps_key}")
                if nickname == "optv1_bubble":
                    print(f"Added bubble covariance for {nickname} {gps_key}")
                if nickname == "stable_newcommits":
                    print(f"Added bubble covariance for  {nickname} {gps_key}")

    # Create plots with trajectory on left, GPS A, B and innovation traces on right
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 1], height_ratios=[1, 1, 1])
    
    # Left: Trajectory plot (spans all rows)
    ax_traj = fig.add_subplot(gs[:, 0])
    
    # Right: Separate trace plots for GPS A, B, and innovation
    ax_trace_a = fig.add_subplot(gs[0, 1])
    ax_trace_b = fig.add_subplot(gs[1, 1])
    ax_innovation = fig.add_subplot(gs[2, 1])
    
    # Plot trajectory comparison
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, nickname in enumerate(nicknames):
        if nickname not in run_data:
            continue
            
        color = colors[i % len(colors)]
        
        # Plot odometry
        if "odom" in run_data[nickname]:
            odom = run_data[nickname]["odom"]
            ax_traj.plot(odom["x"], odom["y"], label=f"Odom {nickname}", 
                        linestyle='-', alpha=0.7, linewidth=2, color=color)
        
        # Plot GPS points
        if "gps_a_enu" in run_data[nickname]:
            x, y, z = run_data[nickname]["gps_a_enu"]
            ax_traj.scatter(x, y, c=color, s=1, label=f"GPS A {nickname}", alpha=0.5)
        if "gps_b_enu" in run_data[nickname]:
            x, y, z = run_data[nickname]["gps_b_enu"]
            ax_traj.scatter(x, y, c=color, s=1, label=f"GPS B {nickname}", alpha=0.3, marker='^')
    
    ax_traj.axis("equal")
    ax_traj.set_xlabel("X (m)")
    ax_traj.set_ylabel("Y (m)")
    ax_traj.set_title("Trajectory Comparison")
    ax_traj.legend()
    ax_traj.grid(True, alpha=0.3)
    
    # Store time-indexed trajectory data for synchronized zooming
    time_to_xy_data = {}
    
    # Plot GPS A covariance trace
    for i, nickname in enumerate(nicknames):
        if nickname not in run_data:
            continue
            
        color = colors[i % len(colors)]
        
        if "gps_a_cov" in run_data[nickname]:
            cov_matrices = run_data[nickname]["gps_a_cov"]
            time_vals = run_data[nickname]["gps_a"]["header_t"].values
            
            # Calculate trace for each covariance matrix
            trace_vals = np.array([np.trace(cov) for cov in cov_matrices])
            
            ax_trace_a.scatter(time_vals, trace_vals, 
                             label=f"GPS A {nickname}", 
                             color=color, alpha=0.7, s=1)
            
            # Store time-indexed odometry data for this nickname
            if "odom" in run_data[nickname]:
                odom = run_data[nickname]["odom"]
                time_to_xy_data[nickname] = {
                    'time': odom["header_t"].values,
                    'x': odom["x"].values,
                    'y': odom["y"].values
                }
    
    ax_trace_a.set_xlabel("Time (s)")
    ax_trace_a.set_ylabel("Covariance Trace (m²)")
    ax_trace_a.set_title("GPS A Covariance Trace Over Time")
    ax_trace_a.legend()
    ax_trace_a.grid(True, alpha=0.3)
    #ax_trace_a.set_yscale('log')
    
    # Plot GPS B covariance trace
    for i, nickname in enumerate(nicknames):
        if nickname not in run_data:
            continue
            
        color = colors[i % len(colors)]
        
        if "gps_b_cov" in run_data[nickname]:
            cov_matrices = run_data[nickname]["gps_b_cov"]
            time_vals = run_data[nickname]["gps_b"]["header_t"].values
            
            # Calculate trace for each covariance matrix
            trace_vals = np.array([np.trace(cov) for cov in cov_matrices])
            
            ax_trace_b.scatter(time_vals, trace_vals, 
                             label=f"GPS B {nickname}", 
                             color=color, alpha=0.7, s=1)
    
    ax_trace_b.set_xlabel("Time (s)")
    ax_trace_b.set_ylabel("Covariance Trace (m²)")
    ax_trace_b.set_title("GPS B Covariance Trace Over Time")
    ax_trace_b.legend()
    ax_trace_b.grid(True, alpha=0.3)
    #ax_trace_b.set_yscale('log')
    
    # Plot GPS innovation
    for i, nickname in enumerate(nicknames):
        if nickname not in run_data:
            continue
            
        color = colors[i % len(colors)]
        
        if "gps_innovation" in run_data[nickname]:
            innovation_data = run_data[nickname]["gps_innovation"]
            time_vals = innovation_data["header_t"].values
            data_vals = innovation_data["data"].values
            
            ax_innovation.plot(time_vals, data_vals, 
                             label=f"GPS Innovation {nickname}", 
                             color=color, alpha=0.7, linewidth=1)
    
    ax_innovation.set_xlabel("Time (s)")
    ax_innovation.set_ylabel("GPS Innovation (m)")
    ax_innovation.set_title("GPS Innovation Over Time")
    ax_innovation.legend()
    ax_innovation.grid(True, alpha=0.3)
    
    # Share x-axis between all time plots
    ax_trace_b.sharex(ax_trace_a)
    ax_innovation.sharex(ax_trace_a)
    
    # Function to update trajectory plot based on time range
    def update_trajectory_view(time_min, time_max):
        # Clear trajectory plot
        ax_traj.clear()
        
        # Re-plot trajectory with time filtering
        for i, nickname in enumerate(nicknames):
            if nickname not in run_data or nickname not in time_to_xy_data:
                continue
                
            color = colors[i % len(colors)]
            
            # Filter odometry data by time range
            time_data = time_to_xy_data[nickname]['time']
            x_data = time_to_xy_data[nickname]['x']
            y_data = time_to_xy_data[nickname]['y']
            
            # Find indices within time range
            mask = (time_data >= time_min) & (time_data <= time_max)
            
            # Plot filtered trajectory
            ax_traj.plot(x_data[mask], y_data[mask], 
                        label=f"Odom {nickname}", 
                        linestyle='-', alpha=0.7, linewidth=2, color=color)
            
            # Plot filtered GPS points
            if "gps_a_enu" in run_data[nickname] and "gps_a" in run_data[nickname]:
                gps_time = run_data[nickname]["gps_a"]["header_t"].values
                x, y, _ = run_data[nickname]["gps_a_enu"]
                gps_mask = (gps_time >= time_min) & (gps_time <= time_max)
                ax_traj.scatter(x[gps_mask], y[gps_mask], c=color, s=1, 
                               label=f"GPS A {nickname}", alpha=0.5)
                
            if "gps_b_enu" in run_data[nickname] and "gps_b" in run_data[nickname]:
                gps_time = run_data[nickname]["gps_b"]["header_t"].values
                x, y, _ = run_data[nickname]["gps_b_enu"]
                gps_mask = (gps_time >= time_min) & (gps_time <= time_max)
                ax_traj.scatter(x[gps_mask], y[gps_mask], c=color, s=1, 
                               label=f"GPS B {nickname}", alpha=0.3, marker='^')
        
        ax_traj.axis("equal")
        ax_traj.set_xlabel("X (m)")
        ax_traj.set_ylabel("Y (m)")
        ax_traj.set_title(f"Trajectory Comparison (Time: {time_min:.1f}s - {time_max:.1f}s)")
        ax_traj.legend()
        ax_traj.grid(True, alpha=0.3)
        ax_traj.figure.canvas.draw_idle()
    
    # Connect zoom event handler
    def on_xlim_changed(ax):
        time_min, time_max = ax.get_xlim()
        update_trajectory_view(time_min, time_max)
    
    # Connect the callback to x-axis limit changes
    ax_trace_a.callbacks.connect('xlim_changed', on_xlim_changed)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
