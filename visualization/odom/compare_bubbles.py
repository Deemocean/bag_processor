#!/usr/bin/env python3
"""
Compare old and new bubble configurations with GPS trajectory
Shows both bubble sets on same plot with GPS path
"""

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
from matplotlib.patches import Circle
import matplotlib.patches as mpatches
from pymap3d import geodetic2enu

from data_processor.data_loader.data_loader import DataLoader
from data_processor.bag_converter.bag_converter import BagConfig
from data_processor.helpers.helpers import select_time_range

# Old bubble configuration
old_bubbles = {
    'bubble0': {'center': [-340.0, 248.0], 'radius': 40.0, 'max_cov': 8.0},
    'bubble1': {'center': [-198.0, 94.0], 'radius': 40.0, 'max_cov': 8.0},
    'bubble2': {'center': [-105.0, -290.0], 'radius': 40.0, 'max_cov': 8.0},
    'bubble3': {'center': [181.0, -407.0], 'radius': 40.0, 'max_cov': 6.0}  # Note: Turn 6 Bridge
}

# New bubble configuration
new_bubbles = {
    'bubble0': {'center': [-361.0, 240.0], 'radius': 40.0, 'max_cov': 8.0},
    'bubble1': {'center': [-160.0, 88.0], 'radius': 40.0, 'max_cov': 8.0},
    'bubble2': {'center': [-105.0, -290.0], 'radius': 40.0, 'max_cov': 8.0},
    'bubble3': {'center': [199.0, -404.0], 'radius': 40.0, 'max_cov': 8.0},
    'bubble4': {'center': [300.0, 251.0], 'radius': 40.0, 'max_cov': 8.0}  # New bubble
}


def get_topics_by_nickname(db, nickname):
    """Get all topics for a given nickname."""
    topics = {}
    for topic in db.topics():
        if topic.endswith(f"_{nickname}"):
            base_name = topic.rsplit(f"_{nickname}", 1)[0]
            topics[base_name] = topic
    return topics


def plot_bubbles_comparison():
    csv_dir = "/tmp/CSVs"
    origin = [36.583880, -121.752955, 250.00]
    
    bag_configs = [
        BagConfig(
            bag_path="/media/dm0/Matrix1/bags/ca2/2025-07-07/2025-07-07-16-29-11/2025-07-07-16-29-11.mcap",
            topics={
                "/state/odom_raw": "odom_raw",
                "/state/odom": "odom",
                "/sensors/manager/gps0": "gps_a",
                "/sensors/manager/gps1": "gps_b",
                "/localization/debug/gps_dist": "gps_innovation"
            },
            nickname="7/7 run3"
        ),
    ]
    
    # Load data from both bags
    loader = DataLoader(bag_configs=bag_configs, output_dir=csv_dir)
    db = loader.load_all()
    print("Loaded topics:", db.topics())

    # Get nicknames from bag configs
    nicknames = [config.nickname for config in bag_configs]
    duration = 2000
    
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
        if "gps_a" in topics:
            run_data[nickname]["gps_a"] = select_time_range(db[topics["gps_a"]], start_time, end_time)
        if "gps_b" in topics:
            run_data[nickname]["gps_b"] = select_time_range(db[topics["gps_b"]], start_time, end_time)
        
        # Convert GPS to ENU
        for gps_key in ["gps_a", "gps_b"]:
            if gps_key in run_data[nickname]:
                gps_df = run_data[nickname][gps_key]
                x, y, z = geodetic2enu(
                    gps_df["lat"].values, gps_df["lon"].values, gps_df["alt"].values, *origin
                )
                run_data[nickname][f"{gps_key}_enu"] = (x, y, z)
    
    # Create single figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_title('Bubble Configuration Comparison with GPS Trajectory', fontsize=16, fontweight='bold')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Plot GPS trajectories
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, nickname in enumerate(nicknames):
        if nickname not in run_data:
            continue
            
        color = colors[i % len(colors)]
        
        # Plot GPS A points
        if "gps_a_enu" in run_data[nickname]:
            x, y, z = run_data[nickname]["gps_a_enu"]
            ax.scatter(x, y, c=color, s=5, label=f"GPS A {nickname}", alpha=0.5)
        
        # Plot GPS B points
        if "gps_b_enu" in run_data[nickname]:
            x, y, z = run_data[nickname]["gps_b_enu"]
            ax.scatter(x, y, c=color, s=5, label=f"GPS B {nickname}", alpha=0.5, marker='^')
    
    # Plot old bubbles (grey)
    for name, bubble in old_bubbles.items():
        center = bubble['center']
        radius = bubble['radius']
        circle = Circle(center, radius, fill=False, edgecolor='grey', linewidth=2, linestyle='--', alpha=0.7)
        ax.add_patch(circle)
        ax.plot(center[0], center[1], 'o', color='grey', markersize=8, alpha=0.7)
        ax.text(center[0], center[1]-radius-20, f"{name} (old)", ha='center', fontsize=9, color='grey')
    
    # Plot new bubbles (green)
    for name, bubble in new_bubbles.items():
        center = bubble['center']
        radius = bubble['radius']
        circle = Circle(center, radius, fill=False, edgecolor='green', linewidth=2)
        ax.add_patch(circle)
        ax.plot(center[0], center[1], 'o', color='green', markersize=8)
        
        # Label for new bubble
        if name == 'bubble4':
            ax.text(center[0], center[1]+radius+15, f"{name} (NEW)", ha='center', fontsize=10, 
                   color='green', fontweight='bold')
        else:
            ax.text(center[0], center[1]+radius+15, name, ha='center', fontsize=10, color='green')
    
    # Set axis limits
    all_x = []
    all_y = []
    for bubbles in [old_bubbles, new_bubbles]:
        for bubble in bubbles.values():
            all_x.append(bubble['center'][0])
            all_y.append(bubble['center'][1])
    
    margin = 100
    ax.set_xlim([min(all_x) - margin, max(all_x) + margin])
    ax.set_ylim([min(all_y) - margin, max(all_y) + margin])
    
    # Legend
    grey_patch = mpatches.Patch(color='grey', label='Old bubbles')
    green_patch = mpatches.Patch(color='green', label='New bubbles')
    legend_elements = [grey_patch, green_patch]
    
    # Add GPS trajectory legend elements
    legend_elements.append(plt.Line2D([0], [0], color='red', marker='o', linestyle='None', 
                                     markersize=5, label='GPS A 7/7 run3'))
    legend_elements.append(plt.Line2D([0], [0], color='blue', marker='o', linestyle='None', 
                                     markersize=5, label='GPS A 7/7 run3 chi1.5_50_1.1_newbubble'))
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_bubbles_comparison()