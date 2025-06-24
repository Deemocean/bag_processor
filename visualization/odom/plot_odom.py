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


def main():
    csv_dir = "/tmp/CSVs"
    origin = [36.583880, -121.752955, 250.00]
    
    # Configure two bags with odometry data
    bag_configs = [
        BagConfig(
            bag_path="/media/dm0/Matrix1/bags/2025-05-XX_California1/2025-05-15/2025-05-15-16-32-31/2025-05-15-16-32-31.mcap",
            topics={
                "/state/odom": "odom",
                "/sensors/novatel/gps_a/bestgnsspos": "gps_a",
                "/sensors/novatel/gps_b/bestgnsspos": "gps_b"
            },
            nickname="run1"
        ),
        BagConfig(
            bag_path="/media/dm0/Matrix1/bags/2025-05-XX_California1/2025-05-14/2025-05-14-10-30-16/2025-05-14-10-30-16.mcap",
            topics={
                "/state/odom": "odom",
                "/sensors/novatel/gps_a/bestgnsspos": "gps_a",
                "/sensors/novatel/gps_b/bestgnsspos": "gps_b"
            },
            nickname="run2"
        )
    ]
    
    # Load data from both bags
    loader = DataLoader(bag_configs=bag_configs, output_dir=csv_dir)
    db = loader.load_all()
    print("Loaded topics:", db.topics())

    # Process Run 1 data
    if "odom_run1" in db.topics():
        start_time_1 = db["odom_run1"]["header_t"].min()
        end_time_1 = start_time_1 + 300
        
        odom_1 = select_time_range(db["odom_run1"], start_time_1, end_time_1)
        gps_a_1 = select_time_range(db["gps_a_run1"], start_time_1, end_time_1)
        gps_b_1 = select_time_range(db["gps_b_run1"], start_time_1, end_time_1)
        
        # Convert GPS to ENU for Run 1
        x_a_1, y_a_1, z_a_1 = geodetic2enu(
            gps_a_1["lat"].values, gps_a_1["lon"].values, gps_a_1["alt"].values, *origin
        )
        x_b_1, y_b_1, z_b_1 = geodetic2enu(
            gps_b_1["lat"].values, gps_b_1["lon"].values, gps_b_1["alt"].values, *origin
        )

    # Process Run 2 data
    if "odom_run2" in db.topics():
        start_time_2 = db["odom_run2"]["header_t"].min()
        end_time_2 = start_time_2 + 300
        
        odom_2 = select_time_range(db["odom_run2"], start_time_2, end_time_2)
        gps_a_2 = select_time_range(db["gps_a_run2"], start_time_2, end_time_2)
        gps_b_2 = select_time_range(db["gps_b_run2"], start_time_2, end_time_2)
        
        # Convert GPS to ENU for Run 2
        x_a_2, y_a_2, z_a_2 = geodetic2enu(
            gps_a_2["lat"].values, gps_a_2["lon"].values, gps_a_2["alt"].values, *origin
        )
        x_b_2, y_b_2, z_b_2 = geodetic2enu(
            gps_b_2["lat"].values, gps_b_2["lon"].values, gps_b_2["alt"].values, *origin
        )

    # Plot both runs
    plt.figure(figsize=(12, 8))
    
    # Run 1 data
    if "odom_run1" in db.topics():
        plt.plot(odom_1["x"], odom_1["y"], label="Odom Run1", linestyle='-', alpha=0.7, linewidth=2)
        plt.scatter(x_a_1, y_a_1, c='red', s=1, label="GPS A Run1", alpha=0.5)
        plt.scatter(x_b_1, y_b_1, c='blue', s=1, label="GPS B Run1", alpha=0.5)
    
    # Run 2 data
    if "odom_run2" in db.topics():
        plt.plot(odom_2["x"], odom_2["y"], label="Odom Run2", linestyle='--', alpha=0.7, linewidth=2)
        plt.scatter(x_a_2, y_a_2, c='darkred', s=1, label="GPS A Run2", alpha=0.5, marker='^')
        plt.scatter(x_b_2, y_b_2, c='darkblue', s=1, label="GPS B Run2", alpha=0.5, marker='^')

    plt.axis("equal")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("Odometry Comparison - Two Runs")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    main()
