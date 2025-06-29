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


def main():
    csv_dir = "/tmp/CSVs"
    origin = [36.583880, -121.752955, 250.00]  # lat, lon, alt
    
        # Configure two bags with odometry data
    bag_configs = [
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
        
        print(f"Processing data for {nickname}")
        print(f"Topics for {nickname}: {topics}")
        
        # Load and filter data
        run_data[nickname] = {}
        
        # Load odometry data
        if "odom" in topics:
            odom_data = select_time_range(db[topics["odom"]], start_time, end_time)
            run_data[nickname]["odom"] = odom_data
            print(f"  Odom data shape: {odom_data.shape}")
            print(f"  Odom columns: {odom_data.columns.tolist()}")
            
        # Load GPS data
        if "gps_a" in topics:
            gps_a_data = select_time_range(db[topics["gps_a"]], start_time, end_time)
            run_data[nickname]["gps_a"] = gps_a_data
            print(f"  GPS A data shape: {gps_a_data.shape}")
            print(f"  GPS A columns: {gps_a_data.columns.tolist()}")
        if "gps_b" in topics:
            gps_b_data = select_time_range(db[topics["gps_b"]], start_time, end_time)
            run_data[nickname]["gps_b"] = gps_b_data
            print(f"  GPS B data shape: {gps_b_data.shape}")
            print(f"  GPS B columns: {gps_b_data.columns.tolist()}")
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Colors for different runs
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, (nickname, data) in enumerate(run_data.items()):
        color = colors[i % len(colors)]
        
        # Plot odometry altitude
        if "odom" in data:
            odom_df = data["odom"]
            
            # Find altitude column in odometry data
            alt_col = None
            for col in ['z', 'position_z', 'pose.pose.position.z']:
                if col in odom_df.columns:
                    alt_col = col
                    break
            
            if alt_col and 'header_t' in odom_df.columns and len(odom_df) > 0:
                # Convert timestamp to seconds from start
                time_sec = (odom_df['header_t'] - odom_df['header_t'].iloc[0]) / 1e9
                altitude = odom_df[alt_col]
                
                # Plot altitude
                ax1.plot(time_sec, altitude, label=f'{nickname} (Odom)', 
                        color=color, linewidth=2, linestyle='-')
                
                # Plot altitude change rate
                if len(altitude) > 1:
                    alt_rate = np.gradient(altitude, time_sec)
                    ax2.plot(time_sec, alt_rate, label=f'{nickname} (Odom)', 
                            color=color, linewidth=2, linestyle='-')
                
                print(f"{nickname} Odom altitude range: [{altitude.min():.2f}, {altitude.max():.2f}] m")
            else:
                print(f"{nickname} Odom: No altitude column found or no data")
        
        # Plot GPS A altitude
        if "gps_a" in data:
            gps_df = data["gps_a"]
            
            # Find altitude column in GPS data
            alt_col = None
            for col in ['alt', 'altitude', 'height', 'h','hgt']:
                if col in gps_df.columns:
                    alt_col = col
                    break
            
            if alt_col and 'header_t' in gps_df.columns and len(gps_df) > 0:
                time_sec = (gps_df['header_t'] - gps_df['header_t'].iloc[0]) / 1e9
                altitude_raw = gps_df[alt_col]
                
                # Convert GPS altitude to relative altitude (subtract origin altitude)
                altitude = altitude_raw - origin[2]  # origin[2] is the altitude component
                
                # Remove invalid values
                valid_mask = ~np.isnan(altitude) & (altitude_raw != 0)
                time_sec_valid = time_sec[valid_mask]
                altitude_valid = altitude[valid_mask]
                
                if len(altitude_valid) > 0:
                    # Plot altitude
                    ax1.plot(time_sec_valid, altitude_valid, label=f'{nickname} (GPS A)', 
                            color=color, linewidth=1.5, linestyle='--', alpha=0.8)
                    
                    # Plot altitude change rate
                    if len(altitude_valid) > 1:
                        alt_rate = np.gradient(altitude_valid, time_sec_valid)
                        ax2.plot(time_sec_valid, alt_rate, label=f'{nickname} (GPS A)', 
                                color=color, linewidth=1.5, linestyle='--', alpha=0.8)
                    
                    print(f"{nickname} GPS A altitude range: [{altitude_valid.min():.2f}, {altitude_valid.max():.2f}] m")
                else:
                    print(f"{nickname} GPS A: No valid altitude data")
            else:
                print(f"{nickname} GPS A: No altitude column found or no data")
        
        # Plot GPS B altitude
        if "gps_b" in data:
            gps_df = data["gps_b"]
            
            # Find altitude column in GPS data
            alt_col = None
            for col in ['alt', 'altitude', 'height', 'h','hgt']:
                if col in gps_df.columns:
                    alt_col = col
                    break
            
            if alt_col and 'header_t' in gps_df.columns and len(gps_df) > 0:
                time_sec = (gps_df['header_t'] - gps_df['header_t'].iloc[0]) / 1e9
                altitude_raw = gps_df[alt_col]
                
                # Convert GPS altitude to relative altitude (subtract origin altitude)
                altitude = altitude_raw - origin[2]  # origin[2] is the altitude component
                
                # Remove invalid values
                valid_mask = ~np.isnan(altitude) & (altitude_raw != 0)
                time_sec_valid = time_sec[valid_mask]
                altitude_valid = altitude[valid_mask]
                
                if len(altitude_valid) > 0:
                    # Plot altitude
                    ax1.plot(time_sec_valid, altitude_valid, label=f'{nickname} (GPS B)', 
                            color=color, linewidth=1.5, linestyle=':', alpha=0.8)
                    
                    # Plot altitude change rate
                    if len(altitude_valid) > 1:
                        alt_rate = np.gradient(altitude_valid, time_sec_valid)
                        ax2.plot(time_sec_valid, alt_rate, label=f'{nickname} (GPS B)', 
                                color=color, linewidth=1.5, linestyle=':', alpha=0.8)
                    
                    print(f"{nickname} GPS B altitude range: [{altitude_valid.min():.2f}, {altitude_valid.max():.2f}] m")
                else:
                    print(f"{nickname} GPS B: No valid altitude data")
            else:
                print(f"{nickname} GPS B: No altitude column found or no data")
    
    # Configure altitude plot
    ax1.set_ylabel('Altitude (m)', fontsize=12)
    ax1.set_title('Altitude Comparison Over Time', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Configure altitude rate plot
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Altitude Rate (m/s)', fontsize=12)
    ax2.set_title('Altitude Change Rate', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add zero line to rate plot
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.8)
    
    # Set time range (will be determined from actual data)
    # ax2.set_xlim(start_time, end_time)
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)  # Make room for legend
    plt.show()
    
    # Print summary statistics
    print("\n=== Altitude Summary Statistics ===")
    for nickname, data in run_data.items():
        print(f"\n{nickname}:")
        
        if "odom" in data and 'z' in data["odom"].columns:
            odom_alt = data["odom"]['z']
            valid_odom = odom_alt[~np.isnan(odom_alt)]
            if len(valid_odom) > 0:
                print(f"  Odom: mean={valid_odom.mean():.2f}m, std={valid_odom.std():.2f}m, "
                      f"range=[{valid_odom.min():.2f}, {valid_odom.max():.2f}]m")
        
        for gps_key in ["gps_a", "gps_b"]:
            if gps_key in data and 'alt' in data[gps_key].columns:
                gps_alt = data[gps_key]['alt']
                valid_gps = gps_alt[~np.isnan(gps_alt) & (gps_alt != 0)]
                if len(valid_gps) > 0:
                    print(f"  {gps_key.upper()}: mean={valid_gps.mean():.2f}m, std={valid_gps.std():.2f}m, "
                          f"range=[{valid_gps.min():.2f}, {valid_gps.max():.2f}]m")


if __name__ == "__main__":
    main()