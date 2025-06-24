#!/usr/bin/env python3
# Example: Load data from multiple MCAP bags with nicknames

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_processor.data_loader.data_loader import DataLoader
from data_processor.bag_converter.bag_converter import BagConfig

def main():
    csv_dir = "/tmp/CSVs"
    
    # Configure multiple bags with topic nicknames
    bag_configs = [
        BagConfig(
            bag_path="/media/dm0/Matrix1/bags/bag1.mcap",
            topics={
                "/sensors/novatel/gps_a/gps": "gps_a",
                "/sensors/novatel/gps_b/gps": "gps_b"
            },
            nickname="run1"
        ),
        BagConfig(
            bag_path="/media/dm0/Matrix1/bags/bag2.mcap",
            topics={
                "/sensors/novatel/gps_a/gps": "gps_a", 
                "/sensors/novatel/gps_b/gps": "gps_b"
            },
            nickname="run2"
        )
    ]
    
    # Load data - auto-converts if CSVs don't exist
    loader = DataLoader(bag_configs=bag_configs, output_dir=csv_dir)
    db = loader.load_all()
    
    print("Loaded topics:", db.topics())
    
    # Access data by nickname
    gps_a_run1 = db["gps_a_run1"]
    gps_b_run1 = db["gps_b_run1"] 
    gps_a_run2 = db["gps_a_run2"]
    gps_b_run2 = db["gps_b_run2"]
    
    print(f"GPS A Run1: {len(gps_a_run1)} rows")
    print(f"GPS B Run1: {len(gps_b_run1)} rows")
    print(f"GPS A Run2: {len(gps_a_run2)} rows")
    print(f"GPS B Run2: {len(gps_b_run2)} rows")

if __name__ == "__main__":
    main()