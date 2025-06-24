#!/usr/bin/env python3
# Example: Convert multiple MCAP bag files with topic nicknames

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_processor.bag_converter.bag_converter import BagLoader, BagConfig

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
    
    # Process bags - CSV files will be named by final nicknames
    loader = BagLoader(bag_configs, csv_dir)
    loader.run_all()
    
    # Generated files:
    # - gps_a_run1.csv
    # - gps_b_run1.csv  
    # - gps_a_run2.csv
    # - gps_b_run2.csv

if __name__ == "__main__":
    main()