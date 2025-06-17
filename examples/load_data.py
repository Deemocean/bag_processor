#!/usr/bin/env python3
# This script converts an MCAP bag file to CSV files for specified topics (If not exist).
# Then it uses DataLoader to read those CSVs into pandas df format.

import os
import sys
# add project root to PYTHONPATH
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)
from data_processor.data_loader.data_loader import DataLoader


def main():
    # =====================================================
    # Output directory for CSV files
    csv_dir = "/media/dm0/Matrix1/bags/2025-05-XX_California1/2025-05-15/2025-05-15-16-32-31/processed"
    # Bag file path
    bag_path = "/media/dm0/Matrix1/bags/2025-05-XX_California1/2025-05-15/2025-05-15-16-32-31/2025-05-15-16-32-31.mcap"
    # Topics to load from the bag file
    topics = ["/sensors/novatel/gps_a/gps", "/sensors/novatel/gps_b/gps"]
    # =====================================================

    loader = DataLoader(bag_path=bag_path, topics=topics, output_dir=csv_dir)
    db = loader.load_all()
    print("Loaded topics:", db.topics())
    

if __name__ == "__main__":
    main()

