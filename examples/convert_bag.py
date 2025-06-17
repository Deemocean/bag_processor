#!/usr/bin/env python3
# This script converts an MCAP bag file to CSV files for specified topics.

import os
import sys
# add project root to PYTHONPATH
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from data_processor.bag_converter.bag_converter import BagLoader

def main():
    # =====================================================
    # Output directory for CSV files
    csv_dir = "/media/dm0/Matrix1/bags/2025-05-XX_California1/2025-05-15/2025-05-15-16-32-31/processed"
    # Bag file path
    bag_path = "/media/dm0/Matrix1/bags/2025-05-XX_California1/2025-05-15/2025-05-15-16-32-31/2025-05-15-16-32-31.mcap"
    # Topics to load from the bag file
    topics = ["/sensors/novatel/gps_a/gps", "/sensors/novatel/gps_b/gps"]
    # =====================================================

    # Instantiate and run
    loader = BagLoader(bag_path, topics, csv_dir)
    loader.run_all()


if __name__ == "__main__":
    main()


