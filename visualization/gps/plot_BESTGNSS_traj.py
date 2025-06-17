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
from data_processor.helpers.helpers import select_time_range


def main():
    
    # =====================================================
    # Output directory for CSV files
    csv_dir = "/media/dm0/Matrix1/bags/2025-05-XX_California1/2025-05-15/2025-05-15-16-32-31/processed"
    # Bag file path
    bag_path = "/media/dm0/Matrix1/bags/2025-05-XX_California1/2025-05-15/2025-05-15-16-32-31/2025-05-15-16-32-31.mcap"
    # Topics to load from the bag file
    topics = ["/sensors/novatel/gps_a/bestgnsspos", "/sensors/novatel/gps_b/bestgnsspos"]
    # =====================================================

    # load CSV via DataLoader
    loader = DataLoader(bag_path=bag_path, topics=topics, output_dir=csv_dir)
    db = loader.load_all()
    print("Loaded topics:", db.topics())

    start_time = db["/sensors/novatel/gps_a/bestgnsspos"]["header_t"].min()
    end_time = start_time + 500
    
    gps_a_df = select_time_range(
        db["/sensors/novatel/gps_a/bestgnsspos"], start_time, end_time
    )
    gps_b_df = select_time_range(
        db["/sensors/novatel/gps_b/bestgnsspos"], start_time, end_time
    )


    # vectorized LLA → ENU
    origin = [36.583880, -121.752955, 250.00]
    gdop_max = 5

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 6),
                             sharex=True, sharey=True)
    for df, ax, title in [
        (gps_a_df, axes[0], "GPS A"),
        (gps_b_df, axes[1], "GPS B"),
    ]:
        lats = df["lat"].to_numpy()
        lons = df["lon"].to_numpy()
        alts = df["alt"].to_numpy()
        x, y, _ = geodetic2enu(lats, lons, alts, *origin)

        skip = max(1, len(x) // 10000)
        x, y= x[::skip], y[::skip]


        sc = ax.scatter(
            x, y,
            s=5
        )
        
        ax.set_title(title)
        ax.set_xlabel("East (m)")
        ax.set_ylabel("North (m)")
        ax.grid(True)
        ax.set_aspect("equal", adjustable="box")

    plt.suptitle("[BESTGNSS] GPS A vs B Trajectories")
    #plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
