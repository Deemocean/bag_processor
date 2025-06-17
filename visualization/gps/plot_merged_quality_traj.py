#!/usr/bin/env python3

import os
import sys
# ensure project root on PYTHONPATH
top_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, top_dir)

import matplotlib.pyplot as plt
import numpy as np
from pymap3d import geodetic2enu

from data_processor.data_loader.data_loader import DataLoader
from data_processor.helpers.helpers import select_time_range, sync_times


def main():
    # =====================================================
    csv_dir = "/media/dm0/Matrix1/bags/2025-05-XX_California1/2025-05-15/2025-05-15-16-32-31/processed"
    bag_path = "/media/dm0/Matrix1/bags/2025-05-XX_California1/2025-05-15/2025-05-15-16-32-31/2025-05-15-16-32-31.mcap"
    topics = [
        "/sensors/novatel/gps_a/bestgnsspos",
        "/sensors/novatel/gps_a/gps",
        "/sensors/novatel/gps_b/bestgnsspos",
        "/sensors/novatel/gps_b/gps",
    ]
    # =====================================================

    # Load CSVs via DataLoader (generates if missing)
    loader = DataLoader(bag_path=bag_path, topics=topics, output_dir=csv_dir)
    db = loader.load_all()
    print("Loaded topics:", db.topics())

    # Define time window
    start = db[topics[1]]["header_t"].min()
    end = start + 500.0

    # Extract time ranges
    fix_a = select_time_range(db[topics[1]], start, end)
    best_a = select_time_range(db[topics[0]], start, end)
    fix_b = select_time_range(db[topics[3]], start, end)
    best_b = select_time_range(db[topics[2]], start, end)

    # Sync so we use bestgnsspos times, and gdop from fix
    df_a = sync_times(best_a, [fix_a], "header_t", ["header_t"], method="closest")
    df_b = sync_times(best_b, [fix_b], "header_t", ["header_t"], method="closest")

    # Print synchronized DataFrames
    print("df_a:\n", df_a.head())
    print("df_b:\n", df_b.head())

    # Vectorized LLA → ENU
    origin = [36.583880, -121.752955, 250.00]
    gdop_max = 5.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    for df, ax, title in [(df_a, axes[0], "GPS A"), (df_b, axes[1], "GPS B")]:
        lats = df["lat"].to_numpy()
        lons = df["lon"].to_numpy()
        alts = df["alt"].to_numpy()
        gdop = df["gdop"].to_numpy()

        # convert LLA to ENU
        x, y, _ = geodetic2enu(lats, lons, alts, *origin)

                # downsample for performance and clip GDOP
        skip = max(1, len(x) // 10000)
        x = x[::skip]
        y = y[::skip]
        gdop = gdop[::skip]
        # clip GDOP and apply same downsampling
        gdop_clipped = np.clip(gdop, 0, gdop_max)
        
        sc = ax.scatter(
            x, y,
            c=gdop_clipped,
            cmap="viridis",
            vmin=0, vmax=gdop_max,
            s=5
        )
        ax.set_title(title)
        ax.set_xlabel("East (m)")
        ax.set_ylabel("North (m)")
        ax.grid(True)
        ax.set_aspect("equal", adjustable="box")

    # Shared colorbar
    fig.colorbar(sc, label=f"GDOP (≤{gdop_max})", ax=axes.ravel().tolist(), orientation="vertical")
    plt.suptitle("GPS A vs B Trajectories (BestGNSSpos LLA + Fix GDOP)")
    #plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
