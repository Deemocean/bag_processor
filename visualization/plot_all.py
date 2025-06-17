#!/usr/bin/env python3

import os
import sys
# ensure project root on PYTHONPATH
top_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, top_dir)

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
from pymap3d import geodetic2enu

from data_processor.data_loader.data_loader import DataLoader
from data_processor.helpers.helpers import select_time_range, sync_times, get_traj_file_df

def main():
    # ==== Configuration ====  
    ref_path = "/home/dm0/Downloads/odom_imu.txt"  # timestamp x y z qx qy qz qw
    ref_topic  = "reference_traj"  # name for reference trajectory in database
    # =======================
    wall_inner_path = "/home/dm0/workspace/mins_ws/src/track_description/bounds/laguna_seca/lagunaseca_real_wall_pit_inner.csv"
    wall_outer_path = "/home/dm0/workspace/mins_ws/src/track_description/bounds/laguna_seca/lagunaseca_real_wall_pit_outer.csv"
    # =======================
    csv_dir = "/media/dm0/Matrix1/bags/2025-05-XX_California1/2025-05-14/2025-05-14-10-30-16/CSVs"
    bag_path = "/media/dm0/Matrix1/bags/2025-05-XX_California1/2025-05-14/2025-05-14-10-30-16/2025-05-14-10-30-16.mcap"
    topics = [
        "/sensors/novatel/gps_a/bestgnsspos",
        "/sensors/novatel/gps_a/gps",
        "/sensors/novatel/gps_b/bestgnsspos",
        "/sensors/novatel/gps_b/gps",
    ]
    # convert LLA → ENU
    origin = [36.583880, -121.752955, 250.00]
    # =======================

    # Load CSVs via DataLoader
    loader = DataLoader(bag_path=bag_path, topics=topics, output_dir=csv_dir)
    db = loader.load_all()
    # add reference trajectory
    ref_df = get_traj_file_df(ref_path)
    db.add_topic(ref_topic, ref_df)
    # add wall boundaries
    db.add_topic("wall_inner", pd.read_csv(wall_inner_path))
    db.add_topic("wall_outer", pd.read_csv(wall_outer_path))

    # time window
    start = db[topics[1]]["header_t"].min() + 100.0
    end = start + 300.0
    # select ranges
    fix_a = select_time_range(db[topics[1]], start, end)
    best_a = select_time_range(db[topics[0]], start, end)
    fix_b = select_time_range(db[topics[3]], start, end)
    best_b = select_time_range(db[topics[2]], start, end)
    ref_traj = select_time_range(db[ref_topic], start, end, time_column="timestamp")

    # sync gdop onto best times
    df_a = sync_times(best_a, [fix_a], "header_t", ["header_t"], method="closest")
    df_b = sync_times(best_b, [fix_b], "header_t", ["header_t"], method="closest")

    # A
    lats_a, lons_a, alts_a, gdop_a = (
        df_a["lat"].to_numpy(), df_a["lon"].to_numpy(),
        df_a["alt"].to_numpy(), df_a["gdop"].to_numpy()
    )
    x_a, y_a, _ = geodetic2enu(lats_a, lons_a, alts_a, *origin)
    # B
    lats_b, lons_b, alts_b, gdop_b = (
        df_b["lat"].to_numpy(), df_b["lon"].to_numpy(),
        df_b["alt"].to_numpy(), df_b["gdop"].to_numpy()
    )
    x_b, y_b, _ = geodetic2enu(lats_b, lons_b, alts_b, *origin)
    # reference 
    x_ref, y_ref = ref_traj["x"].to_numpy(), ref_traj["y"].to_numpy()

    # wall boundaries
    wall_inner_lats = db["wall_inner"]["lat"].to_numpy()
    wall_inner_lons = db["wall_inner"]["lon"].to_numpy()
    wall_outer_lats = db["wall_outer"]["lat"].to_numpy()
    wall_outer_lons = db["wall_outer"]["lon"].to_numpy()

    wall_inner_x, wall_inner_y, _ = geodetic2enu(
        wall_inner_lats, wall_inner_lons, np.zeros_like(wall_inner_lats), *origin
    )
    wall_outer_x, wall_outer_y, _ = geodetic2enu(
        wall_outer_lats, wall_outer_lons, np.zeros_like(wall_outer_lats), *origin
    )


    # plot
    fig, ax = plt.subplots(figsize=(8, 8))
    # scatter A with gdop colormap
    sc_a = plt.scatter(x_a, y_a, c=gdop_a, cmap="viridis", vmin=0, vmax=5, s=0.5, label="GPS A")
    # scatter B with different colormap
    sc_b = plt.scatter(x_b, y_b, c=gdop_b, cmap="plasma", vmin=0, vmax=5, s=0.5, label="GPS B")
    # overlay reference trajectory

    plt.plot(x_ref, y_ref, color="black", linewidth=0.5, label="Reference Traj")

    # plot wall boundaries
    plt.plot(wall_inner_x, wall_inner_y, color="red", linewidth=0.5)
    plt.plot(wall_outer_x, wall_outer_y, color="red", linewidth=0.5, label="Wall Boundaries")
    plt.xlabel("East (m)")
    plt.ylabel("North (m)")
    plt.axis("equal")
    plt.title("Trajectories")
    # combined legend
    plt.legend(loc="best")
    # single colorbar for both (normalized)
    divider = make_axes_locatable(ax)
    cax_a = divider.append_axes("right", size="2%", pad=0.2)
    cax_b = divider.append_axes("right", size="2%", pad=0.1)

    cb_a = fig.colorbar(sc_a, cax=cax_a)
    cb_a.ax.yaxis.set_ticks([])

    cb_b = fig.colorbar(sc_b, cax=cax_b)
    cb_b.set_label("GDOP", rotation=270, labelpad=10)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
