#!/usr/bin/env python3

import os
import sys
# ensure project root on PYTHONPATH
top_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, top_dir)

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
from pymap3d import geodetic2enu
import open3d as o3d

from data_processor.data_loader.data_loader import DataLoader, Database
from data_processor.helpers.helpers import select_time_range, sync_times
from data_processor.traj_converter.traj_converter import TrajectoryConverter

def make_thick_wall_mesh(x, y, z_offsets, thickness, half_height, color):
    half_thick = thickness / 2.0
    verts, tris = [], []
    for i in range(len(x) - 1):
        x0, y0 = x[i],   y[i]
        x1, y1 = x[i+1], y[i+1]
        z0, z1 = z_offsets[i], z_offsets[i+1]
        dx, dy = x1 - x0, y1 - y0
        norm = np.hypot(dx, dy)
        if norm == 0: continue
        t = np.array([dx, dy]) / norm
        n = np.array([-t[1], t[0]])
        p0 = np.array([x0, y0]) + n * half_thick
        p1 = np.array([x0, y0]) - n * half_thick
        p2 = np.array([x1, y1]) + n * half_thick
        p3 = np.array([x1, y1]) - n * half_thick
        segment_verts = [
            [*p0, z0 - half_height],
            [*p1, z0 - half_height],
            [*p2, z1 - half_height],
            [*p3, z1 - half_height],
            [*p0, z0 + half_height],
            [*p1, z0 + half_height],
            [*p2, z1 + half_height],
            [*p3, z1 + half_height],
        ]
        base = len(verts)
        verts.extend(segment_verts)
        tris += [
            [base+0, base+2, base+1], [base+2, base+3, base+1],
            [base+4, base+5, base+6], [base+6, base+5, base+7],
            [base+0, base+4, base+2], [base+2, base+4, base+6],
            [base+1, base+3, base+5], [base+3, base+7, base+5],
            [base+0, base+1, base+4], [base+1, base+5, base+4],
            [base+2, base+7, base+3], [base+2, base+6, base+7],
        ]
    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(verts),
        triangles=o3d.utility.Vector3iVector(tris)
    )
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    return mesh

def main():
    # =======================
    wall_inner_path = "/home/dm0/workspace/mins_ws/src/track_description/bounds/laguna_seca/lagunaseca_real_wall_main_inner.csv"
    wall_outer_path = "/home/dm0/workspace/mins_ws/src/track_description/bounds/laguna_seca/lagunaseca_real_wall_main_outer.csv"
    # =======================
    db = Database({})
    # add wall boundaries
    db.add_topic("wall_inner", pd.read_csv(wall_inner_path))
    db.add_topic("wall_outer", pd.read_csv(wall_outer_path))
    # =======================
    map_pcd_path = "/media/dm0/Matrix1/LIDAR_MAPS/ls/large_ls_v2.pcd"
    # convert LLA → ENU
    origin = [36.583880, -121.752955, 250.00]
    horizontal_thickness = 0.5   # meters
    vertical_half_height  = 5.0  # meters
    window_size           = 300   # moving-average window (odd = symmetric)
    sample_count          = 10000 # points per wall mesh

    # wall boundaries
    wall_inner_lats = db["wall_inner"]["lat"].to_numpy()
    wall_inner_lons = db["wall_inner"]["lon"].to_numpy()
    wall_outer_lats = db["wall_outer"]["lat"].to_numpy()
    wall_outer_lons = db["wall_outer"]["lon"].to_numpy()

    xi, yi, _ = geodetic2enu(wall_inner_lats,
                              wall_inner_lons,
                              np.zeros(len(wall_inner_lats)),
                              *origin)
    xo, yo, _ = geodetic2enu(wall_outer_lats,
                              wall_outer_lons,
                              np.zeros(len(wall_outer_lats)),
                              *origin)
    # Load & colorize LiDAR
    pcd = o3d.io.read_point_cloud(map_pcd_path)
    pcd.paint_uniform_color([1.0, 1.0, 1.0])
    pts = np.asarray(pcd.points)

    # KDTree for local height
    kdt = o3d.geometry.KDTreeFlann(pcd)
    def sample_z(xarr, yarr):
        zs = []
        for xpt, ypt in zip(xarr, yarr):
            _, idx, _ = kdt.search_knn_vector_3d([xpt, ypt, 0.0], 1)
            zs.append(pts[idx[0], 2])
        return np.array(zs)

    inner_z = sample_z(xi, yi)
    outer_z = sample_z(xo, yo)

    # Smooth heights
    kernel   = np.ones(window_size) / window_size
    inner_z  = np.convolve(inner_z, kernel, mode='same')
    outer_z  = np.convolve(outer_z, kernel, mode='same')

    # Build thick meshes
    inner_mesh = make_thick_wall_mesh(xi, yi, inner_z,
                                      horizontal_thickness,
                                      vertical_half_height,
                                      color=[1,0,0])
    outer_mesh = make_thick_wall_mesh(xo, yo, outer_z,
                                      horizontal_thickness,
                                      vertical_half_height,
                                      color=[1,0,0])

    # ==== Sample mesh surfaces into point clouds (Poisson-disk) ====
    inner_sampled = inner_mesh.sample_points_poisson_disk(
        number_of_points=sample_count,
        init_factor=5
    )
    outer_sampled = outer_mesh.sample_points_poisson_disk(
        number_of_points=sample_count,
        init_factor=5
    )
    inner_sampled.paint_uniform_color([1.0, 0.0, 0.0])
    outer_sampled.paint_uniform_color([1.0, 0.0, 0.0])

    # ==== Save combined point cloud as PCD with wall surfaces ====
    combined_pts  = np.vstack((
        pts,
        np.asarray(inner_sampled.points),
        np.asarray(outer_sampled.points)
    ))
    combined_cols = np.vstack((
        np.asarray(pcd.colors),
        np.asarray(inner_sampled.colors),
        np.asarray(outer_sampled.colors)
    ))
    pcd_out = o3d.geometry.PointCloud()
    pcd_out.points = o3d.utility.Vector3dVector(combined_pts)
    pcd_out.colors = o3d.utility.Vector3dVector(combined_cols)
    o3d.io.write_point_cloud("/media/dm0/Matrix1/LIDAR_MAPS/ls/track_with_walls.pcd", pcd_out)
    print("Saved track_with_walls.pcd with true wall surfaces.")

    # Visualize
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    opts = vis.get_render_option()
    opts.background_color = np.asarray([0,0,0])
    opts.point_size       = 1.0

    vis.add_geometry(pcd)
    vis.add_geometry(inner_mesh)
    vis.add_geometry(outer_mesh)

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
