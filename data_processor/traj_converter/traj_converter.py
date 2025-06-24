import os
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

class TrajectoryConverter:
    def __init__(
        self,
        glim_dir: str,
        output_dir: str
    ):
        """
        glim_dir   : path to GLIM output deliverables
        output_dir : where CSV lives
        """
        self.glim_dir   = glim_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self) -> pd.DataFrame:
        """Reads GLIM trajectory into DataFrame with CSV caching."""

        cached_trajectory = os.path.join(self.output_dir, "traj.csv")

        if not os.path.exists(cached_trajectory):
            # transform reference traj

            trajectory_path = os.path.join(self.glim_dir, "traj_imu.txt")
            transform_path = os.path.join(self.glim_dir, "T_enu_map.txt")
            
            traj_df = pd.read_csv(trajectory_path, sep=" ", header=None)
            traj_df.columns = ["timestamp", "x", "y", "z", "qx", "qy", "qz", "qw"]

            isometry_df = pd.read_csv(transform_path, sep=" ", header=None)
            isometry_df.columns = ["x", "y", "z", "qx", "qy", "qz", "qw"]

            traj_pos = traj_df[["x", "y", "z"]].to_numpy()
            traj_att = R.from_quat(traj_df[["qx", "qy", "qz", "qw"]].to_numpy())

            isometry_trans = isometry_df[["x", "y", "z"]].iloc[0].to_numpy()
            isometry_rot = R.from_quat(isometry_df[["qx", "qy", "qz", "qw"]].iloc[0].to_numpy())

            # Apply the isometric transformation to the positions and orientations
            transformed_pos = (isometry_rot.apply(traj_pos)) + isometry_trans
            transformed_att = isometry_rot * traj_att

            transformed_quat = transformed_att.as_quat()

            output_df = pd.DataFrame({
                "header_t": traj_df["timestamp"],
                "x": transformed_pos[:, 0],
                "y": transformed_pos[:, 1],
                "z": transformed_pos[:, 2],
                "qx": transformed_quat[:, 0],
                "qy": transformed_quat[:, 1],
                "qz": transformed_quat[:, 2],
                "qw": transformed_quat[:, 3],
            })

            output_df.to_csv(cached_trajectory, index=False)

        else:
            output_df = pd.read_csv(cached_trajectory)

        return output_df