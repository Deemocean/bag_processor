"""
Helper functions for filtering and processing DataFrame records.
"""
import pandas as pd
import numpy as np

from bisect import bisect_left
from typing import List, Optional

# import open3d as o3d
from typing import Tuple

def get_traj_file_df(
    file_path: str) -> pd.DataFrame:
    """
    Load trajectory data from a file into a DataFrame.

    Args:
        file_path: Path to the trajectory file.

    Returns:
        DataFrame containing the trajectory data.
    """
    df = pd.read_csv(file_path, sep=" ", header=None)
    df.columns = ["timestamp", "x", "y", "z", "qx", "qy", "qz", "qw"]
    return df

def select_time_range(
    df: pd.DataFrame,
    start: float,
    end: float,
    time_column: str = "header_t"
) -> pd.DataFrame:
    """
    Return rows where `time_column` is between `start` and `end` (inclusive).

    Args:
        df: DataFrame containing time data.
        start: Lower bound for time filtering.
        end:   Upper bound for time filtering.
        time_column: name of the time column in df.
    """
    mask = (df[time_column] >= start) & (df[time_column] <= end)
    return df.loc[mask].reset_index(drop=True)

def sync_time(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    time_label1: str,
    time_label2: str,
    method: str = "closest",
    suffix: Optional[str] = None
) -> pd.DataFrame:
    """
    Synchronize df2 onto df1’s timeline using pandas.merge_asof for performance.

    Args:
        df1: Primary DataFrame (must be sorted by time_label1).
        df2: Secondary DataFrame (must be sorted by time_label2).
        time_label1: Time column in df1.
        time_label2: Time column in df2.
        method: 'closest' or 'interpolate'.
        suffix: suffix to append to df2 columns on conflict.
    Returns:
        Merged DataFrame with df1 columns plus df2 data, with conflicting df2 columns suffixed.
    """
    # sort
    df1s = df1.sort_values(time_label1).reset_index(drop=True)
    df2s = df2.sort_values(time_label2).reset_index(drop=True)

    # default suffix
    if suffix is None:
        suffix = "_sync"

    # perform merge based on method
    if method == "closest":
        merged = pd.merge_asof(
            df1s,
            df2s,
            left_on=time_label1,
            right_on=time_label2,
            direction="nearest",
            suffixes=("", suffix)
        )
    elif method == "interpolate":
        # reindex and interpolate df2 to df1 times
        df2i = (
            df2s.set_index(time_label2)
                .reindex(df1s[time_label1], method="nearest")
                .interpolate(method="index")
                .reset_index()
        )
        # rename time column
        df2i = df2i.rename(columns={time_label2: time_label1})
        # determine conflicting columns
        conflicts = set(df1s.columns) & set(df2i.columns) - {time_label1}
        # build rename mapping
        rename_map = {col: f"{col}{suffix}" for col in conflicts}
        df2i = df2i.rename(columns=rename_map)
        # merge horizontally, dropping duplicate time
        merged = pd.concat(
            [df1s, df2i.drop(columns=[time_label1])],
            axis=1
        )
    else:
        raise ValueError("method must be 'closest' or 'interpolate'")

    # restore original index
    if hasattr(df1, 'index'):
        merged.index = df1.index
    return merged

def sync_times(
    df1: pd.DataFrame,
    dfs: List[pd.DataFrame],
    time_label1: str,
    time_labels2: List[str],
    method: str = "closest",
    suffixes: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Sequentially sync multiple DataFrames onto df1’s timeline.

    Args:
        df1: Primary DataFrame.
        dfs: List of secondary DataFrames to sync.
        time_label1: Time column in df1.
        time_labels2: Corresponding time columns in each df2.
        method: 'closest' or 'interpolate'.
        suffixes: Optional list of custom suffixes for each df2. If provided,
                  len(suffixes) must equal len(dfs). Otherwise, defaults to ['_2', '_3', ...].
    Returns:
        DataFrame with df1 columns plus merged columns from all dfs.
    """
    if len(dfs) != len(time_labels2):
        raise ValueError("dfs and time_labels2 must have the same length")
    if suffixes and len(suffixes) != len(dfs):
        raise ValueError("suffixes must match length of dfs if provided")

    # generate default suffixes if none
    if suffixes is None:
        suffixes = [f"_{i+2}" for i in range(len(dfs))]

    result = df1.copy().reset_index(drop=True)
    for df2, tl2, sfx in zip(dfs, time_labels2, suffixes):
        result = sync_time(result, df2, time_label1, tl2, method, suffix=sfx)
    return result.reset_index(drop=True)
