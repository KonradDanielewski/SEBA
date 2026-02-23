"""
@author: K. Danielewski
"""

import datetime as dt
import os

import numpy as np
import pandas as pd

from seba.utils import auxiliary, auxfun_data


def read_bvs(
    rec_folder: str,
    behaview_output: str,
    onsets_only: bool = True,
):
    """Reads BehaActive behavior annotation data from .bvs files and saves timestamps of events and binary DataFrame

    Args:
        rec_folder: path to the folder containing corresponding recording ephys data
        filepath: path to the .bvs file
        onsets_only: if True saves a txt file with only onsets timestamps, otherwise a separate file with offset timestamps is also saved
    """
    try:
        df = pd.read_csv(
            behaview_output,
            skiprows=1,
            sep=" - ",
            header=None,
            names=["timestamp", "behavior"],
        )
    except FileNotFoundError:
        raise

    df.loc[:, "timestamp"] = pd.to_timedelta(
        (
            df["timestamp"]
            .apply(lambda x: dt.datetime.strptime(x, "%H:%M:%S:%f").time())
            .astype(str)
        )
    ).dt.total_seconds()

    df = df.set_index("timestamp")

    save_path = auxiliary.make_dir_save(rec_folder, "events")

    behaview_events(df, rec_folder, save_path, onsets_only)


def behaview_events(
    events: pd.DataFrame,
    dir: str,
    save_path: str,
    onsets_only: bool,
):
    """Writes behaview annotations in a binary format"""
    cam_TTL = np.loadtxt(os.path.join(dir, "cam_TTL.txt"))
    event_names = events["behavior"].unique()

    df_binary = pd.DataFrame(0, index=range(len(cam_TTL)), columns=event_names)

    for name in event_names:
        starts = (
            events.loc[events["behavior"] == name, "behavior"].iloc[::2].index
            + cam_TTL[0]
        )
        stops = (
            events.loc[events["behavior"] == name, "behavior"].iloc[1::2].index
            + cam_TTL[0]
        )

        auxfun_data.save_timestamps_txt(starts, stops, name, save_path, onsets_only)
        auxfun_data.fill_binary_data(df_binary, name, starts, stops, cam_TTL)

    df_binary["camera_timestamp"] = cam_TTL
    df_binary.to_csv(os.path.join(save_path, "events_binary.csv"), index=False)


def read_boris(
    rec_folder: str,
    boris_output: str,
    onsets_only: bool = True,
):
    """Reads and extracts behavior annotations from Boris, writes timestamps of events and a binary DataFrame

    Args:
        rec_folder: path to the folder containing corresponding recording ephys data
        boris_output: path to output of Boris with behavior annotations matching this ephys recording
        onsets_only: if True saves a txt file with only onsets timestamps, otherwise a separate file with offset timestamps is also saved
    """
    try:
        df = pd.read_csv(
            boris_output, skiprows=15, usecols=["Time", "Subject", "Behavior", "Status"]
        )
        cam_TTL = np.loadtxt(os.path.join(rec_folder, "cam_TTL.txt"))
    except FileNotFoundError:
        raise

    pairs = df.groupby(["Behavior", "Subject"]).count().index
    col_names = [f"{behavior}_{subject}" for behavior, subject in pairs]

    df_binary = pd.DataFrame(0, index=range(len(cam_TTL)), columns=col_names)

    save_path = auxiliary.make_dir_save(rec_folder, "events")

    for pair, name in zip(pairs, col_names):
        starts = (
            df.query(
                f"Subject == '{pair[1]}' & Behavior == '{pair[0]}' & Status == 'START'"
            ).loc[:, "Time"]
            + cam_TTL[0]
        )
        stops = (
            df.query(
                f"Subject == '{pair[1]}' & Behavior == '{pair[0]}' & Status == 'STOP'"
            ).loc[:, "Time"]
            + cam_TTL[0]
        )

        auxfun_data.save_timestamps_txt(starts, stops, name, save_path, onsets_only)
        auxfun_data.fill_binary_data(df_binary, name, starts, stops, cam_TTL)

    df_binary["camera_timestamp"] = cam_TTL
    df_binary.to_csv(os.path.join(save_path, "events_binary.csv"), index=False)
