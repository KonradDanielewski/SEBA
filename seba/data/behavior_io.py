"""
@author: K. Danielewski
"""
import os

import numpy as np
import pandas as pd

from seba.utils import (
    auxiliary,
    auxfun_data
)

def read_bvs(filepath: str, save_path: str, onsets_only: bool = True):
    """Reads BehaActive behavior annotation data from .bvs files and saves timestamps of events and binary DataFrame

    Args:
        filepath: path to the .bvs file
        framerate: framerate of the video
        save_path: path to the folder containing corresponding recording ephys data
        onsets_only: if True saves a txt file with only onsets timestamps, otherwise a separate file with offset timestamps is also saved
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise
    df = pd.read_csv(filepath)
    behav = df["[BehaViewSeq]"].str.split(" - ", expand=True)
    df = behav[0].str.split(":", expand=True)
    df = df.merge(behav[1], left_index=True, right_index=True)
    
    df = df.iloc[:,0:4].astype(int)
    
    df.iloc[:, 0] *= 3600000
    df.iloc[:, 1] *= 60000
    df.iloc[:, 2] *= 1000
    df.iloc[:, 3] *= 10
    
    milis = df.iloc[:, 0:4].sum(axis=1)/1000

    events = pd.DataFrame(milis).join(behav[1])
    events.columns = ['timestamp', 'behavior']
    events = events.set_index("timestamp")
    
    save_events = auxiliary.make_dir_save(save_path, "events")

    behaview_events(events, save_path, save_events, onsets_only)

def behaview_events(events: pd.DataFrame, dir: str, save_events: str, onsets_only: bool):
    """Writes behaview annotations in a binary format
    """
    cam_TTL = np.loadtxt(os.path.join(dir, "cam_TTL.txt"))
    event_names = events["behavior"].unique()
    df_binary = pd.DataFrame(0, index=range(len(cam_TTL)), columns=event_names)

    for name in event_names:
        starts = events.loc[events["behavior"] == name, "behavior"].iloc[::2].index + cam_TTL[0]
        stops = events.loc[events["behavior"] == name, "behavior"].iloc[1::2].index + cam_TTL[0]

        #If last event has no stop add stop as the last index of the recording
        if len(starts) - len(stops) == 1:
            stops[len(starts)] = list(cam_TTL)[-1]

        #Save event timestamps
        if len(starts) > 0:
            np.savetxt(os.path.join(save_events, f"{name}_onsets.txt"), starts, fmt='%1.6f')
        if len(stops) > 0 and not onsets_only:
            np.savetxt(os.path.join(save_events, f"{name}_offsets.txt"), stops, fmt='%1.6f')
        
        starts = np.searchsorted(cam_TTL, starts)
        stops = np.searchsorted(cam_TTL, stops)
        
        auxfun_data.event_filled_preparation(df_binary, name, starts, stops)

    df_binary["camera_timestamp"] = cam_TTL
    
    df_binary.to_csv(os.path.join(save_events, "events_binary.csv"), index=False)

def read_boris(directory: str, boris_output: str, onsets_only: bool = True):
    """Reads and extracts behavior annotations from Boris, writes timestamps of events and a binary DataFrame

    Args:
        directory: path to a folder containing all ephys data folders
        boris_output: path to output of Boris with behavior annotations matching this ephys recording
        onsets_only: if True saves a txt file with only onsets timestamps, otherwise a separate file with offset timestamps is also saved
    """
    try:
        df = pd.read_csv(boris_output, skiprows = 15, usecols=["Time", "Subject", "Behavior", "Status"])
        cam_TTL = np.loadtxt(os.path.join(directory, "cam_TTL.txt"))
    except FileNotFoundError:
        raise
 
    pairs = [(behavior, subject) for behavior, subject in df.groupby(["Behavior", "Subject"]).count().index]
    col_names = [f"{behavior}_{subject}" for behavior, subject in pairs]

    df_binary = pd.DataFrame(0, index=range(len(cam_TTL)), columns=col_names)

    save_path = auxiliary.make_dir_save(directory, "events")

    for pair, name in zip(pairs, col_names):
        starts = df.query(f"Subject == '{pair[1]}' & Behavior == '{pair[0]}' & Status == 'START'").loc[:, "Time"] + cam_TTL[0]
        stops = df.query(f"Subject == '{pair[1]}' & Behavior == '{pair[0]}' & Status == 'STOP'").loc[:, "Time"] + cam_TTL[0]

        # Save text files with timestamps
        if len(starts) > 0:
            np.savetxt(os.path.join(save_path, f"{name}_onsets.txt"), starts, fmt='%1.6f')
        if len(stops) > 0 and not onsets_only:
            np.savetxt(os.path.join(save_path, f"{name}_offsets.txt"), stops, fmt='%1.6f')
        
        # If last event has no stop add stop as the last index of the recording
        if len(starts) - len(stops) == 1:
            stops[len(starts)] = cam_TTL[-1]

        # Match closest timestamps between camera and event and get indices
        starts = np.searchsorted(cam_TTL, starts)
        stops = np.searchsorted(cam_TTL, stops)

        auxfun_data.event_filled_preparation(df_binary, name, starts, stops)

    df_binary["camera_timestamp"] = cam_TTL
    df_binary.to_csv(os.path.join(save_path, "events_binary.csv"), index=False)