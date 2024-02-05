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

def read_bvs(filepath: str, save_path: str):
    """Reads BehaActive behavior annotation data from .bvs files and saves timestamps of events and binary DataFrame

    Args:
        filepath: path to the .bvs file
        framerate: framerate of the video
        save_path: path to the folder containing corresponding recording ephys data
    """    
    
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

    behaview_events(events, save_path, save_events)

def behaview_events(events: pd.DataFrame, dir: str, save_events: str):
    """Writes behaview annotations in a binary format
    """    

    cam_TTL = np.loadtxt(os.path.join(dir, "cam_TTL.txt"))

    events = pd.read_csv(os.path.join(dir, "events_readout.csv"))
    event_names = events["behavior"].unique()

    df_binary = pd.DataFrame(0, index=range(len(cam_TTL)), columns=event_names)

    for name in event_names:
        starts = events.query("behavior == @name")[::2].index
        stops = events.query("behavior == @name")[1::2].index

        starts = cam_TTL[np.searchsorted(cam_TTL, starts+cam_TTL[0])]
        stops = cam_TTL[np.searchsorted(cam_TTL, stops+cam_TTL[0])]

        #If last event has no stop add stop as the last index of the recording
        if len(starts) - len(stops) == 1:
            stops[len(starts)] = list(cam_TTL)[-1]

        #Save event timestamps
        if len(starts) > 0:
            np.savetxt(os.path.join(save_events, f"{name}_onsets.txt"), starts, fmt='%1.6f')
        if len(stops) > 0:
            np.savetxt(os.path.join(save_events, f"{name}_offsets.txt"), stops, fmt='%1.6f')

        auxfun_data.event_filled_preparation(df_binary, name, starts, stops, method="behaview")

    df_binary["TS"] = cam_TTL
    
    df_binary.to_csv(os.path.join(save_events, "events_binary.csv"), index_label="frame_id")

def read_boris(directory: str, boris_output: str):
    """Reads and extracts behavior annotations from Boris, writes timestamps of events and a binary DataFrame

    Args:
        directory: path to a folder containing all ephys data folders
        boris_output: path to output of Boris with behavior annotations matching this ephys recording
    """
    try:
        df = pd.read_csv(boris_output, skiprows = 15, usecols=["Time", "Subject", "Behavior", "Status"])
        cam_TTL = np.load_txt(os.path.join(directory, "cam_TTL.txt"))
    except FileNotFoundError:
        raise
 
    pairs = [(behavior, subject) for behavior, subject in df.groupby(["Behavior", "Subject"]).count().index]
    col_names = [f"{behavior}_{subject}" for behavior, subject in pairs]

    df_binary = pd.DataFrame(0, index=range(len(cam_TTL)), columns=col_names)

    save_path = auxiliary.make_dir_save(directory, "events")

    for behavior, subject, name in zip(pairs, col_names):
        starts_ser = df.query(f"Subject == '{subject}' & Behavior == '{behavior}' & Status == 'START'").loc[:, "Time"] + cam_TTL[0]
        stops_ser = df.query(f"Subject == '{subject}' & Behavior == '{behavior}' & Status == 'STOP'").loc[:, "Time"] + cam_TTL[0]

        # Match closest timestamps between camera and event and get indices
        starts = cam_TTL[np.searchsorted(cam_TTL, starts)]
        stops = cam_TTL[np.searchsorted(cam_TTL, stops)]

        # Save text files with timestamps
        if len(starts) > 0:
            np.savetxt(os.path.join(save_path, f"{name}_onsets.txt"), starts, fmt='%1.6f')
        if len(stops) > 0:
            np.savetxt(os.path.join(save_path, f"{name}_offsets.txt"), stops, fmt='%1.6f')
        
        #If last event has no stop add stop as the last index of the recording
        if len(starts_ser) - len(stops_ser) == 1:
            stops_ser[len(starts_ser)] = cam_TTL[-1]

        auxfun_data.event_filled_preparation(df_binary, name, starts_ser, stops_ser, method="boris")
    
    df_binary["TS"] = cam_TTL
    df_binary.to_csv(os.path.join(save_path, "events_binary.csv"), index_label="frame_id")