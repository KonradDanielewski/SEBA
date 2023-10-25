"""
@author: K. Danielewski
"""
import os
from glob import glob

import numpy as np
import pandas as pd

from seba.data import auxfun_data


def read_bvs(filepath: str, framerate: int, save_path: str):
    """
    Reads BehaActive behavior annotation data from .bvs files

    Args:
        filepath (str): path to the .bvs file
        framerate (int): framerate of the video
        save_path (str): path to the folder containing animal ephys data

    Returns:
        Creates "events" folder in the save_path. Saves events_readout.csv that contains columns with a frame number and behavior. 
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
    
    milis = df.iloc[:, 0:4].sum(axis=1)
    framerate = 1000/framerate
    milis = milis/framerate
    milis = milis.astype(int)

    events = pd.DataFrame(milis).join(behav[1])
    events.columns = ['frame', 'behavior']
    events = events.set_index("frame")
    if not os.path.exists(os.path.join(save_path, "events")):
        os.mkdir(os.path.join(save_path, "events"))
        save_path = os.path.join(save_path, "events")
    else:
        save_path = os.path.join(save_path, "events")
    events.to_csv(os.path.join(save_path, "events_readout.csv"))

def extract_raw_events_TS_BehaView(data_folder: str | list, framerate: int, append_CS=False, CS_filename=None, CS_len=None, padding=False, pad_size=None):
    """
    Divides all annotations to specific events with their timestamps synced to ephys data

    Args:
        data_folder (str): path to a folder containing all ephys data folders
        events_readout (str): path to events_readout.csv file
        camera_TTL (str): path to file containing camera frame timestamps
        framerate (int): framerate of the video
        append_CS (bool): whether to append conditioned stimulus onset timestamps. Defaults to False
        CS_len (int): length of the conditioned stimulus in seconds
        padding (bool): Toggles if event data should be 1-padded around it to make conditions more conservative. Applied to before onset and after offset
        pad_size (float): in seconds, how big the padding should be
    Returns:
        In animal folder creates events subfolder where it saves timestamps of onsets and offsets of events and events_filled_conditions.csv
        that contains per frame bool status of events
    """
    try:
        if isinstance(data_folder, list):
            pass
        elif isinstance(data_folder, str):
            data_folder = glob(data_folder + "\\*")
    except:
        print(f"Passed data folder should be either string or a list, {type(data_folder)} was passed")
    
    if pad_size is not None:
        pad = str(pad_size)
        pad = pad.replace(".", "_")
    
    for folder in data_folder: 
        cam_TS = pd.read_csv(os.path.join(folder, "cam_TTL.txt"), index_col="TS")
        cam_TTL = np.array(cam_TS.index)
        events = pd.read_csv(os.path.join(folder, "events", "events_readout.csv"))
        event_names = events["behavior"].unique()

        df_conditions = pd.DataFrame(0, index=range(len(cam_TTL)), columns=event_names)

        for name in event_names:
            starts = events.query("behavior == @name")[::2]
            stops = events.query("behavior == @name")[1::2]

            starts_ser = pd.Series(cam_TTL, index=range(len(cam_TTL)))
            stops_ser = pd.Series(cam_TTL, index=range(len(cam_TTL)))
            starts_ser = starts_ser.iloc[starts["frame"]]
            stops_ser = stops_ser.iloc[stops["frame"]]

            if len(starts_ser) != len(stops_ser):
                stops_ser[len(stops_ser) + 1] = list(cam_TTL)[-1]

            #Save raw events without any conditions
            if len(starts_ser) > 0:
                np.savetxt(os.path.join(folder, f"events\\{name}_onsets.txt"), starts_ser, fmt='%1.6f')
            if len(stops_ser) > 0:
                np.savetxt(os.path.join(folder, f"events\\{name}_offsets.txt"), stops_ser, fmt='%1.6f')
            auxfun_data.event_filled_preparation(df_conditions=df_conditions, starts_ser=starts_ser, stops_ser=stops_ser,
                                    framerate=framerate, padding=padding, pad_size=pad_size, method="behaview", name=name)

        df_conditions["TS"] = cam_TTL
        
        try:
            if append_CS  and os.path.isfile(os.path.join(folder, CS_filename)):
                auxfun_data.append_event(folder, CS_filename, cam_TTL, CS_len, df_conditions, framerate, padding, pad_size)
        except FileNotFoundError:
                print(f"The file {CS_filename} is not in the events folder. Please move the file to events folder and try again")

        output = df_conditions
        if padding == False:
            output.to_csv(os.path.join(folder, f"events\\events_filled_conditions.csv"), index_label="frame_id")
        else:
            output.to_csv(os.path.join(folder, f"events\\events_filled_conditions{pad}.csv"), index_label="frame_id") 

def read_extract_boris(data_folder: str | list, boris_output: str, camera_TTL: str, append_CS=False, CS_filename=None, CS_len=None, padding=False, pad_size=None):
    """
    Reads and extracts behavior annotations from Boris

    Args:
        data_folder (str): path to a folder containing all ephys data folders
        boris_output (str): path to output of Boris with behavior annotations
        camera_TTL (str): path to file containing camera frame timestamps 
        append_CS (bool, optional): whether to append conditioned stimulus onset timestamps. Defaults to False
        CS_len (int, optional): length of the conditioned stimulus in seconds
        padding (bool): Toggles if event data should be 1-padded around it to make conditions more conservative. Applied to before onset and after offset
        pad_size (float): in seconds, how big the padding should be
    """
    try:
        if isinstance(data_folder, list):
            pass
        elif isinstance(data_folder, str):
            data_folder = glob(data_folder + "\\*")
    except:
        print(f"Passed data folder should be either string or a list, {type(data_folder)} was passed")

    if pad_size is not None:
        pad = str(pad_size)
        pad = pad.replace(".", "_")

    for folder in data_folder:
        df = pd.read_csv(os.path.join(folder, boris_output), skiprows = 15, usecols=["Time", "Total length", "FPS", 
                                                                            "Subject", "Behavior", "Status"])

        subjects = df["Subject"].unique()
        behaviors = df["Behavior"].unique()
        framerate = int(df["FPS"].unique())
        vid_len = range(int(df["Total length"].unique() * framerate))
        cam_TS = pd.read_csv(os.path.join(folder, camera_TTL), usecols=["TS"])
        cam_TTL = cam_TS["TS"]
        cols = []

        for subject in subjects:
            for behavior in behaviors:
                cols.append(f"{subject}_{behavior}")

        df_conditions = pd.DataFrame(0, index = vid_len, columns = cols)
        events_readout = pd.DataFrame(0, index = vid_len, columns = cols)

        if not os.path.exists(os.path.join(folder, "events")):
            os.mkdir(os.path.join(folder, "events"))
            save_path = os.path.join(folder, "events")
        else:
            save_path = os.path.join(folder, "events")

        for subject in subjects:
            for behavior in behaviors:
                starts_ser = df.query(f"Subject == '{subject}' & Behavior == '{behavior}' & Status == 'START'").copy()
                stops_ser = df.query(f"Subject == '{subject}' & Behavior == '{behavior}' & Status == 'STOP'").copy()

                starts_ser =  cam_TS.iloc[(starts_ser["Time"] * framerate).astype(int)]
                stops_ser = cam_TS.iloc[(stops_ser["Time"] * framerate).astype(int)]

                if len(starts_ser) != len(stops_ser):
                    stops_ser[len(stops_ser) + 1] = cam_TTL.iloc[-1]

                if len(starts_ser) > 0:
                    np.savetxt(os.path.join(save_path, f"{subject}_{behavior}_onsets.txt"), starts_ser, fmt='%1.6f')
                if len(stops_ser) > 0:
                    np.savetxt(os.path.join(save_path, f"{subject}_{behavior}_offsets.txt"), stops_ser, fmt='%1.6f')

                auxfun_data.event_filled_preparation(df_conditions=df_conditions, starts_ser=starts_ser, stops_ser=stops_ser, 
                                         framerate=framerate, method="boris", padding=padding, 
                                         pad_size=pad_size, subject=subject, behavior=behavior)

        df_conditions["TS"] = cam_TTL
        try:
            if append_CS  and os.path.isfile(os.path.join(folder, CS_filename)):     
                auxfun_data.append_event(folder, CS_filename, cam_TTL, CS_len, df_conditions, framerate, padding, pad_size)
        except FileNotFoundError:
                print("The file CS.txt is not in the events folder. Please move the file to events folder and try again")

        # drop columns filled with zeros
        output = df_conditions.loc[:, (df_conditions != 0).any(axis=0)]
        output_raw = events_readout.loc[:, (df_conditions != 0).any(axis=0)]
        if padding == False:
            output.to_csv(os.path.join(save_path, f"events_filled_conditions.csv"), index_label="frame_id")
        else:
            output.to_csv(os.path.join(save_path, f"events_filled_conditions{pad}.csv"), index_label="frame_id")
        output_raw.to_csv(os.path.join(save_path, "events_readout.csv"), index_label="frame_id") 