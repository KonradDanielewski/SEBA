"""
@author: K. Danielewski
"""
import pickle
import os
from itertools import chain

import numpy as np
import pandas as pd

from seba.utils import auxiliary

def apply_conditions(data_folder: str or list, input_event: str, conditions: list, exclusive=True):
    """Function used to apply conditions to events to take only independent instances of an event or only instances when events happened together
    TODO: Rewrite, this is disgusting

    Args:
        data_folder (str): path to a folder containing all ephys data folders
        input_event (str): name of the event file that conditions should be applied to, without file extension
        conditions (list): list of strings containing raw event names (uses names of columns in events_filled_conditions.csv) 
                           first has to be the same as the input_event
        exclusive (bool, optional): Toggles if output should be for independent instances or instances when events happened together.
    Returns:
        Text file saved into the events subfolder for each recorded animal.
    """
    data_folder = auxiliary.check_data_folder(data_folder)

    for folder in data_folder:
        query_conditions = []
        if exclusive:
            for idx, condition in enumerate(conditions):
                if idx == 0:
                    query_conditions.append(f"{condition} == 1")
                else:
                    query_conditions.append(f"{condition} == 0")
        else:
            for condition in conditions:
                query_conditions.append(f"{condition} == 1")

        query_conditions = " and ".join(query_conditions)
        
        df = pd.read_csv(os.path.join(folder, "events", "events_binary.csv"), index_col="TS")
    
        conditional_event = np.loadtxt(os.path.join(folder, "events", f"{input_event}.txt"))
        condition_met = df.query(query_conditions).index
        output = np.intersect1d(conditional_event, condition_met)
        
        save_path = auxiliary.make_dir_save(folder, "events")
        
        if exclusive :
            np.savetxt(os.path.join(save_path, input_event + "_exclusive.txt"), output, fmt='%1.6f')
        else:
            name = "_".join(conditions[1:])
            np.savetxt(os.path.join(save_path, input_event + "_" + name + "_together.txt"), output, fmt='%1.6f')

def update_data_structure(data_obj: dict, responsive_units: dict, save_output=True, save_path=None):
    """
    Updates data structure stored in ephys_data.pickle

    Args:
        data_obj (dict): output of structurize_data. Stored in ephys_data.pickle
        responsive_units (dict): output of get_responsive_units
        save_output (bool, optional): saves output of the function to a pickle file. Defaults to True.
        save_path (str, optional): path to which the file should be saved. File will be named ephys_data.pickle.
    Returns:
        data_obj (dict): updated input with new key containing significantly responsive neurons. Also overwrites old ephys_data.pickle
    """
    if not data_obj.get("responsive_units"):
        data_obj["responsive_units"] = responsive_units
    else:
        pass
    if data_obj.get("responsive_units") and save_output :
        with open(f"{save_path}\\ephys_data.pickle", "wb") as handle:
            pickle.dump(data_obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return data_obj

def event_filled_preparation(df_binary, name, starts_ser, stops_ser, method):
    """
    Aux function for creating events_filled_conditions
    """    
    if method == "behaview" or method=="boris":
        ones = list(chain.from_iterable(
            [range(start, stop) for start, stop in zip(starts_ser.index, stops_ser.index)]
            ))
        df_binary.loc[ones, name] = 1

def append_event_to_binary(directory: str, event_filepath: str, event_len: int, framerate):
    """Aux function for appending TTL event

    Args:
        directory: directory of the recording to which data is being added
        event_filepath: path to the file which contains timestamps of the event
        event_len: length of the event in second
        framerate: framerate of the video recording
    """
    if event_filepath.endswith(".csv"):
        event = pd.read_csv(event_filepath, header=None)
    if event_filepath.endswith(".tsv") or event_filepath.endswith(".txt"):
        event = pd.read_csv(event_filepath, header=None, sep="\t")
    else:
        print(f"File extensions not handled. '.csv', '.tsv' and '.txt' are supported but {event_filepath.split('.')[1]} was passed")
    
    # Load DF to which data is added
    events_df = pd.read_csv(os.path.join(directory, "events", "events_binary.csv"), index_col="frame_id")

    # Try to load camera frame timestamps from global clock
    try:
        cam_TTL = np.loadtxt(os.path.join(directory, "cam_TTL.txt"))
    except FileNotFoundError:
        raise

    name = os.path.basename(event_filepath).split('.')[0]
    event_starts = np.searchsorted(cam_TTL, event, side='left')
    event_stops = event_starts + (event_len*framerate)
    events_df[name] = 0
    
    # Write ones where event is taking place
    ones = list(chain.from_iterable(
        [range(start, stop) for start, stop in zip(event_starts, event_stops)]
        ))
    events_df.loc[ones, name] = 1

def responsive_neurons2events(data_folder: str | list, data_obj: dict):
    """Assigns 1 to neuron ids in cluster_info_good.csv that were significantly responsive to event

    Args:
        data_folder: path to a folder containing all ephys data folders or list of those folders
        data_obj: output of structurize_data. Stored in ephys_data.pickle
    Returns:
        Overwrites existing cluster_info_good.csv with a new one containing bool columns of events with True assigned to responsive neurons
    """
        
    data_folder = auxiliary.check_data_folder(data_folder)
    responsive_units = data_obj["responsive_units"]

    for folder, rat in zip(data_folder, responsive_units):
        df = pd.read_csv(os.path.join(folder, "cluster_info_good.csv"), index_col="cluster_id")
        col_names = responsive_units[rat].keys()
        for col_name in col_names:
            df[col_name] = np.nan
            if responsive_units[rat][col_name] == None:
                continue
            df.loc[responsive_units[rat][col_name], col_name] = 1
        df.to_csv(os.path.join(folder, "cluster_info_good.csv"))
