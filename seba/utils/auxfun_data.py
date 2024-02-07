"""
@author: K. Danielewski
"""

import pickle
import os
from itertools import chain

import numpy as np
import pandas as pd

from seba.utils import auxiliary


def prepare_data_structure(
    event_names: list[str],
    rec_names: list[str],
    pre_event: float,
    post_event: float,
    bin_size: float,
) -> dict:
    """Auxfun preparing data structure

    Args:
        event_names: names of all possible events
        rec_names: names of all recordings.
        pre_event: pre event time in seconds. Defaults to 1.0.
        post_event: post event time in seconds. Defaults to 3.0.
        bin_size: Resampling factor used for firing rate calculation. Defaults to 0.01.
    Return:
        Empty dictiorany to be filled with data using structurize_data
    """
    if isinstance(rec_names, list):
        pass
    else:
        print("No recordings found!")

    # Prepare internal stuctures
    all_fr_events_per_rat = {event: {recording: None for recording in rec_names} for event in event_names}
    all_zscored_events_per_rat = {event: {recording: None for recording in rec_names} for event in event_names}
    mean_fr_events_all_rats = {event: None for event in event_names}
    mean_fr_events_per_rat = {event: None for event in event_names}
    mean_zscored_events_all_rats = {event: None for event in event_names}
    mean_zscored_events_per_rat = {event: None for event in event_names}
    centered_spike_timestamps = {event: {recording: None for recording in rec_names} for event in event_names}
    spike_timestamps = {recording: None for recording in rec_names}
    unit_ids = {recording: None for recording in rec_names}
    brain_regions = {recording: None for recording in rec_names}
    bin_params = {
        "pre_event": pre_event,
        "post_event": post_event,
        "bin_size": bin_size,
    }
    responsive_units = {recording: {event: [] for event in event_names} for recording in rec_names}
    events = event_names
    recording_names = rec_names

    # List default keys and variables
    output_keys = [
        "all_fr_events_per_rat",
        "all_zscored_events_per_rat",
        "mean_fr_events_all_rats",
        "mean_fr_events_per_rat",
        "mean_zscored_events_all_rats",
        "mean_zscored_events_per_rat",
        "centered_spike_timestamps",
        "spike_timestamps",
        "unit_ids",
        "brain_regions",
        "responsive_units",
        "bin_params",
        "event_names",
        "recording_names",
    ]
    output_values = [
        all_fr_events_per_rat,
        all_zscored_events_per_rat,
        mean_fr_events_all_rats,
        mean_fr_events_per_rat,
        mean_zscored_events_all_rats,
        mean_zscored_events_per_rat,
        centered_spike_timestamps,
        spike_timestamps,
        unit_ids,
        brain_regions,
        responsive_units,
        bin_params,
        events,
        recording_names,
    ]

    # Zip into the output structure
    data_obj = dict(zip(output_keys, output_values))

    return data_obj


def add_to_data_structure(data_obj: dict | str, new_key: str, save_path: str, data_to_add):
    """Auxfun to add new data to the data structure"""
    if isinstance(data_obj, dict):
        pass
    if isinstance(data_obj, str):
        data_obj = pd.read_pickle(data_obj)

    if data_obj.get(new_key) == None:
        data_obj[new_key] = data_to_add
    else:
        print(f"Key {new_key} already exists in data obj")

    with open(f"{save_path}\\ephys_data.pickle", "wb") as handle:
        pickle.dump(data_obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def apply_conditions(data_folder: str or list, input_event: str, conditions: list, exclusive=True):
    """Function used to apply conditions to events to take only independent instances of an event or only instances when events happened together
    TODO: Consider cleaning up and if it's necessary

    Args:
        data_folder (str): path to a folder containing all ephys data folders
        input_event (str): name of the event file that conditions should be applied to, without file extension
        conditions (list): list of strings containing raw event names (uses names of columns in events_filled_conditions.csv)
                           first has to be the same as the input_event
        exclusive (bool, optional): Toggles if output should be for independent instances or instances when events happened together.
    Returns:
        Text file saved into the events subfolder for each recording.
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

        df = pd.read_csv(os.path.join(folder, "events", "events_binary.csv"), index_col="camera_timestamp")

        conditional_event = np.loadtxt(os.path.join(folder, "events", f"{input_event}.txt"))
        condition_met = df.query(query_conditions).index
        output = np.intersect1d(conditional_event, condition_met)

        save_path = auxiliary.make_dir_save(folder, "events")

        if exclusive:
            np.savetxt(os.path.join(save_path, input_event + "_exclusive.txt"), output, fmt="%1.6f")
        else:
            name = "_".join(conditions[1:])
            np.savetxt(os.path.join(save_path, input_event + "_" + name + "_together.txt"), output, fmt="%1.6f")


def save_timestamps_txt(starts, stops, name, save_path, onsets_only):
    """Auxfun for saving files with timestamps"""
    # Save text files with timestamps
    if len(starts) > 0:
        np.savetxt(os.path.join(save_path, f"{name}_onsets.txt"), starts, fmt="%1.6f")
    if len(stops) > 0 and not onsets_only:
        np.savetxt(os.path.join(save_path, f"{name}_offsets.txt"), stops, fmt="%1.6f")


def event_filled_preparation(df_binary, name, starts, stops, cam_TTL):
    """Aux function for creating events_filled_conditions"""
    # If last event has no stop add stop as the last index of the recording
    if len(starts) - len(stops) == 1:
        stops[len(starts)] = list(cam_TTL)[-1]

    # Match closest timestamps between camera and event and get indices
    starts = np.searchsorted(cam_TTL, starts)
    stops = np.searchsorted(cam_TTL, stops)

    ones = sum([list(range(start, stop)) for start, stop in zip(starts, stops)], [])
    df_binary.loc[ones, name] = 1


def append_event_to_binary(
    directory: str, event_filepath: str, event_len: int, onsets_only: bool = True
):
    """Aux function for appending TTL event

    Args:
        directory: directory of the recording to which data is being added
        event_filepath: path to the file which contains timestamps of the event
        event_len: length of the event in second
        onsets_only: if True saves a txt file with only onsets timestamps, otherwise a separate file with offset timestamps is also saved
    """
    if event_filepath.endswith(".csv"):
        event = pd.read_csv(event_filepath, header=None).values.reshape(-1)
    if event_filepath.endswith(".tsv") or event_filepath.endswith(".txt"):
        event = pd.read_csv(event_filepath, header=None, sep="\t").values.reshape(-1)
    else:
        print(f"File extensions not handled. '.csv', '.tsv' and '.txt' are supported but {os.path.basename(event_filepath).split('.')[-1]} was passed")

    # Load DF to which data is added
    events_df = pd.read_csv(os.path.join(directory, "events", "events_binary.csv"))

    # Try to load camera frame timestamps from global clock
    try:
        cam_TTL = np.loadtxt(os.path.join(directory, "cam_TTL.txt"))
    except FileNotFoundError:
        raise

    name = os.path.basename(event_filepath).split(".")[0].lower()

    np.savetxt(os.path.join(directory, "events", f"{name}_onsets.txt"), event, fmt="%1.6f")
    if not onsets_only:
        offsets = cam_TTL(np.searchsorted(cam_TTL, event + event_len))
        np.savetxt(os.path.join(directory, "events", f"{name}_onsets.txt"), offsets, fmt="%1.6f")

    event_starts = np.searchsorted(cam_TTL, event, side="left")
    event_stops = np.searchsorted(cam_TTL, event + event_len)
    events_df[name] = 0

    # Write ones where event is taking place
    ones = sum([list(range(start, stop)) for start, stop in zip(event_starts, event_stops)], [])
    events_df.loc[ones, name] = 1
    events_df = events_df.sort_index(axis=1)
    events_df.to_csv(os.path.join(directory, "events", "events_binary.csv"), index=False)


def responsive_neurons2events(data_folder: str | list, data_obj: dict):
    """Assigns 1 to neuron ids in cluster_info_good.csv that were significantly responsive to event

    Args:
        data_folder: path to a folder containing all ephys data folders or list of those folders
        data_obj: output of structurize_data. Stored in ephys_data.pickle
    Returns:
        Overwrites existing cluster_info_good.csv with a new one containing bool columns of events with True assigned to responsive neurons
    """
    data_folder = auxiliary.check_data_folder(data_folder)
    rec_names = data_obj["recording_names"]
    events = data_obj["event_names"]

    for folder, rec_name in zip(data_folder, rec_names):
        df = pd.read_csv(os.path.join(folder, "cluster_info_good.csv"), index_col="cluster_id")
        for event in events:
            df[event] = np.nan
            if data_obj["responsive_units"][rec_name][event] == None:
                continue
            df.loc[data_obj["responsive_units"][rec_name][event], event] = 1
        df.to_csv(os.path.join(folder, "cluster_info_good.csv"))


def add_brain_regions(data_folder: str | list, data_obj: dict):
    """Auxfun adds brain regions to data_obj. Each is assigned to it's unit_id"""
    data_folder = auxiliary.check_data_folder(data_folder)
    rec_names = data_obj["recording_names"]

    for folder, rec_name in zip(data_folder, rec_names):
        df = pd.read_csv(os.path.join(folder, "cluster_info_good.csv"), index_col="cluster_id")

        data_obj["brain_regions"][rec_name] = pd.Series(df.loc[:, "Structure"])
