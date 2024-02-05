"""
@author: K. Danielewski
"""
import os
from glob import glob

def make_dir_save(save_path, name) -> str:
    """Auxfun. Checks if dir of joined path exists, if not makes dir, else outputs joined path
    """    
    if not os.path.exists(os.path.join(save_path, name)):
        os.mkdir(os.path.join(save_path, name))
        save_here = os.path.join(save_path, name)
    else:
        save_here = os.path.join(save_path, name)
    return save_here

def check_data_folder(data_folder) -> str:
    """Auxfun to check whether data folder was passed correctly.
    """    
    try:
        if isinstance(data_folder, list):
            pass
        elif isinstance(data_folder, str):
            data_folder = glob(data_folder + "\\*")
    except:
        print(f"Passed data folder should be either string or a list, {type(data_folder)} was passed")
    return data_folder

def prepare_data_structure(event_names: list[str], rec_names: list[str], pre_event: float, post_event: float, bin_size: float) -> dict:
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
    bin_params = {"pre_event": pre_event,
                  "post_event": post_event,
                  "bin_size": bin_size}
    responsive_units = {recording: {event: None for event in event_names} for recording in rec_names}
    
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
        "responsive_units",
        "bin_params",
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
        responsive_units,
        bin_params,
        ]
    
    # Zip into the output structure
    data_obj = dict(zip(output_keys, output_values))

    return data_obj
