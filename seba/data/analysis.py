"""
@author: K. Danielewski
"""
import os
import pickle
from glob import glob


import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from seba.ephys import (
    ephys,
    ephys_io,
)
from seba.data import auxfun_data

def apply_conditions(data_folder: str or list, input_event: str, conditions: list, pad_size=None, exclusive=True):
    """
    Function used to apply conditions to events to take only independent instances of an event or only instances when events happened together

    Args:
        data_folder (str): path to a folder containing all ephys data folders
        input_event (str): name of the event file that conditions should be applied to, without file extension
        conditions (list): list of strings containing raw event names (uses names of columns in events_filled_conditions.csv) 
                           first has to be the same as the input_event
        pad_size (float): in seconds, specifies which evet_filled_conditions.csv should be used. Default None, no padding
        exclusive (bool, optional): Toggles if output should be for independent instances or instances when events happened together. Defaults to True.
    Returns:
        Text file saved into the events subfolder for each recorded animal.
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
        query_conditions = []
        if exclusive :
            for idx, condition in enumerate(conditions):
                if idx == 0:
                    query_conditions.append(f"{condition} == 1")
                else:
                    query_conditions.append(f"{condition} == 0")
        else:
            for condition in conditions:
                query_conditions.append(f"{condition} == 1")

        query_conditions = " and ".join(query_conditions)
        
        if pad_size == None:
            df = pd.read_csv(os.path.join(folder, "events", "events_filled_conditions.csv"), index_col="TS")
        else:
            df = pd.read_csv(os.path.join(folder, "events", f"events_filled_conditions{pad}.csv"), index_col="TS")
        conditional_event = np.loadtxt(os.path.join(folder, "events", f"{input_event}.txt"))
        condition_met = df.query(query_conditions).index
        output = np.intersect1d(conditional_event, condition_met)
        
        if not os.path.exists(os.path.join(folder, f"events\\conditions{pad}" if pad_size is not None else "events")):
            os.mkdir(os.path.join(folder, f"events\\conditions{pad}" if pad_size is not None else "events"))
            save_path = os.path.join(folder, f"events\\conditions{pad}" if pad_size is not None else "events")
        else:
            save_path = os.path.join(folder, f"events\\conditions{pad}" if pad_size is not None else "events")
        
        if exclusive :
            np.savetxt(os.path.join(save_path, input_event + "_exclusive.txt"), output, fmt='%1.6f')
        else:
            name = "_".join(conditions[1:])
            np.savetxt(os.path.join(save_path, input_event + "_" + name + "_together.txt"), output, fmt='%1.6f')

def structurize_data(
    data_folder: str or list,
    sigma_sec: float = 0.1, 
    pre_event: float = 1.0,
    post_event: float = 3.0,
    bin_size: float = 0.01,
    calculate_responsive: bool = True,
    p_bound: float | None = None,
    save_output: bool = True,
    save_path: str | None = None,
    event_names: list[str] | None = None
    ):
    """Fills in a datastructure to be used for plotting and further analysis, saves it to pickle to not repeat computations.

    Args:
        data_folder: path to a folder containing all ephys data folders
        sigma_sec: width of the Gaussian kernel in seconds. Passed to fr_events_binless
        pre_event: pre event time in seconds. Defaults to 1.0.
        post_event: post event time in seconds. Defaults to 3.0.
        bin_size: resampling factor used for firing rate calculation in seconds. Defaults to 0.01.
        calculate_responsive: Whether to check neuron responsivity. Defaults to True.
        p_bound: If not provided and calculate_responsive is True, set to 0.01. Defaults to None.
        save_output: saves output data structure to a pickle file. Defaults to True.
        save_path: path to which the file should be saved if save_output is True. File will be named ephys_data.pickle. Defaults to None.
        event_names: list of all possible events if different then the ones in the first recording directory. Defaults to None.

    Returns:
        _description_
    """

    try:
        if isinstance(data_folder, list):
            pass
        elif isinstance(data_folder, str):
            data_folder = glob(data_folder + "\\*")
    except:
        print(f"Passed data folder should be either string or a list, {type(data_folder)} was passed")
    
    if isinstance(event_names, list):
        pass
    else:
        event_names = [os.path.basename(filename).split(".")[0] for filename in glob(os.path.join(f"{data_folder[0]}", "events", "*.txt"))]
    
    rec_names = [os.path.basename(i) for i in data_folder]

    data_obj = prepare_data_structure(event_names, rec_names, pre_event, post_event, bin_size)
    
    if calculate_responsive and p_bound == None:
        p_bound = 0.01

    sampling_out = 1/bin_size

    for event in event_names:
        #Get all mean z-scores to combine into one df
        mean_zscs = []
        mean_frs = []

        for spks_dir, rec in zip(data_folder, rec_names):
            condition = os.path.join(spks_dir, "events", f"{event}.txt")
            
            if len(np.loadtxt(condition)) == 0:
                data_obj["spike_timestamps"][rec] = spikes_ts
                data_obj["unit_ids"][rec] = unit_ids
                data_obj["all_fr_events_per_rat"][event][rec] = None
                data_obj["all_zscored_events_per_rat"][event][rec] = None
                data_obj["centered_spike_timestamps"][event][rec] = None
                
                cols = [[rec], np.around(bin_edges, 2)]
                index = pd.MultiIndex.from_product(cols, names=["rat", "time"])
                mean_zsc = pd.DataFrame(np.nan, index=unit_ids, columns=index)
                mean_zscs.append(mean_zsc)            
                mean_fr = pd.DataFrame(np.nan, index=unit_ids, columns=index)
                mean_frs.append(mean_fr)
            else:
                # Take only spikes timestamps
                spikes_ts, unit_ids = ephys_io.read_spikes(spks_dir)
                centered_ts = ephys.calc_rasters(spikes_ts, condition, pre_event, post_event)

                # Write spike times and unit IDs
                data_obj["spike_timestamps"][rec] = spikes_ts
                data_obj["unit_ids"][rec] = unit_ids

                # Take only all_fr, all_zsc and mean_fr, mean_zsc and bin_edges
                all_fr, mean_fr = ephys.fr_events_binless(centered_ts, sigma_sec, sampling_out, pre_event, post_event)[0:2]
                all_zsc, mean_zsc, sem, bin_edges = ephys.zscore_events(all_fr, bin_size, pre_event, post_event)   

                # Prepare index and columns
                cols = [[rec], np.around(bin_edges[:-1], 2)]
                index = pd.MultiIndex.from_product(cols, names=["rat", "time"])
                
                mean_zsc = pd.DataFrame(mean_zsc, index=unit_ids, columns=index)
                mean_zscs.append(mean_zsc)            
                mean_fr = pd.DataFrame(mean_fr, index=unit_ids, columns=index)
                mean_frs.append(mean_fr)

                # Write all firing rates per rat, all zscored per rat and center spike timestamps
                data_obj["all_fr_events_per_rat"][event][rec] = np.array(all_fr)
                data_obj["all_zscored_events_per_rat"][event][rec] = all_zsc
                data_obj["centered_spike_timestamps"][event][rec] = centered_ts
        
        #per rat dataframes with level 0 containing rat name
        per_rat_fr = pd.concat(mean_frs, axis=1)
        per_rat_zscore = pd.concat(mean_zscs, axis=1)

        for i in range(len(mean_zscs)):
            mean_zscs[i] = mean_zscs[i].droplevel(0, axis=1)

        for i in range(len(mean_frs)):
            mean_frs[i] = mean_frs[i].droplevel(0, axis=1)
        
        # Concat mean z-scores and firing rates into one DataFrame per event
        mean_zscored_rats = pd.concat(mean_zscs, axis=0).reset_index(drop=True)
        mean_fr_rats = pd.concat(mean_frs, axis=0).reset_index(drop=True)
        
        # Write means for all and per rat into the data 
        data_obj["mean_zscored_events_all_rats"][event] = mean_zscored_rats
        data_obj["mean_fr_events_all_rats"][event] = mean_fr_rats
        data_obj["mean_fr_events_per_rat"][event] = per_rat_fr
        data_obj["mean_zscored_events_per_rat"][event] = per_rat_zscore

    #TODO: make a safe saving function that checks dir and saves files to avoid this code being reduntantly written across codebase
    if save_output:
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        with open(f"{save_path}\\ephys_data.pickle", "wb") as handle:
            pickle.dump(data_obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if calculate_responsive:
        responsive_units = get_responsive_units(data_folder, data_obj, p_bound)
        data_obj_responsive = auxfun_data.update_data_structure(data_obj, responsive_units, save_output, save_path)
        return data_obj_responsive
    else:
        return data_obj
    
def prepare_data_structure(event_names: list[str], rec_names: list[str], pre_event: float, post_event: float, bin_size: float):
    """Auxfun preparing data structure

    Args:
        event_names: names of all possible events
        rec_names: names of all recordings.
        pre_event (float, optional): pre event time in seconds. Defaults to 1.0.
        post_event (float, optional): post event time in seconds. Defaults to 3.0.
        bin_size (float, optional): size of the bin used for firing rate calculation in seconds. Defaults to 0.01.
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


def get_responsive_units(data_folder: str or list, data_obj: dict, p_bound=0.05, pad_size=None):
    """
    Perform Wilcoxon signed-rank test to check which neurons responded significantly to specific condition/event

    Args:
        data_folder (str): path to a folder containing all ephys data folders
        data_obj (dict): output of structurize_data. Stored in ephys_data.pickle
        p_bound (float): p-value below which neurons are considered as significantly responsive
        pad_size: int if specified. Padding for event length used in creating the data structure
    Returns:
        responsive_units (dict): dict of responsive units that can be appended to ephys data structure
    """
    pre_event = data_obj["bin_params"]["pre_event"]
    post_event = data_obj["bin_params"]["post_event"]
    bin_size = data_obj["bin_params"]["bin_size"]

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
    rats = [i.split("\\")[-1] for i in data_folder]
    conditions = [os.path.splitext(filename)[0] for filename in os.listdir(os.path.join(data_folder[0], 
                                                                                        f"events\\conditions{pad}" if pad_size is not None else "events")) 
                                                                                        if filename.endswith(".txt")]
    
    responsive_units = auxfun_data.responsive_units_wilcoxon(data_obj, conditions, rats, pre_event, post_event, bin_size, p_bound)

    return responsive_units