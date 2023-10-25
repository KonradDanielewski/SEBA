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

def apply_conditions(data_folder: str | list, input_event: str, conditions: list, pad_size=None, exclusive=True):
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

def structurize_data(data_folder: str | list, pad_size=None, pre_event=1.0, post_event=3.0, bin_size=0.01, calculate_responsive=True, p_bound=None, save_output=True, save_path=None):
    """
    Creates a datastructure to be used for plotting and further analysis, saves it to pickle to not repeat computations.

    Args:
        data_folder (str): path to a folder containing all ephys data folders
        pre_event (float, optional): pre event time in seconds. Defaults to 1.0.
        post_event (float, optional): post event time in seconds. Defaults to 3.0.
        bin_size (float, optional): size of the bin used for firing rate calculation in seconds. Defaults to 0.01.
        save_output (bool, optional): saves output of the function to a pickle file. Defaults to True.
        save_path (str, optional): path to which the file should be saved. File will be named ephys_data.pickle.

    Returns:
        dict: contains fr(firing rate) and means of fr and z-score and its' means. Used as data structure for all following analyses
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
    events = [os.path.splitext(filename)[0] for filename in os.listdir(os.path.join(data_folder[0],
                                                                       f"events\\conditions{pad}" if pad_size is not None else "events")) 
                                                                       if filename.endswith(".txt")]
    rats = [i.split("\\")[-1] for i in data_folder]

    mean_zscored_events_per_rat = {key: None for key in events}
    mean_zscored_events_all_rats = {key: None for key in events}
    mean_fr_events_per_rat = {key: None for key in events}
    mean_fr_events_all_rats = {key: None for key in events}

    all_zscored_events_per_rat = {key: None for key in events}
    all_fr_events_per_rat = {key: None for key in events}
    
    units_ids = {key: None for key in rats}

    centered_spike_timestamps = {key: None for key in events}
    spike_timestamps = {key: None for key in rats}

    for event in events:
        #Get all mean z-scores to combine into one df
        
        all_zscs = {key: None for key in rats}
        all_frs = {key: None for key in rats}
        centered_spike_timestamps[event] = {key: None for key in rats}
        
        mean_zscs = []
        mean_frs = []

        for folder, rat in zip(data_folder, rats):
            spks_dir = folder
            which_to_use = f"events\\conditions{pad}" if pad_size is not None else "events"
            condition = os.path.join(folder, which_to_use, f"{event}.txt")
            
            #Take only spikes timestamps
            spikes_ts, unit_ids = ephys_io.read_spikes(spks_dir)
            centered_ts = ephys.calc_rasters(spikes_ts, condition, pre_event, post_event)
            sampling_out = 1/bin_size
            
            if units_ids.get(rat) is None:
                units_ids[rat] = unit_ids
            else:
                pass

            if spike_timestamps.get(rat) is None:
                spike_timestamps[rat] = spikes_ts
            else:
                pass

            #Take only all_fr and mean_fr
            all_fr, mean_fr = ephys.fr_events_binless(centered_ts, 0.1, 4, sampling_out, pre_event, post_event)[0:2]
            
            #Take only all_zsc and mean_zsc
            all_zsc, mean_zsc = ephys.zscore_events(all_fr, bin_size, pre_event, post_event)[0:2]
            rat = folder.split("\\")[-1]
            cols = [[rat], np.around(np.arange(-abs(pre_event), post_event, bin_size), 2)]
            index = pd.MultiIndex.from_product(cols, names=["rat", "time"])
            
            all_zscs[rat] = all_zsc
            mean_zsc = pd.DataFrame(mean_zsc, index=unit_ids, columns=index)
            mean_zscs.append(mean_zsc)

            all_frs[rat] = np.array(all_fr)
            mean_fr = pd.DataFrame(mean_fr, index=unit_ids, columns=index)
            mean_frs.append(mean_fr)

            centered_spike_timestamps[event][rat] = centered_ts
        
        #per rat dataframes with level 0 containing rat name
        per_rat_zscore = pd.concat(mean_zscs, axis=1)
        mean_zscored_events_per_rat[event] = per_rat_zscore
        per_rat_fr = pd.concat(mean_frs, axis=1)
        mean_fr_events_per_rat[event] = per_rat_fr
        
        #all rats combined, no unit id kept (for heatmaps, correlations)
        for i in range(len(mean_zscs)):
            mean_zscs[i] = mean_zscs[i].droplevel(0, axis=1)
        mean_zscored_rats = pd.concat(mean_zscs, axis=0)
        mean_zscored_rats = mean_zscored_rats.reset_index(drop=True)
        mean_zscored_events_all_rats[event] = mean_zscored_rats

        for i in range(len(mean_frs)):
            mean_frs[i] = mean_frs[i].droplevel(0, axis=1)
        mean_fr_rats = pd.concat(mean_frs, axis=0)
        mean_fr_rats = mean_fr_rats.reset_index(drop=True)
        mean_fr_events_all_rats[event] = mean_fr_rats

        all_fr_events_per_rat[event] = all_frs
        all_zscored_events_per_rat[event] = all_zscs
    
    bin_params = {"pre_event": pre_event,
                  "post_event": post_event,
                  "bin_size": bin_size,}

    output_keys = ["all_fr_events_per_rat", "all_zscored_events_per_rat", "mean_fr_events_all_rats", "mean_fr_events_per_rat", "mean_zscored_events_all_rats", "mean_zscored_events_per_rat", "centered_spike_timestamps", "spike_timestamps", "units_ids", "bin_params"]
    output_values = [all_fr_events_per_rat, all_zscored_events_per_rat, mean_fr_events_all_rats, mean_fr_events_per_rat, mean_zscored_events_all_rats, mean_zscored_events_per_rat, centered_spike_timestamps, spike_timestamps, units_ids, bin_params]
    data_obj = dict(zip(output_keys, output_values))
    if save_output:
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        with open(f"{save_path}\\ephys_data_{pad}.pickle" if pad_size is not None else f"{save_path}\\ephys_data.pickle", "wb") as handle:
            pickle.dump(data_obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if calculate_responsive:
        responsive_units = get_responsive_units(data_folder, data_obj, p_bound, pad_size)
        data_obj_responsive = auxfun_data.update_data_structure(data_obj, responsive_units, save_output, save_path, pad_size)
        return data_obj_responsive
    else:
        return data_obj

def get_responsive_units(data_folder: str | list, data_obj: dict, p_bound=0.05, pad_size=None):
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