"""
@author: K. Danielewski
"""
import pickle
import os
from glob import glob
from typing import List

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

def update_data_structure(data_obj: dict, responsive_units: dict, save_output=True, save_path=None, pad_size=None):
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
    if pad_size is not None:
        pad = str(pad_size)
        pad = pad.replace(".", "_")    
    if not data_obj.get("responsive_units"):
        data_obj["responsive_units"] = responsive_units
    else:
        pass
    if data_obj.get("responsive_units") and save_output :
        with open(f"{save_path}\\ephys_data_{pad}.pickle" if pad_size is not None else f"{save_path}\\ephys_data.pickle", "wb") as handle:
            pickle.dump(data_obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return data_obj

def event_filled_preparation(df_conditions, starts_ser, stops_ser, framerate, method, padding, pad_size=None, name=None, subject=None, behavior=None):
    """
    Aux function for creating events_filled_conditions
    """    
    if method == "behaview":
        if padding :
            pad = int(np.around(pad_size*framerate, 0))
            for start, stop in zip(list(starts_ser.index), list(stops_ser.index)):                
                if start < pad:
                    df_conditions[name][0:stop+pad] = 1
                elif stop+pad > len(df_conditions):
                    df_conditions[name][start-pad:len(df_conditions)] = 1
                else:
                    df_conditions[name][start-pad:stop+pad] = 1
            if len(starts_ser) > len(stops_ser):
                    df_conditions[name][starts_ser.index[-1]-pad:len(df_conditions)] = 1
        else:
            for start, stop in zip(list(starts_ser.index), list(stops_ser.index)):
                    df_conditions[name][start:stop] = 1
            if len(starts_ser) > len(stops_ser):
                df_conditions[name][starts_ser.index[-1]:len(df_conditions)] = 1
                    

    if method == "boris":
        if padding :
            pad = int(np.around(pad_size*framerate, 0))
            for start, stop in zip(list(starts_ser.index), list(stops_ser.index)):
                if start < pad:
                    df_conditions[f"{behavior}_{subject}"][0:stop+pad] = 1
                if stop+pad > len(df_conditions):
                    df_conditions[f"{behavior}_{subject}"][start-pad:len(df_conditions)] = 1
                else:
                    df_conditions[f"{behavior}_{subject}"][start-pad:stop+pad] = 1
            if len(starts_ser) > len(stops_ser):
                    df_conditions[name][starts_ser.index[-1]-pad:len(df_conditions)] = 1
        else:
            for start, stop in zip(list(starts_ser.index), list(stops_ser.index)):
                df_conditions[f"{behavior}_{subject}"][start:stop] = 1
            if len(starts_ser) > len(stops_ser):
                        df_conditions[name][starts_ser.index[-1]:len(df_conditions)] = 1

def append_event(folder, CS_filename, cam_TTL, CS_len, df_conditions, framerate, padding, pad_size=None):
    """
    Aux function for appending TTL event
    """
    CS = pd.read_csv(os.path.join(folder, CS_filename), header=None)
    CS_starts = np.searchsorted(cam_TTL, CS, side='left')
    CS_stops = np.searchsorted(cam_TTL, CS+CS_len, side='left')
    np.savetxt(os.path.join(folder, "events", "CS_onsets.txt"), df_conditions["TS"].iloc[list(CS_starts.flat)], fmt='%1.6f')
    np.savetxt(os.path.join(folder, "events", "CS_offsets.txt"), df_conditions["TS"].iloc[list(CS_stops.flat)], fmt='%1.6f')
    df_conditions["CS"] = 0
    if padding :
        pad = int(np.around(pad_size*framerate, 0))
        for stimulus in CS_starts:
            df_conditions.loc[stimulus[0]-pad:stimulus[0]+(CS_len*framerate)+pad, "CS"] = 1
    else:
        for stimulus in CS_starts:
            df_conditions.loc[stimulus[0]:stimulus[0]+(CS_len*framerate), "CS"] = 1

def responsive_neurons2events(data_folder: str | list, data_obj: dict):
    """
    Assigns 1 to neuron ids in cluster_info_good.csv that were significantly responsive to event

    Args:
        data_folder (str): path to a folder containing all ephys data folders
        data_obj (dict): output of structurize_data. Stored in ephys_data.pickle
    Returns:
        Overwrites existing cluster_info_good.csv with a new one containing bool columns of events with True assigned to responsive neurons
    """
        
    try:
        if isinstance(data_folder, list):
            pass
        elif isinstance(data_folder, str):
            data_folder = glob(data_folder + "\\*")
    except:
        print(f"Passed data folder should be either string or a list, {type(data_folder)} was passed")
    responsive_units = data_obj["responsive_units"]

    for folder, rat in zip(data_folder, responsive_units):
        df = pd.read_csv(os.path.join(folder, "cluster_info_good.csv"), index_col='cluster_id')
        col_names = responsive_units[rat].keys()
        for col_name in col_names:
            df[col_name] = np.nan
            if responsive_units[rat][col_name] == None:
                continue
            df.loc[responsive_units[rat][col_name], col_name] = 1
        df.to_csv(os.path.join(folder, "cluster_info_good.csv"))

def responsive_units_wilcoxon(data_obj: dict, conditions: List, rats: str, pre_event: float, post_event: float, bin_size: float, p_bound: float) -> dict:
    
    responsive_units = {key: None for key in rats}

    for rat in responsive_units:
        responsive_units[rat] = {key: [] for key in conditions}

    fdr_test = pd.Series(dtype="float64")
    for condition in data_obj["all_fr_events_per_rat"]:
        for rat in data_obj["all_fr_events_per_rat"][condition]:
            if data_obj["centered_spike_timestamps"][condition][rat] == None:
                    continue
            for idx, unit_id in enumerate(data_obj["unit_ids"][rat]):
                n_events = len(data_obj["centered_spike_timestamps"][condition][rat][idx])
                    
                if n_events == 0:
                    continue
                
                col_axis = np.around(np.arange(-abs(pre_event), post_event, bin_size), 2)
                df = pd.DataFrame(data_obj["all_fr_events_per_rat"][condition][rat][idx]).set_axis(col_axis, axis=1)
                event = df.iloc[:, int(pre_event/bin_size):].mean(axis=1)
                
                invalid = sum([1 if len(i) < 10 else 0 for i in data_obj["centered_spike_timestamps"][condition][rat][idx]])
                
                if invalid/n_events > 0.5:
                    continue
                
                baseline = df.iloc[:, :int(pre_event/bin_size)].mean(axis=1)
                #Take p-value only
                if any(event) and len(event) > 10:
                    wilcoxon = stats.wilcoxon(event, baseline, correction=True, zero_method="zsplit", method="approx")[1]
                    if wilcoxon < p_bound:
                        fdr_test.loc[f"{rat}#{condition}#{unit_id}"] = wilcoxon
    
    corrected_p = multipletests(pvals=fdr_test, alpha=0.95, method="fdr_tsbh")[1]
    corr_idx = fdr_test.index
    orig_vs_corr = pd.concat([fdr_test, pd.Series(np.around(corrected_p, 4), index=corr_idx)], keys=["original", "corrected"], axis=1)
    update = orig_vs_corr.query("corrected < 0.05").copy()
    for i in range(len(update)):
        place = update.iloc[i].name
        places = place.split("#")
        responsive_units[places[0]][places[1]].append(int(places[2]))
    return responsive_units