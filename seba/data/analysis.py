"""
@author: K. Danielewski
"""
import os
import pickle
from glob import glob

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels import multipletests

from seba.data import ephys
from seba.plotting import plotting_funcs
from seba.utils import (
    auxfun_data,
    auxfun_ephys,
    auxiliary,
)

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
    ) -> dict:
    """Fills in a datastructure to be used for plotting and further analysis, saves it to pickle to not repeat computations.

    Args:
        data_folder: path to a folder containing all ephys data folders
        sigma_sec: width of the Gaussian kernel in seconds. Passed to fr_events_binless
        pre_event: pre event time in seconds.
        post_event: post event time in seconds.
        bin_size: resampling factor used for firing rate calculation in seconds.
        calculate_responsive: Whether to check neuron responsivity.
        p_bound: If not provided and calculate_responsive is True, set to 0.01.
        save_output: saves output data structure to a pickle file.
        save_path: path to which the file should be saved if save_output is True. File will be named ephys_data.pickle.
        event_names: list of all possible events if different then the ones in the first recording directory. Should be provided if 

    Returns:
        data_obj: data structure in form of a dictionary. If save_path specified, then saved as a pickle file.
    """

    data_folder = auxiliary.check_data_folder(data_folder)
    
    if isinstance(event_names, list):
        pass
    else:
        event_names = [os.path.basename(filename).split(".")[0] for filename in glob(os.path.join(f"{data_folder[0]}", "events", "*.txt"))]
    
    rec_names = [os.path.basename(i) for i in data_folder]

    data_obj = auxfun_data.prepare_data_structure(event_names, rec_names, pre_event, post_event, bin_size)
    
    if calculate_responsive and p_bound == None:
        p_bound = 0.01

    sampling_out = 1/bin_size

    for event in event_names:
        #Get all mean z-scores to combine into one df
        mean_zscs = []
        mean_frs = []

        for spks_dir, rec in zip(data_folder, rec_names):
            condition = os.path.join(spks_dir, "events", f"{event}.txt")
            try:
                file_not_there = np.loadtxt(condition)
            except FileNotFoundError:
                file_not_there = True
            # TODO: think about this. Data is already filled with None
            if file_not_there:
                #Fill data with None
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
                spikes_ts, unit_ids = auxfun_ephys.read_spikes(spks_dir)
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

    try: 
        if calculate_responsive:
            data_obj["responsive_units"] = responsive_units_wilcoxon(data_obj, event_names, rec_names, pre_event, post_event, bin_size, p_bound)
            return data_obj
        else:
            return data_obj
    finally:
        if save_output:
            if not os.path.exists(save_path):
                os.mkdir(save_path)
        with open(f"{save_path}\\ephys_data.pickle", "wb") as handle:
            pickle.dump(data_obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def responsive_units_wilcoxon(
    data_obj: dict,
    events: list[str],
    rec_names: list[str],
    pre_event: float,
    post_event: float,
    bin_size: float,
    p_bound: float,
    spike_threshold: int,
    ) -> dict:
    """Performs Wilcoxon rank test to establish whether a neuron is responsive to a specific event.

    Args:
        data_obj: output of structurize_data. Stored in ephys_data.pickle
        events: names of events
        rec_names: names of recordings
        pre_event: pre event time in seconds.
        post_event: post event time in seconds.
        bin_size: resampling factor used for firing rate calculation in seconds.
        p_bound: p value to be used in filtering rank test output
        spike_threshold: specifies minimum number of spikes that should happen per event instance to be considered in rank test. 
            if more than half of the events have less than that number of spikes this neuron is discarded from the rank test.

    Returns:
        Dictionary of responsive units per animal per event.
    """    
    
    responsive_units = {key: {event for event in events} for key in rec_names}

    fdr_test = pd.Series(dtype="float32")
    for event in data_obj["all_fr_events_per_rat"]:
        for rec_name in data_obj["all_fr_events_per_rat"][event]:
            if data_obj["centered_spike_timestamps"][event][rec_name] == None:
                    continue
            for idx, unit_id in enumerate(data_obj["unit_ids"][rec_name]):
                n_events = len(data_obj["centered_spike_timestamps"][event][rec_name][idx])
                    
                if n_events == 0:
                    continue
                
                col_axis = np.around(np.arange(-abs(pre_event), post_event, bin_size), 2)
                df = pd.DataFrame(data_obj["all_fr_events_per_rat"][event][rec_name][idx]).set_axis(col_axis, axis=1)
                event = df.iloc[:, int(pre_event/bin_size):].mean(axis=1)
                
                invalid = sum([1 if len(i) < spike_threshold else 0 for i in data_obj["centered_spike_timestamps"][event][rec_name][idx]])
                
                if invalid/n_events > 0.5:
                    continue
                
                baseline = df.iloc[:, :int(pre_event/bin_size)].mean(axis=1)
                #Take p-value only
                if any(event) and len(event) > 10:
                    wilcoxon = stats.wilcoxon(event, baseline, correction=True, zero_method="zsplit", method="approx")[1]
                    if wilcoxon < p_bound:
                        fdr_test.loc[f"{rec_name}#{event}#{unit_id}"] = wilcoxon
    
    corrected_p = multipletests(pvals=fdr_test, alpha=0.95, method="fdr_tsbh")[1]
    corr_idx = fdr_test.index
    orig_vs_corr = pd.concat([fdr_test, pd.Series(np.around(corrected_p, 4), index=corr_idx)], keys=["original", "corrected"], axis=1)
    update = orig_vs_corr.query(f"corrected < {p_bound}").copy()
    for i in range(len(update)):
        place = update.iloc[i].name
        places = place.split("#")
        responsive_units[places[0]][places[1]].append(int(places[2]))
    return responsive_units

def neurons_per_structure(data_folder: str | list, data_obj: dict, save_path: str, plot: bool = True):
    """Summary of a number of neurons recorded from each structure

    Args:
        data_folder: path to a folder containing all ephys data folders
        data_obj: output of structurize_data function. Object containing ephys data structure
        save_path: path to which the plots and csv files will be saved
        plot: default True, plots simple bar plot summarizing neurons per structure
    Returns:
        Saves histograms of a number of neurons per structure and csv files with the data 
    """
    data_folder = auxiliary.check_data_folder(data_folder)

    per_structure = []
    for subject, folder in zip(list(data_obj["responsive_units"].keys()), data_folder):
        subject_responsive = []
        for behavior in list(data_obj["responsive_units"][subject].keys()):
            responsive_units = data_obj["responsive_units"][subject][behavior]
            subject_responsive.append(responsive_units)
        subject_responsive = sum(subject_responsive, [])
        subject_responsive = pd.Series(subject_responsive).unique()
        path = os.path.join(folder, "cluster_info_good.csv")
        df = pd.read_csv(path, index_col="cluster_id")
        df = df.loc[subject_responsive, "Structure"].value_counts()
        
        per_structure.append(df)

    df = pd.concat(per_structure)
    df = df.groupby(level=0).sum()

    df.to_csv(os.path.join(save_path, "neurons_per_structure.csv"))
    
    if plot:
        plotting_funcs.plot_nrns_per_structure(df, save_path)

def neurons_per_event(data_folder: str | list, data_obj: dict, save_path: str, plot: bool = True):
    """Summary of a number of neurons per animal, event in a csv, creates csv for each structure
    NOTE: Neurons are repeated if a neuron is responsive to more than one behavior.
    
    Args:
        data_folder: path to a folder containing all ephys data folders
        data_obj: output of structurize_data function. Object containing ephys data structure
        save_path: path to which the plots and csv files will be saved
        plot: default True, if True creates simple bar plots per strucutre, x axis are events, y axis are neurons
    Returns:
        Saves histograms and data per event to desired location
    """
    data_folder = auxiliary.check_data_folder(data_folder)

    behaviors = list(data_obj["all_fr_events_per_rat"].keys())
    subjects = list(data_obj["responsive_units"].keys())
    per_structure = []

    for subject, folder in zip(subjects, data_folder):
        subject_responsive = []
        for behavior in behaviors:
            responsive_units = data_obj["responsive_units"][subject][behavior]
            subject_responsive.append(responsive_units)
        subject_responsive = sum(subject_responsive, [])
        subject_responsive = pd.Series(subject_responsive).unique()
        path = os.path.join(folder, "cluster_info_good.csv")
        df = pd.read_csv(path, index_col="cluster_id")
        structures = df["Structure"].unique()
        temp = pd.DataFrame(np.nan, index=structures, columns=behaviors)
        for behavior in behaviors:
            for structure in structures:
                temp.loc[structure, behavior] = len(df[behavior].loc[((df["Structure"] == structure) & (df[behavior] == 1))])
        per_structure.append(temp)

    df = pd.concat(per_structure)
    df = df.groupby(level=0).sum()

    df.to_csv(os.path.join(save_path, "neurons_per_behavior&structure.csv"))

    if plot:
        plotting_funcs.plot_nrns_per_event(df, save_path)
