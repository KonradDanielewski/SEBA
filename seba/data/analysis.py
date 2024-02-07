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
    spike_threshold: int | None = None,
    save_output: bool = True,
    save_path: str | None = None,
    event_names: list[str] | None = None,
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
        event_names = [
            os.path.basename(filename).split(".")[0]
            for filename in glob(os.path.join(f"{data_folder[0]}", "events", "*.txt"))
        ]

    rec_names = [os.path.basename(i) for i in data_folder]

    sampling_out = 1 / bin_size

    data_obj = auxfun_data.prepare_data_structure(event_names, rec_names, pre_event, post_event, bin_size)

    if calculate_responsive and p_bound == None:
        p_bound = 0.01

    for event in event_names:
        # Get all mean z-scores to combine into one df
        mean_zscs = []
        mean_frs = []

        for spks_dir, rec in zip(data_folder, rec_names):
            condition = os.path.join(spks_dir, "events", f"{event}.txt")
            try:
                file = np.loadtxt(condition)
            except FileNotFoundError:
                file = []

            if len(file) == 0:
                # Fill data with None
                spikes_ts, unit_ids = auxfun_ephys.read_spikes(spks_dir)
                data_obj["spike_timestamps"][rec] = spikes_ts
                data_obj["unit_ids"][rec] = unit_ids

                cols = [[rec], np.around(np.arange(-abs(pre_event), post_event, bin_size), 2)]
                index = pd.MultiIndex.from_product(cols, names=["rat", "time"])
                mean_zsc = pd.DataFrame(np.nan, index=unit_ids, columns=index)
                # Can be the same since it's only a placeholder to produce the full df later
                mean_zscs.append(mean_zsc)
                mean_frs.append(mean_zsc)

            else:
                spikes_ts, unit_ids = auxfun_ephys.read_spikes(spks_dir)
                centered_ts = ephys.calc_rasters(spikes_ts, condition, pre_event, post_event)

                # Write spike times and unit IDs
                data_obj["spike_timestamps"][rec] = spikes_ts
                data_obj["unit_ids"][rec] = unit_ids
                data_obj["centered_spike_timestamps"][event][rec] = centered_ts

                # Take only all_fr, all_zsc and mean_fr, mean_zsc and bin_edges
                all_fr, mean_fr = ephys.fr_events_binless(centered_ts, sigma_sec, sampling_out, pre_event, post_event)[:2]
                all_zsc, mean_zsc, sem, bin_edges = ephys.zscore_events(all_fr, bin_size, pre_event, post_event)

                # Prepare index and columns
                cols = [[rec], np.around(np.arange(-abs(pre_event), post_event, bin_size), 2)]
                index = pd.MultiIndex.from_product(cols, names=["rat", "time"])

                mean_zsc = pd.DataFrame(mean_zsc, index=unit_ids, columns=index)
                mean_zscs.append(mean_zsc)
                mean_fr = pd.DataFrame(mean_fr, index=unit_ids, columns=index)
                mean_frs.append(mean_fr)

                # Write all firing rates per rat, all zscored per rat and center spike timestamps
                data_obj["all_fr_events_per_rat"][event][rec] = np.array(all_fr)
                data_obj["all_zscored_events_per_rat"][event][rec] = all_zsc

        # per rat dataframes with level 0 containing rat name
        per_rat_fr = pd.concat(mean_frs, axis=1)
        per_rat_zscore = pd.concat(mean_zscs, axis=1)

        for i in range(len(mean_zscs)):
            mean_zscs[i] = mean_zscs[i].droplevel(0, axis=1)
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
            data_obj = responsive_units_wilcoxon(
                data_obj=data_obj,
                p_bound=p_bound,
                bin_edges=bin_edges,
                events=event_names,
                rec_names=rec_names,
                spike_threshold=spike_threshold,
                )
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
    p_bound: float,
    bin_edges: np.array,
    events: list[str] | None = None,
    rec_names: list[str] | None = None,
    spike_threshold: int | None = None,
) -> dict:
    """Performs Wilcoxon rank test to establish whether a neuron is responsive to a specific event.

    Args:
        data_obj: output of structurize_data. Stored in ephys_data.pickle
        events: names of events
        rec_names: names of recordings
        p_bound: p value to be used in filtering rank test output
        spike_threshold: specifies minimum number of spikes that should happen per event instance to be considered in rank test.
            if more than half of the events have less than that number of spikes this neuron is discarded from the rank test.

    Returns:
        Dictionary of responsive units per recording per event.
    """
    if isinstance(events, list):
        pass
    else:
        events = data_obj["event_names"]

    if isinstance(rec_names, list):
        pass
    else:
        rec_names = data_obj["recording_names"]

    fdr_test = pd.DataFrame(columns=["raw", "corrected"])
    for event_name in events:
        for rec_name in rec_names:
            if data_obj["centered_spike_timestamps"][event_name][rec_name] == None:
                continue
            for idx, unit_id in enumerate(data_obj["unit_ids"][rec_name]):
                n_events = len(data_obj["centered_spike_timestamps"][event_name][rec_name][idx])

                if n_events == 0:
                    continue

                if isinstance(spike_threshold, int):
                    spikes_per_event = [len(i) for i in data_obj["centered_spike_timestamps"][event_name][rec_name][idx]]
                    invalid = sum([1 if i < spike_threshold else 0 for i in spikes_per_event])

                    if invalid / n_events > 0.5:
                        continue

                baseline = []
                event = []
                for instance in data_obj["centered_spike_timestamps"][event_name][rec_name][idx]:
                    baseline.append(len(instance[instance < 0]) / abs(min(bin_edges)))
                    event.append(len(instance[instance > 0]) / max(bin_edges))
                if (len(event) > 9):  # Minimum value that doesn't result in a warning that there aren't enough events
                    wilcoxon = stats.wilcoxon(
                        event,
                        baseline,
                        correction=True,
                        zero_method="zsplit",
                        method="approx",
                    )[1]  # Take only p value
                else:
                    continue
                if wilcoxon < p_bound:
                    fdr_test.loc[f"{rec_name}#{event_name}#{unit_id}", "raw"] = wilcoxon

    fdr_test["corrected"] = multipletests(pvals=fdr_test["raw"], alpha=0.95, method="fdr_tsbh")[1]
    significant = fdr_test[fdr_test["corrected"] < p_bound].index
    for i in significant:
        recording, eve, id = i.split("#")
        data_obj["responsive_units"][recording][eve].append(int(id))

    return data_obj


def neurons_per_structure(data_folder: str | list, data_obj: dict, save_path: str, plot: bool = True):
    """Summary of a number of neurons recorded from each structure

    Args:
        data_folder: path to a folder containing all ephys data folders
        data_obj: output of structurize_data function. Object containing ephys data structure
        save_path: path to which the plots and csv files will be saved
        plot: default True, plots simple bar plot summarizing neurons per structure

    Returns:
        Saves histograms of a number of neurons per structure and csv files with the data
    """# TODO: Add responsive/all
    data_folder = auxiliary.check_data_folder(data_folder)
    events = data_obj["event_names"]
    rec_names = data_obj["recording_names"]

    per_structure = []
    for rec_name, folder in zip(rec_names, data_folder):
        subject_responsive = sum([data_obj["responsive_units"][rec_name][event] for event in events], [])
        subject_responsive = pd.Series(subject_responsive).unique()

        df = pd.read_csv(os.path.join(folder, "cluster_info_good.csv"), index_col="cluster_id")
        df = df.loc[subject_responsive, "Structure"].value_counts()

        per_structure.append(df)

    df = pd.concat(per_structure)
    df = df.groupby(level=0).sum()

    df.to_csv(os.path.join(save_path, "neurons_per_structure.csv"))

    if plot:
        plotting_funcs.plot_nrns_per_structure(df, save_path)
