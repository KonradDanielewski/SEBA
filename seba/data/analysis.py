"""
@author: K. Danielewski
"""

import os
import pickle
from glob import glob

import numpy as np
import pandas as pd
import polars as pl
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
    data_folder: str | list,
    sigma_sec: float = 0.1,
    pre_event: float = 1.0,
    post_event: float = 3.0,
    bin_size: float = 0.01,
    calculate_responsive: bool = True,
    p_bound: float | None = None,
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
        calculate_responsive: whether to classify neuron responsivity via z-score threshold.
        p_bound: two-tailed p-value used to derive the z-score cutoff (e.g. 0.05 → z=1.96).
            If not provided and calculate_responsive is True, defaults to 0.05.
        save_output: saves output data structure to a pickle file.
        save_path: path to which the file should be saved if save_output is True. File will be named ephys_data.pickle.
        event_names: list of all possible events if different from the ones in the first recording directory.

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

                all_fr, mean_fr = ephys.fr_events_binless(centered_ts, sigma_sec, sampling_out, pre_event, post_event)[:2]
                all_zsc, mean_zsc, sem, _ = ephys.zscore_events(all_fr, bin_size, pre_event, post_event)

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
            data_obj["responsive_units"] = responsive_units_zscore(
                data_obj=data_obj,
                p_bound=p_bound,
                events=event_names,
                rec_names=rec_names,
            )
            return data_obj
        else:
            return data_obj
    finally:
        if save_output:
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            with open(os.path.join(f"{save_path}", "ephys_data.pickle"), "wb") as handle:
                pickle.dump(data_obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def responsive_units_zscore(
    data_obj: dict,
    p_bound: float = 0.05,
    events: list[str] | None = None,
    rec_names: list[str] | None = None,
) -> pl.DataFrame:
    """Detect responsive neurons by thresholding the mean z-scored PSTH.

    A neuron is responsive if its mean z-score exceeds the threshold derived from
    p_bound in any post-event bin. The threshold is the two-tailed critical value:
    z = norm.ppf(1 - p_bound / 2), e.g. 1.96 for p=0.05, 2.58 for p=0.01.
    Only post-event bins are evaluated (bins at t >= 0).

    Args:
        data_obj: output of structurize_data
        p_bound: two-tailed p-value threshold used to derive the z-score cutoff
        events: subset of events to test (default: all)
        rec_names: subset of recordings to test (default: all)

    Returns:
        pl.DataFrame with columns: recording (str), event (str), neuron_id (int),
        responsivity (str: "positive" | "negative" | "no_response")
    """
    if events is None:
        events = data_obj["event_names"]
    if rec_names is None:
        rec_names = data_obj["recording_names"]

    z_threshold = stats.norm.ppf(1 - p_bound / 2)

    pre_event = abs(data_obj["bin_params"]["pre_event"])
    bin_size = data_obj["bin_params"]["bin_size"]
    post_event_col = round(pre_event / bin_size)

    rows = []
    for rec_name in rec_names:
        for event in events:
            psth = data_obj["mean_zscored_events_per_rat"][event].get(rec_name)
            if psth is None:
                continue

            post_event_psth = psth.dropna().iloc[:, post_event_col:]

            for neuron_id in post_event_psth.index:
                row = post_event_psth.loc[neuron_id]
                if row.max() > z_threshold:
                    responsivity = "positive"
                elif row.min() < -z_threshold:
                    responsivity = "negative"
                else:
                    responsivity = "no_response"
                rows.append({
                    "recording": rec_name,
                    "event": event,
                    "unit_id": int(neuron_id),
                    "responsivity": responsivity,
                })

    return pl.DataFrame(rows)


def neurons_per_structure(
    data_folder: str | list,
    data_obj: dict,
    save_path: str,
    plot: bool = True,
):
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

        df = pd.read_csv(os.path.join(folder, "cluster_info.tsv"), sep="\t", index_col="cluster_id")
        df = df[df["group"] == "good"]
        valid = df.index.intersection(subject_responsive)
        df = df.loc[valid, "Structure"].value_counts()

        per_structure.append(df)

    df = pd.concat(per_structure)
    df = df.groupby(level=0).sum()

    df.to_csv(os.path.join(save_path, "neurons_per_structure.csv"))

    if plot:
        plotting_funcs.plot_nrns_per_structure(df, save_path)
