"""
@author: K. Danielewski

Data preparation functions for SEBA plotting.
Each function returns ready-to-plot arrays/DataFrames; no figures are created here.
"""

import os
from itertools import combinations_with_replacement

import numpy as np
import pandas as pd
import polars as pl


# ---------------------------------------------------------------------------
# Responsive unit helpers
# ---------------------------------------------------------------------------

def _responsive_ids(data_obj: dict, rec_name: str, event: str) -> list[int]:
    """Return unit IDs labelled 'increase' or 'decrease' for a recording/event."""
    df: pl.DataFrame | None = data_obj.get("responsive_units")
    if df is None:
        return []
    return (
        df.filter(
            (pl.col("recording") == rec_name)
            & (pl.col("event") == event)
            & (pl.col("responsivity") != "no_response")
        )["unit_id"]
        .to_list()
    )


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------

def load_dict(
    which: str,
    z_score: bool,
    data_obj: dict,
    event_pair: list[str, str] | None = None,
    event: str | None = None,
    single: bool | None = None,
) -> tuple:
    """Load a slice of data_obj for plotting.

    Args:
        which: one of 'mean_all_rats', 'mean_per_rat', 'all_rats'
        z_score: return z-scored data when True, raw firing rate when False
        data_obj: ephys data structure (output of structurize_data)
        event_pair: two event names for paired plots
        event: single event name for single-event plots
        single: True → return (y_axis, subject); False → return (y_axis, subjects, partners)

    Returns:
        Tuple of (y_axis_label, subject_data [, partner_data])
    """
    y_axis = "z-score" if z_score else "fr"

    if which == "mean_all_rats":
        key = "mean_zscored_events_all_rats" if z_score else "mean_fr_events_all_rats"
        if single:
            return y_axis, data_obj[key][event]
        return y_axis, data_obj[key][event_pair[0]], data_obj[key][event_pair[1]]

    if which == "mean_per_rat":
        key = "mean_zscored_events_per_rat" if z_score else "mean_fr_events_per_rat"
        if single:
            return y_axis, data_obj[key][event]
        return y_axis, data_obj[key][event_pair[0]], data_obj[key][event_pair[1]]

    if which == "all_rats":
        key = "all_zscored_events_per_rat" if z_score else "all_fr_events_per_rat"
        if single:
            return y_axis, data_obj[key][event]
        return y_axis, data_obj[key][event_pair[0]], data_obj[key][event_pair[1]]


# ---------------------------------------------------------------------------
# Per-unit PSTH helpers
# ---------------------------------------------------------------------------

def get_psth_unit_data(
    data_obj: dict,
    event: str,
    rec_name: str,
    unit_id: int,
    z_score: bool = True,
) -> tuple[list, np.ndarray, np.ndarray, np.ndarray, str]:
    """Prepare raster and PSTH data for a single unit.

    Args:
        data_obj: ephys data structure
        event: event name
        rec_name: recording name
        unit_id: cluster ID of the unit
        z_score: use z-score when True, raw firing rate when False

    Returns:
        spike_trains: list of per-trial spike time arrays (for raster)
        col_axis: time bin centres
        mean: mean firing rate/z-score per bin
        sem: standard error of the mean per bin
        y_axis: axis label string
    """
    pre_event = data_obj["bin_params"]["pre_event"]
    post_event = data_obj["bin_params"]["post_event"]
    bin_size = data_obj["bin_params"]["bin_size"]
    col_axis = np.around(np.arange(-abs(pre_event), post_event, bin_size), 2)

    unit_idx = list(data_obj["unit_ids"][rec_name]).index(unit_id)
    y_axis, all_data = load_dict(which="all_rats", z_score=z_score, data_obj=data_obj, event=event, single=True)

    unit_fr = np.array(all_data[rec_name][unit_idx])
    mean = unit_fr.mean(axis=0)
    sem = unit_fr.std(axis=0) / np.sqrt(len(unit_fr))
    spike_trains = data_obj["centered_spike_timestamps"][event][rec_name][unit_idx]

    return spike_trains, col_axis, mean, sem, y_axis


def get_paired_psth_unit_data(
    data_obj: dict,
    event_pair: list[str, str],
    rec_name: str,
    unit_id: int,
    z_score: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Prepare PSTH data for a single unit across two paired events.

    Returns:
        col_axis, mean_subject, sem_subject, mean_partner, sem_partner, y_axis
    """
    pre_event = data_obj["bin_params"]["pre_event"]
    post_event = data_obj["bin_params"]["post_event"]
    bin_size = data_obj["bin_params"]["bin_size"]
    col_axis = np.around(np.arange(-abs(pre_event), post_event, bin_size), 2)

    unit_idx = list(data_obj["unit_ids"][rec_name]).index(unit_id)
    y_axis, subjects, partners = load_dict(
        which="all_rats", z_score=z_score, data_obj=data_obj, event_pair=event_pair, single=False
    )

    def _mean_sem(data_dict):
        arr = np.array(data_dict[rec_name][unit_idx])
        return arr.mean(axis=0), arr.std(axis=0) / np.sqrt(len(arr))

    mean_sub, sem_sub = _mean_sem(subjects)
    mean_par, sem_par = _mean_sem(partners)
    return col_axis, mean_sub, sem_sub, mean_par, sem_par, y_axis


# ---------------------------------------------------------------------------
# Common neurons matrix
# ---------------------------------------------------------------------------

def get_common_matrix_data(
    data_obj: dict,
    per_recording: bool,
    percent_all_neurons: bool,
) -> list[tuple[pd.DataFrame, str | None]]:
    """Compute pairwise common responsive neuron counts for all event pairs.

    Returns:
        List of (df, rec_name) tuples. rec_name is None for the combined case.
    """
    rec_names = data_obj["recording_names"]
    events = data_obj["event_names"]
    results = []

    if per_recording:
        for rec_name in rec_names:
            df = pd.DataFrame(np.nan, index=events, columns=events)
            for event1, event2 in combinations_with_replacement(events, 2):
                units1 = _responsive_ids(data_obj, rec_name, event1)
                units2 = _responsive_ids(data_obj, rec_name, event2)
                if not units1 or not units2:
                    continue
                df.loc[event2, event1] = len(np.intersect1d(units1, units2))
            if percent_all_neurons:
                df = df / len(data_obj["unit_ids"][rec_name]) * 100
            results.append((df, rec_name))
    else:
        df = pd.DataFrame(0, index=events, columns=events)
        for rec_name in rec_names:
            for event1, event2 in combinations_with_replacement(events, 2):
                units1 = _responsive_ids(data_obj, rec_name, event1)
                units2 = _responsive_ids(data_obj, rec_name, event2)
                if not units1 or not units2:
                    continue
                df.loc[event2, event1] += len(np.intersect1d(units1, units2))
        if percent_all_neurons:
            total = sum(len(data_obj["unit_ids"][r]) for r in rec_names)
            df = df / total * 100
        results.append((df, None))

    return results


# ---------------------------------------------------------------------------
# Linear regression scatter
# ---------------------------------------------------------------------------

def get_lin_reg_data(
    data_obj: dict,
    event_pair: list[str, str],
    per_recording: bool,
    responsive_only: bool,
    z_score: bool = True,
) -> list[tuple[pd.DataFrame, str | None]]:
    """Prepare subject/partner mean firing rate data for regression scatter plot.

    Returns:
        List of (df, rec_name) where df has 'subject' and 'partner' columns.
        rec_name is None for the combined case.
    """
    rec_names = data_obj["recording_names"]
    _, subjects, partners = load_dict(
        which="mean_per_rat", z_score=z_score, data_obj=data_obj, event_pair=event_pair, single=False
    )
    results = []

    def _extract(rec_name):
        subject = subjects[rec_name].dropna().mean(axis=1).rename("subject")
        partner = partners[rec_name].dropna().mean(axis=1).rename("partner")
        if responsive_only:
            resp = np.unique(
                _responsive_ids(data_obj, rec_name, event_pair[0])
                + _responsive_ids(data_obj, rec_name, event_pair[1])
            )
            subject = subject.loc[subject.index.intersection(resp)]
            partner = partner.loc[partner.index.intersection(resp)]
        return subject, partner

    if per_recording:
        for rec_name in rec_names:
            subject, partner = _extract(rec_name)
            if len(subject) == 0 or len(partner) == 0:
                continue
            results.append((pd.concat([subject, partner], axis=1), rec_name))
    else:
        sub_all, par_all = [], []
        for rec_name in rec_names:
            subject, partner = _extract(rec_name)
            if len(subject) == 0 or len(partner) == 0:
                continue
            sub_all.append(subject)
            par_all.append(partner)
        df = pd.concat(
            [pd.concat(sub_all).reset_index(drop=True), pd.concat(par_all).reset_index(drop=True)],
            axis=1,
        )
        results.append((df, None))

    return results


# ---------------------------------------------------------------------------
# Single heatmap
# ---------------------------------------------------------------------------

def get_heatmap_data(
    data_obj: dict,
    event: str,
    per_recording: bool,
    responsive_only: bool,
    z_score: bool,
) -> list[tuple[pd.DataFrame, np.ndarray, int, str, str | None]]:
    """Prepare heatmap data, sorted by post-onset mean activity.

    Returns:
        List of (df, col_axis, onset_col_idx, y_axis, rec_name).
        rec_name is None for the combined case.
    """
    pre_event = data_obj["bin_params"]["pre_event"]
    post_event = data_obj["bin_params"]["post_event"]
    bin_size = data_obj["bin_params"]["bin_size"]
    col_axis = np.around(np.arange(-abs(pre_event), post_event, bin_size), 2)
    onset_col_idx = int(pre_event / bin_size)
    rec_names = data_obj["recording_names"]
    results = []

    if per_recording:
        y_axis, subjects = load_dict(which="all_rats", z_score=z_score, data_obj=data_obj, event=event, single=True)
        for rec_name in rec_names:
            if subjects[rec_name] is None:
                continue
            unit_ids = data_obj["unit_ids"][rec_name]
            per_unit_mean = [np.array(arr).mean(axis=0) for arr in subjects[rec_name]]
            df = pd.DataFrame(per_unit_mean, index=unit_ids, columns=col_axis)
            if responsive_only:
                resp = _responsive_ids(data_obj, rec_name, event)
                if not resp:
                    continue
                df = df.loc[resp]
            sorted_idx = df.iloc[:, onset_col_idx:].mean(axis=1).sort_values().index
            results.append((df.reindex(sorted_idx).reset_index(drop=True), col_axis, onset_col_idx, y_axis, rec_name))
    else:
        if responsive_only:
            y_axis, subjects = load_dict(which="all_rats", z_score=z_score, data_obj=data_obj, event=event, single=True)
            chunks = []
            for rec_name in rec_names:
                if subjects[rec_name] is None:
                    continue
                resp = _responsive_ids(data_obj, rec_name, event)
                if not resp:
                    continue
                per_unit_mean = [np.array(arr).mean(axis=0) for arr in subjects[rec_name]]
                df = pd.DataFrame(per_unit_mean, index=data_obj["unit_ids"][rec_name], columns=col_axis)
                chunks.append(df.loc[resp])
            if not chunks:
                return results
            df = pd.concat(chunks).reset_index(drop=True)
        else:
            y_axis, df = load_dict(which="mean_all_rats", z_score=z_score, data_obj=data_obj, event=event, single=True)

        sorted_idx = df.iloc[:, onset_col_idx:].mean(axis=1).sort_values().index
        results.append((df.reindex(sorted_idx).reset_index(drop=True), col_axis, onset_col_idx, y_axis, None))

    return results


# ---------------------------------------------------------------------------
# Paired heatmap
# ---------------------------------------------------------------------------

def get_paired_heatmap_data(
    data_obj: dict,
    event_pair: list[str, str],
    per_recording: bool,
    responsive_only: bool,
    z_score: bool,
) -> list[tuple[pd.DataFrame, pd.DataFrame, np.ndarray, int, str, str | None]]:
    """Prepare paired heatmap data, sorted by combined post-onset activity.

    Returns:
        List of (subject_df, partner_df, col_axis, onset_col_idx, y_axis, rec_name).
        rec_name is None for the combined case.
    """
    pre_event = data_obj["bin_params"]["pre_event"]
    post_event = data_obj["bin_params"]["post_event"]
    bin_size = data_obj["bin_params"]["bin_size"]
    col_axis = np.around(np.arange(-abs(pre_event), post_event, bin_size), 2)
    onset_col_idx = int(pre_event / bin_size)
    rec_names = data_obj["recording_names"]
    results = []

    y_axis, subjects_dict, partners_dict = load_dict(
        which="all_rats", z_score=z_score, data_obj=data_obj, event_pair=event_pair, single=False
    )

    def _to_df(data_dict, rec_name):
        per_unit = [np.array(arr).mean(axis=0) for arr in data_dict[rec_name]]
        return pd.DataFrame(per_unit, index=data_obj["unit_ids"][rec_name], columns=col_axis)

    def _sort_pair(subj_df, part_df):
        sorted_idx = (
            (subj_df.iloc[:, onset_col_idx:] + part_df.iloc[:, onset_col_idx:])
            .mean(axis=1).sort_values().index
        )
        return (
            subj_df.reindex(sorted_idx).reset_index(drop=True),
            part_df.reindex(sorted_idx).reset_index(drop=True),
        )

    if per_recording:
        for rec_name in rec_names:
            if subjects_dict[rec_name] is None or partners_dict[rec_name] is None:
                continue
            subj_df = _to_df(subjects_dict, rec_name)
            part_df = _to_df(partners_dict, rec_name)
            if responsive_only:
                resp = np.unique(
                    _responsive_ids(data_obj, rec_name, event_pair[0])
                    + _responsive_ids(data_obj, rec_name, event_pair[1])
                )
                if len(resp) == 0:
                    continue
                subj_df = subj_df.loc[resp]
                part_df = part_df.loc[resp]
            subj_df, part_df = _sort_pair(subj_df, part_df)
            results.append((subj_df, part_df, col_axis, onset_col_idx, y_axis, rec_name))
    else:
        if responsive_only:
            subj_chunks, part_chunks = [], []
            for rec_name in rec_names:
                if subjects_dict[rec_name] is None or partners_dict[rec_name] is None:
                    continue
                resp = np.unique(
                    _responsive_ids(data_obj, rec_name, event_pair[0])
                    + _responsive_ids(data_obj, rec_name, event_pair[1])
                )
                if len(resp) == 0:
                    continue
                subj_chunks.append(_to_df(subjects_dict, rec_name).loc[resp])
                part_chunks.append(_to_df(partners_dict, rec_name).loc[resp])
            subj_df = pd.concat(subj_chunks).reset_index(drop=True)
            part_df = pd.concat(part_chunks).reset_index(drop=True)
        else:
            y_axis, subj_df, part_df = load_dict(
                which="mean_all_rats", z_score=z_score, data_obj=data_obj, event_pair=event_pair, single=False
            )
        subj_df, part_df = _sort_pair(subj_df, part_df)
        results.append((subj_df, part_df, col_axis, onset_col_idx, y_axis, None))

    return results


# ---------------------------------------------------------------------------
# Structure × event heatmap
# ---------------------------------------------------------------------------

def get_per_event_structure_data(
    data_folder: list,
    data_obj: dict,
    per_recording: bool,
) -> list[tuple[pd.DataFrame, str | None]]:
    """Prepare structure-vs-event neuron count data from cluster_info.tsv.

    Returns:
        List of (df, rec_name) tuples. rec_name is None for the combined case.
    """
    rec_names = data_obj["recording_names"]
    event_names = data_obj["event_names"]
    cols = event_names + ["Structure"]
    results = []

    if per_recording:
        for folder, rec_name in zip(data_folder, rec_names):
            df = pd.read_csv(
                os.path.join(folder, "cluster_info.tsv"),
                sep="\t",
                index_col="cluster_id",
                usecols=cols + ["cluster_id", "group"],
            )
            df = df[df["group"] == "good"].loc[:, event_names + ["Structure"]].groupby("Structure").sum()
            results.append((df, rec_name))
    else:
        frames = [
            pd.read_csv(
                os.path.join(folder, "cluster_info.tsv"),
                sep="\t",
                index_col="cluster_id",
                usecols=cols + ["cluster_id", "group"],
            )
            for folder in data_folder
        ]
        df = pd.concat(frames)
        df = df[df["group"] == "good"].loc[:, event_names + ["Structure"]].groupby("Structure").sum()
        results.append((df, None))

    return results
