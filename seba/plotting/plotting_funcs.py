"""
@author: K. Danielewski
"""

import os
from itertools import combinations_with_replacement

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from seba.utils import (
    auxiliary,
    auxfun_plotting,
)


def plot_common_nrns_matrix(
    data_obj: dict,
    save_path: str,
    per_recording: bool = False,
    percent_all_neurons: bool = True,
    colormap: str = "inferno",
):
    """Function creating matrix plot of a perctage of common responsive neurons between pairs of events

    Args:
        data_obj: output of structurize_data. Stored in ephys_data.pickle
        save_path: path to which the plots will be saved. Additional subfolders will be created on per-recording basis
        per_recording: Toggles if the regression is done and plotted for all recorded neurons or on per-recording basis. Defaults to False.
        percent_all_neurons: Toggles whether to show values as percentage of all neurons or percentage of responsive neurons. Defaults to True.
    Returns:
        Saves a matrix of the amount of common responsive neurons across event pairs
    """
    rec_names = data_obj["recording_names"]
    events = data_obj["event_names"]

    if per_recording:
        for rec_name in rec_names:
            df = pd.DataFrame(np.nan, index=events, columns=events)
            for event1, event2 in combinations_with_replacement(events, 2):
                if (
                    data_obj["responsive_units"][rec_name][event1] is None
                    or data_obj["responsive_units"][rec_name][event2] is None
                ):
                    continue
                df.loc[event2, event1] = len(
                    np.intersect1d(data_obj["responsive_units"][rec_name][event1], data_obj["responsive_units"][rec_name][event2])
                )

            if percent_all_neurons:
                nrns = len(data_obj["unit_ids"][rec_name])
                df = df / nrns * 100

            auxfun_plotting.make_common_matrix(df, save_path, colormap=colormap, per_recording=per_recording, rec_name=rec_name)
    else:
        df = pd.DataFrame(0, index=events, columns=events)

        for rec_name in rec_names:
            for event1, event2 in combinations_with_replacement(events, 2):
                if (
                    data_obj["responsive_units"][rec_name][event1] is None
                    or data_obj["responsive_units"][rec_name][event2] is None
                ):
                    continue
                count = len(
                    np.intersect1d(data_obj["responsive_units"][rec_name][event1], data_obj["responsive_units"][rec_name][event2])
                )
                df.loc[event2, event1] += count

        if percent_all_neurons:
            nrns = []
            for rec_name in rec_names:
                nrns.append(len(data_obj["unit_ids"][rec_name]))
            df = df / sum(nrns) * 100

        auxfun_plotting.make_common_matrix(df, save_path, colormap=colormap, per_recording=per_recording)


def plot_psths(
    data_obj: dict,
    save_path: str,
    responsive_only: bool = False,
    z_score: bool = True,
    ylimit: list[float, float] = [-1, 2.5],
):
    """Function used for plotting psths using either z-score or raw firing rate

    Args:
        data_obj: output of structurize_data function. Object containing ephys data structure
        save_path: path to which the plots will be saved. Additional subfolders will be created on per-recording basis
        responsive_only: Toggles whether to plot for all recorded cells or only the ones deemed significantly responsive. Defaults to False
        z_score: if to use z-score or raw firing rate. Defaults to True.
        ylimit: y_axis limit, first value is lower, second is upper bound. Defaults to [-1, 2.5].
    Returns:
        Saves plots to set location, creating subfolders on per-recording basis. Each plot is named using event name and neuron ID
    """
    sns.set_palette("colorblind")

    pre_event = data_obj["bin_params"]["pre_event"]
    post_event = data_obj["bin_params"]["post_event"]
    bin_size = data_obj["bin_params"]["bin_size"]

    rec_names = data_obj["recording_names"]
    events = data_obj["event_names"]

    unit_ids = data_obj["unit_ids"]
    responsive_ids = data_obj["responsive_units"]

    for event in events:
        y_axis, subject = auxfun_plotting.load_dict(
            which="all_rats",
            z_score=z_score,
            data_obj=data_obj,
            event=event,
            single=True,
        )
        for rec_name in rec_names:
            if subject[rec_name] is None:
                continue
            if data_obj["responsive_units"][rec_name][event] is None:
                continue
            save_here = auxiliary.make_dir_save(save_path, rec_name)

            for idx, unit_id in enumerate(responsive_ids[rec_name][event]):
                # Prepare data for proper axes
                col_axis = np.around(np.arange(-abs(pre_event), post_event, bin_size), 2)
                df0 = pd.DataFrame(subject[rec_name][idx]).set_axis(col_axis, axis=1)
                df0 = df0.melt(var_name="Seconds", value_name=y_axis, ignore_index=False)

                # Build figure
                fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 8), sharex=True)
                axes = axes.flatten()

                axes[0].eventplot(
                    data_obj["centered_spike_timestamps"][event][rec_name][idx],
                    linelengths=0.7,
                )
                sns.lineplot(df0, ax=axes[1], x="Seconds", y=y_axis, errorbar="se")

                # Plottting params
                axes[0].set_ylabel("Trial #", fontsize=14)
                axes[0].set_title(event)
                axes[1].set_xlabel("Seconds", fontsize=14)
                axes[1].set_ylabel(y_axis, fontsize=14)
                axes[1].set_ylim([ylimit[0], ylimit[1]])

                save = auxiliary.make_dir_save(save_here, event)

                fig.savefig(os.path.join(save, f"{event}_psth_{str(unit_id)}.png"), dpi=100, bbox_inches="tight")
                plt.close(fig)


def plot_psths_paired(
    data_obj: dict,
    event_pair: list[str, str],
    save_path: str,
    responsive_only: bool = False,
    z_score: bool = True,
    ylimit: list[float, float] = [-1, 2.5],
):
    """Function used for plotting psths using either z-score or raw firing rate

    Args:
        data_obj: output of structurize_data function. Object containing ephys data structure
        event_pair: list of two keys to plot together, assumes [subject, partner] order e.g., ["freezing_subject", "freezing_partner"]
        save_path: path to which the plots will be saved. Additional subfolders will be created on per-recording basis
        responsive_only: Toggles whether to plot for all recorded cells or only the ones deemed significantly responsive. Defaults to False
        z_score: if to use z-score or raw firing rate. Defaults to True.
        ylimit: y_axis limit, first value is lower, second is upper bound. Defaults to [-1, 2.5].
    Returns:
        Saves plots to set location, creating subfolders on per-recording basis. Each plot is named using event name and neuron ID
    """
    sns.set_palette("colorblind")

    pre_event = data_obj["bin_params"]["pre_event"]
    post_event = data_obj["bin_params"]["post_event"]
    bin_size = data_obj["bin_params"]["bin_size"]
    unit_ids = data_obj["unit_ids"]
    rec_names = data_obj["recording_names"]
    responsive_ids = data_obj["responsive_units"]

    filename = event_pair[0] + "_" + event_pair[1]

    y_axis, subjects, partners = auxfun_plotting.load_dict(
        which="all_rats",
        z_score=z_score,
        data_obj=data_obj,
        event_pair=event_pair,
        single=False,
    )

    for rec_name in rec_names:
        save_here = auxiliary.make_dir_save(save_path, rec_name)
        if responsive_only:
            responsive_cells = list(
                set(
                    responsive_ids[rec_name][event_pair[0]]
                    + responsive_ids[rec_name][event_pair[1]]
                )
            )

        for idx, unit_id in enumerate(unit_ids[rec_name]):
            if responsive_only and unit_id not in responsive_cells:
                continue

            if subjects[rec_name] is None or partners[rec_name] is None:
                continue

            # Prepare data for proper axes
            col_axis = np.around(np.arange(-abs(pre_event), post_event, bin_size), 2)
            df0 = pd.DataFrame(subjects[rec_name][idx]).set_axis(col_axis, axis=1)
            df1 = pd.DataFrame(partners[rec_name][idx]).set_axis(col_axis, axis=1)
            df0 = df0.melt(var_name="Seconds", value_name=y_axis, ignore_index=False)
            df1 = df1.melt(var_name="Seconds", value_name=y_axis, ignore_index=False)

            # Build figure
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6), sharey=True)

            sns.lineplot(
                df0,
                ax=ax,
                x="Seconds",
                y=y_axis,
                errorbar="se",
                label=f"{event_pair[0]}",
            )
            sns.lineplot(
                df1,
                ax=ax,
                x="Seconds",
                y=y_axis,
                errorbar="se",
                label=f"{event_pair[1]}",
            )

            # Plottting params
            ax.set_title(filename.capitalize())
            ax.set_xlabel("Seconds")
            ax.set_ylabel(y_axis)
            ax.set_ylim([ylimit[0], ylimit[1]])

            save = auxiliary.make_dir_save(save_here, filename)

            fig.savefig(os.path.join(save, f"{filename}_psth_{str(unit_id)}.png"), dpi=100, bbox_inches="tight")
            plt.close(fig)


def plot_lin_reg_scatter(
    data_obj: dict,
    event_pair: list[str, str],
    save_path: str,
    per_recording: bool = False,
    responsive_only: bool = False,
    ax_limit: list[float, float] = [-4, 4],
    z_score: bool = True,
):
    """Creates scatter plots with regression results showing linear relationship between neuron
    responses to subjects' and partners' events

    Args:
        data_obj: output of structurize_data function. Object containing ephys data structure
        event_pair: list of two keys to plot together, assumes [subject, partner] order e.g., ["freezing_subject", "freezing_partner"]
        save_path: path to which the plots will be saved. Additional subfolders will be created on per-recording basis
        responsive_only: Toggles whether to plot for all recorded cells or only the ones deemed significantly responsive. Defaults to False
        per_recording: Toggles if the regression is done and plotted for all recorded neurons
            or on per-recording basis. Defaults to False.
        ax_limit: axes limits, assumes 0 centered, even per axis. First value is x axis, second is y axis. Defaults to [-4, 4].
        z_score: required for compatibility. Always True
    Returns:
        Saves a scatter plot to a desired location with results of linear regression containing slope, r-value and p-value of the fit.
    """
    filename = event_pair[0] + "_" + event_pair[1]
    rec_names = data_obj["recording_names"]

    subjects, partners = auxfun_plotting.load_dict(
        which="mean_per_rat",
        z_score=z_score,
        data_obj=data_obj,
        event_pair=event_pair,
        single=False,
    )[1:3]  # y_axis not used

    if per_recording:
        for rec_name in rec_names:
            subject = subjects[rec_name].dropna().mean(axis=1).rename("subject")
            partner = partners[rec_name].dropna().mean(axis=1).rename("partner")

            if len(subject) == 0 or len(partner) == 0:
                continue

            if responsive_only:
                responsive_units = np.unique(
                    data_obj["responsive_units"][rec_name][event_pair[0]]
                    + data_obj["responsive_units"][rec_name][event_pair[1]]
                )
                subject = subject.loc[responsive_units]
                partner = partner.loc[responsive_units]

            df = pd.concat([subject, partner], axis=1)
            # Make plots
            save_here = auxiliary.make_dir_save(save_path, rec_name)
            auxfun_plotting.make_paired_scatter(
                df=df,
                filename=filename,
                event_pair=event_pair,
                ax_limit=ax_limit,
                save_path=save_here,
                )

    else:
        sub_combined = []
        par_combined = []

        for rec_name in rec_names:
            subject = subjects[rec_name].dropna().mean(axis=1).rename("subject")
            partner = partners[rec_name].dropna().mean(axis=1).rename("partner")

            if len(subject) == 0 or len(partner) == 0:
                continue

            if responsive_only:
                responsive_units = np.unique(
                    data_obj["responsive_units"][rec_name][event_pair[0]]
                    + data_obj["responsive_units"][rec_name][event_pair[1]]
                )
                subject = subject.loc[responsive_units]
                partner = partner.loc[responsive_units]

            sub_combined.append(subject)
            par_combined.append(partner)

        subject = pd.concat(sub_combined).reset_index(drop=True)
        partner = pd.concat(par_combined).reset_index(drop=True)
        df = pd.concat([subject, partner], axis=1)

        # Make plots
        auxfun_plotting.make_paired_scatter(
                df=df,
                filename=filename,
                event_pair=event_pair,
                ax_limit=ax_limit,
                save_path=save_path,
            )


def plot_heatmaps(
    data_obj: dict,
    save_path: str,
    per_recording: bool = False,
    responsive_only: bool = False,
    colormap: str = "inferno",
    z_score: bool = True,
    x_tick: int = 50,
    y_tick: int = 25,
):
    """Plots paired heatmaps for sorted neurons comparison (how the same neuron responed to self and parters' event)

    Args:
        data_obj: output of structurize_data function. Object containing ephys data structure
        save_path: path to which the plots will be saved. Additional subfolders will be created on per-recording basis
        per_recording: Toggles if the regression is done and plotted for all recorded neurons
            or on per-recording basis. Defaults to False.
        responsive_only: Toggles whether to plot for all recorded cells or only the ones deemed significantly responsive. Defaults to False
        colormap: matplotlib colormap used for heatmap plotting. Defaults to "viridis".
        z_score: required for compatibility. Always True
    Returns:
        Saves figures to desired location. If per-recording makes/uses folders with recording name
    """
    pre_event = data_obj["bin_params"]["pre_event"]
    post_event = data_obj["bin_params"]["post_event"]
    bin_size = data_obj["bin_params"]["bin_size"]

    rec_names = data_obj["recording_names"]
    events = data_obj["event_names"]

    onset_col_idx = int(pre_event / bin_size)

    for event in events:
        if per_recording == False:
            col_axis = np.around(np.arange(-abs(pre_event), post_event, bin_size), 2)
            if responsive_only:
                y_axis, subjects = auxfun_plotting.load_dict(
                    which="all_rats",
                    z_score=z_score,
                    data_obj=data_obj,
                    event=event,
                    single=True,
                )

                responsive_subject = []

                for rec_name in rec_names:
                    if subjects[rec_name] is None:
                        continue
                    responsive_units = data_obj["responsive_units"][rec_name][event]
                    if len(responsive_units) == 0:
                        continue
                    temp_subject = [i.mean(axis=0) for i in subjects[rec_name]]

                    df_subs = pd.DataFrame(temp_subject, index=data_obj["unit_ids"][rec_name]).loc[responsive_units]
                    responsive_subject.append(df_subs)

                subjects = pd.concat(responsive_subject)
                subject = (subjects.reset_index(drop=True).set_axis(col_axis, axis=1).copy())
            else:
                y_axis, subject = auxfun_plotting.load_dict(
                    which="mean_all_rats",
                    z_score=z_score,
                    data_obj=data_obj,
                    event=event,
                    single=True,
                )

            auxfun_plotting.make_heatmap(
                subject=subject,
                onset_col_idx=onset_col_idx,
                colormap=colormap,
                y_axis=y_axis,
                filename=event,
                save_path=save_path,
                xticklabel=x_tick,
                yticklabel=y_tick,
            )

        if per_recording:
            y_axis, subjects = auxfun_plotting.load_dict(
                which="all_rats",
                z_score=z_score,
                data_obj=data_obj,
                event=event,
                single=True,
            )
            col_axis = np.around(np.arange(-abs(pre_event), post_event, bin_size), 2)

            for rec_name in rec_names:

                save_here = auxiliary.make_dir_save(save_path, rec_name)

                if responsive_only:
                    responsive_units = data_obj["responsive_units"][rec_name][event]
                    if subjects[rec_name] is None:
                        continue
                    if len(responsive_units) == 0:
                        continue
                    subject = pd.DataFrame(np.array(subjects[rec_name]).mean(axis=1), index=data_obj["unit_ids"][rec_name], columns=col_axis).loc[responsive_units]
                    auxfun_plotting.make_heatmap(
                        subject=subject,
                        onset_col_idx=onset_col_idx,
                        colormap=colormap,
                        y_axis=y_axis,
                        filename=event,
                        save_path=save_here,
                        xticklabel=x_tick,
                        yticklabel=y_tick,
                    )
                else:
                    if subjects[rec_name] is None:
                        continue
                    subject = pd.DataFrame(np.array(subjects[rec_name]).mean(axis=1), columns=col_axis)
                    if subject is None:
                        continue
                    auxfun_plotting.make_heatmap(
                        subject=subject,
                        onset_col_idx=onset_col_idx,
                        colormap=colormap,
                        y_axis=y_axis,
                        filename=event,
                        save_path=save_here,
                        xticklabel=x_tick,
                        yticklabel=y_tick,
                    )


def plot_heatmaps_paired(
    data_obj: dict,
    event_pair: list[str, str],
    save_path: str,
    per_recording: bool = False,
    responsive_only: bool = False,
    colormap: str = "inferno",
    z_score: bool = True,
    x_tick: int = 50,
    y_tick: int = 25,
):
    """Plots paired heatmaps for sorted neurons comparison (how the same neuron responed to self and parters' event)

    Args:
        data_obj: output of structurize_data function. Object containing ephys data structure
        event_pair: list of two keys to plot together, assumes [subject, partner] order e.g., ["freezing_subject", "freezing_partner"]
        save_path: path to which the plots will be saved. Additional subfolders will be created on per-recording basis
        per_recording: Toggles if the regression is done and plotted for all recorded neurons
            or on per-recording basis. Defaults to False.
        responsive_only: Toggles whether to plot for all recorded cells or only the ones deemed significantly responsive. Defaults to False
        colormap: matplotlib colormap used for heatmap plotting. Defaults to "viridis".
        z_score: required for compatibility. Always True
    Returns:
        Saves figures to desired location. If per-recording makes/uses folders with recording name
    """
    # TODO: Fix plotting firing rate (cmap min/max are hardcoded at the moment - makes no sense)
    pre_event = data_obj["bin_params"]["pre_event"]
    post_event = data_obj["bin_params"]["post_event"]
    bin_size = data_obj["bin_params"]["bin_size"]

    onset_col_idx = int(pre_event / bin_size)
    filename = event_pair[0] + "_" + event_pair[1]

    if per_recording:
        y_axis, subjects, partners = auxfun_plotting.load_dict(
            which="all_rats",
            z_score=z_score,
            data_obj=data_obj,
            event_pair=event_pair,
            single=False,
        )
        col_axis = np.around(np.arange(-abs(pre_event), post_event, bin_size), 2)

        for rec_name in subjects.keys():
            if subjects[rec_name] is None or partners[rec_name] is None:
                continue
            save_here = auxiliary.make_dir_save(save_path, rec_name)
            if responsive_only:
                if (
                    data_obj["responsive_units"][rec_name][event_pair[0]] is None
                    or data_obj["responsive_units"][rec_name][event_pair[1]] is None
                ):
                    continue

                responsive_units = np.unique(
                    data_obj["responsive_units"][rec_name][event_pair[0]]
                    + data_obj["responsive_units"][rec_name][event_pair[1]]
                )
                subject = pd.DataFrame(
                    np.array(subjects[rec_name]).mean(axis=1),
                    index=data_obj["unit_ids"][rec_name],
                    columns=col_axis,
                ).loc[responsive_units]
                partner = pd.DataFrame(
                    np.array(partners[rec_name]).mean(axis=1),
                    index=data_obj["unit_ids"][rec_name],
                    columns=col_axis,
                ).loc[responsive_units]

                auxfun_plotting.make_paired_heatmap(
                    subject=subject,
                    partner=partner,
                    event_pair=event_pair,
                    onset_col_idx=onset_col_idx,
                    colormap=colormap,
                    y_axis=y_axis,
                    filename=filename,
                    save_path=save_here,
                    xticklabel=x_tick,
                    yticklabel=y_tick,
                )
            else:
                subject = pd.DataFrame(np.array(subjects[rec_name]).mean(axis=1), columns=col_axis)
                partner = pd.DataFrame(np.array(partners[rec_name]).mean(axis=1), columns=col_axis)

                auxfun_plotting.make_paired_heatmap(
                    subject=subject,
                    partner=partner,
                    event_pair=event_pair,
                    onset_col_idx=onset_col_idx,
                    colormap=colormap,
                    y_axis=y_axis,
                    filename=filename,
                    save_path=save_here,
                    xticklabel=x_tick,
                    yticklabel=y_tick,
                )
    else:
        col_axis = np.around(np.arange(-abs(pre_event), post_event, bin_size), 2)
        if responsive_only:
            y_axis, subjects, partners = auxfun_plotting.load_dict(
                which="all_rats",
                z_score=z_score,
                data_obj=data_obj,
                event_pair=event_pair,
                single=False,
            )

            responsive_subjects = []
            responsive_partners = []

            for rec_name in subjects:
                if subjects[rec_name] is None or partners[rec_name] is None:
                    continue
                if (
                    data_obj["responsive_units"][rec_name][event_pair[0]] is None
                    or data_obj["responsive_units"][rec_name][event_pair[1]] is None
                ):
                    continue

                responsive_units = np.unique(
                    data_obj["responsive_units"][rec_name][event_pair[0]] 
                    + data_obj["responsive_units"][rec_name][event_pair[1]]
                    )

                temp_subject = [i.mean(axis=0) for i in subjects[rec_name]]
                temp_partner = [i.mean(axis=0) for i in partners[rec_name]]

                df_subs = pd.DataFrame(temp_subject, index=data_obj["unit_ids"][rec_name]).loc[responsive_units]
                df_partns = pd.DataFrame(temp_partner, index=data_obj["unit_ids"][rec_name]).loc[responsive_units]

                responsive_subjects.append(df_subs)
                responsive_partners.append(df_partns)

            subjects = pd.concat(responsive_subjects)
            partners = pd.concat(responsive_partners)

            subjects = subjects.reset_index(drop=True).set_axis(col_axis, axis=1).copy()
            partners = partners.reset_index(drop=True).set_axis(col_axis, axis=1).copy()
        else:
            y_axis, subjects, partners = auxfun_plotting.load_dict(
                which="mean_all_rats",
                z_score=z_score,
                data_obj=data_obj,
                event_pair=event_pair,
                single=False,
            )

        sorted_idx = (
            (subjects.iloc[:, onset_col_idx:] + partners.iloc[:, onset_col_idx:])
            .mean(axis=1)
            .sort_values()
            .index
        )
        subject = subjects.reindex(sorted_idx).reset_index(drop=True).copy()
        partner = partners.reindex(sorted_idx).reset_index(drop=True).copy()

        auxfun_plotting.make_paired_heatmap(
            subject=subject,
            partner=partner,
            event_pair=event_pair,
            onset_col_idx=onset_col_idx,
            colormap=colormap,
            y_axis=y_axis,
            filename=filename,
            save_path=save_path,
            xticklabel=x_tick,
            yticklabel=y_tick,
        )


def plot_nrns_per_structure(df: pd.DataFrame, save_path: str):
    """Plot number of responsive neurons per structure. Optional."""
    sns.set_style("darkgrid")
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))

    sns.barplot(
        x=df.index,
        y=df.values,
        palette="colorblind",
        legend=False,
        hue=df.index,
        )

    bars = ax.bar(df.index, df.values)

    plt.tight_layout(pad=2)

    ax.set_title("Per structure summary")
    ax.set_xlabel("Structure")
    ax.set_ylabel("# of neurons")
    ax.bar_label(bars)

    fig.savefig(os.path.join(save_path, "summary_per_structure.png"), dpi=300)
    fig.clf()


def plot_neurons_per_event_structure(
    data_folder: str | list,
    data_obj: dict,
    save_path: str,
    colormap: str = "inferno",
    per_recording: bool = False,
):
    """Creates a heatmap plot of structures vs events where intersection is a number of neurons from a specific structure
    encoding a specific event. Neurons that respond to many events are repeated.

    Args:
        data_folder: path to a folder containing all ephys data folders
        data_obj: output of structurize_data. Stored in ephys_data.pickle
        save_path: path to which the plots will be saved. Additional subfolders will be created on per-recording basis
        per_recording: Toggles if the regression is done and plotted for all recorded neurons or on per-recording basis.
    """
    data_folder = auxiliary.check_data_folder(data_folder)
    rec_names = data_obj["recording_names"]
    event_names = data_obj["event_names"]
    cols = data_obj["event_names"] + ["Structure"]

    if per_recording:
        for dir, rec_name in zip(data_folder, rec_names):
            df = pd.read_csv(
                    os.path.join(dir, "cluster_info_good.csv"),
                    index_col="cluster_id",
                    usecols=cols + ["cluster_id"],
                )
            df = df.loc[:, event_names + ["Structure"]].groupby("Structure").sum()
            
            auxfun_plotting.make_per_event_structure_plot(
                df=df,
                save_path=save_path,
                event_names=event_names,
                per_recording=per_recording,
                colormap=colormap,
                rec_name=rec_name,
                )
    else:
        df_list = []
        for dir in data_folder:
            df_list.append(
                pd.read_csv(
                    os.path.join(dir, "cluster_info_good.csv"), 
                    index_col="cluster_id", 
                    usecols=cols + ["cluster_id"],
                )
            )
        df = pd.concat(df_list).loc[:, event_names + ["Structure"]].groupby("Structure").sum()
        
        auxfun_plotting.make_per_event_structure_plot(
            df=df,
            save_path=save_path,
            event_names=event_names,
            colormap=colormap,
            per_recording=per_recording,
            )


def plot_nrns_per_structure(df: pd.DataFrame, save_path: str):
    """Plot number of responsive neurons per structure. Optional."""
    sns.set_style("darkgrid")
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))

    sns.barplot(
        x=df.index,
        y=df.values,
        palette="colorblind",
        legend=False,
        hue=df.index,
        )

    bars = ax.bar(df.index, df.values)

    plt.tight_layout(pad=2)

    ax.set_title("Per structure summary")
    ax.set_xlabel("Structure")
    ax.set_ylabel("# of neurons")
    ax.bar_label(bars)

    fig.savefig(os.path.join(save_path, "summary_per_structure.png"), dpi=300)
    fig.clf()
