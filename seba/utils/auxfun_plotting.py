"""
@author: K. Danielewski
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import linregress

from seba.utils.auxiliary import make_dir_save


def load_dict(
    which: str,
    z_score: bool,
    data_obj: dict,
    event_pair: list[str, str] | None = None,
    event: str | None = None,
    single: bool | None = None,
) -> tuple:
    """Aux for loading specific parts of the data_obj for plotting (electrophysiology_data.pickle)

    Args:
        which (str): options include 'mean_all_rats', 'mean_per_rat', and 'all_rats'

    Returns:
        y_axis (str): name of the plot axis
        subjects (dict/DataFrame): holding firing rate/z-score data for subjects' event
        partners (dict/DataFrame): holding firing rate/z-score data for partners' event
    """
    if which == "mean_all_rats":
        if single:
            if z_score:
                y_axis = "z-score"
                subject = data_obj["mean_zscored_events_all_rats"][event]
            else:
                y_axis = "fr"
                subject = data_obj["mean_fr_events_all_rats"][event]
        else:
            if z_score:
                y_axis = "z-score"
                subjects = data_obj["mean_zscored_events_all_rats"][event_pair[0]]
                partners = data_obj["mean_zscored_events_all_rats"][event_pair[1]]
            else:
                y_axis = "fr"
                subjects = data_obj["mean_fr_events_all_rats"][event_pair[0]]
                partners = data_obj["mean_fr_events_all_rats"][event_pair[1]]
    if which == "mean_per_rat":
        if single:
            if z_score:
                y_axis = "z-score"
                subject = data_obj["mean_zscored_events_per_rat"][event]
            else:
                y_axis = "fr"
                subject = data_obj["mean_fr_events_per_rat"][event]
        else:
            if z_score:
                y_axis = "z-score"
                subjects = data_obj["mean_zscored_events_per_rat"][event_pair[0]]
                partners = data_obj["mean_zscored_events_per_rat"][event_pair[1]]
            else:
                y_axis = "fr"
                subjects = data_obj["mean_fr_events_per_rat"][event_pair[0]]
                partners = data_obj["mean_fr_events_per_rat"][event_pair[1]]
    if which == "all_rats":
        if single:
            if z_score:
                y_axis = "z-score"
                subject = data_obj["all_zscored_events_per_rat"][event]
            else:
                y_axis = "fr"
                subject = data_obj["all_fr_events_per_rat"][event]
        else:
            if z_score:
                y_axis = "z-score"
                subjects = data_obj["all_zscored_events_per_rat"][event_pair[0]]
                partners = data_obj["all_zscored_events_per_rat"][event_pair[1]]
            else:
                y_axis = "fr"
                subjects = data_obj["all_fr_events_per_rat"][event_pair[0]]
                partners = data_obj["all_fr_events_per_rat"][event_pair[1]]
    if single:
        return y_axis, subject
    elif single == False:
        return y_axis, subjects, partners


def make_paired_scatter(
    df,
    filename: str,
    event_pair: list[str, str],
    ax_limit: list[float, float],
    save_path: str,
):
    """Auxiliary for making scatter plots"""
    result = np.around(linregress([df["subject"], df["partner"]]), 2)
    result = f"slope = {result[0]}\nr_val = {result[2]}\np_val = {result[3]}"

    # Build figure
    sns.set_style("darkgrid")
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6), sharey=True)
    sns.set(style="ticks")
    sns.regplot(
        data=df,
        x="subject",
        y="partner",
        scatter=True,
        fit_reg=True,
        ci=0.95,
        scatter_kws={"color": "royalblue"},
        line_kws={"color": "orangered", "alpha": 0.8},
    )

    # Plotting params
    ax.set_title(filename.capitalize())
    ax.set_xlabel(f"{event_pair[0]}")
    ax.set_ylabel(f"{event_pair[1]}")
    ax.set_xlim(ax_limit[0], ax_limit[1])
    ax.set_ylim(ax_limit[0], ax_limit[1])

    plt.text(
        ax_limit[0] + 1,
        ax_limit[1] - 2,
        result,
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="orangered", alpha=0.7),
    )
    plt.axvline(0, c="black", ls="--", alpha=0.5)
    plt.axhline(0, c="black", ls="--", alpha=0.5)

    fig.savefig(os.path.join(save_path, f"{filename}_reg_plot.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_heatmap(
    subject,
    onset_col_idx: int,
    colormap: str,
    y_axis: str,
    filename: str,
    save_path: str,
    xticklabel: int = 50,
    yticklabel: int = 25,
):
    """Auxiliary for making sinlge heatmap"""
    sorted_idx = subject.iloc[:, onset_col_idx:].mean(axis=1).sort_values().index
    subject = subject.reindex(sorted_idx).reset_index(drop=True).copy()

    # Build figure
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))  # , sharey="all", sharex="all", gridspec_kw={"width_ratios": [0.80, 1]})

    sns.heatmap(
        subject,
        ax=ax,
        vmin=-3,
        vmax=3,
        center=0,
        xticklabels=xticklabel,
        yticklabels=yticklabel,
        cbar=True,
        cbar_kws={"label": y_axis},
        cmap=colormap,
    )

    ax.set_title(filename)
    ax.set_ylabel("Neurons")
    fig.text(0.5, 0.04, "Seconds", ha="center")

    fig.savefig(os.path.join(save_path, f"{filename}_heatmaps.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_paired_heatmap(
    subject,
    partner,
    event_pair: list[str, str],
    onset_col_idx: int,
    colormap: str,
    y_axis: str,
    filename: str,
    save_path: str,
    xticklabel: int = 50,
    yticklabel: int = 25,
):
    """Auxiliary for making heatmaps"""
    sorted_idx = (
        (subject.iloc[:, onset_col_idx:] + partner.iloc[:, onset_col_idx:])
        .mean(axis=1)
        .sort_values()
        .index
    )
    subject = subject.reindex(sorted_idx).reset_index(drop=True).copy()
    partner = partner.reindex(sorted_idx).reset_index(drop=True).copy()

    # Build figure
    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(8, 8),
        sharey="all",
        sharex="all",
        gridspec_kw={"width_ratios": [0.80, 1]},
    )

    sns.heatmap(
        subject,
        ax=axes[0],
        vmin=-3,
        vmax=3,
        center=0,
        xticklabels=xticklabel,
        yticklabels=yticklabel,
        cbar=False,
        cmap=colormap,
    )
    sns.heatmap(
        partner,
        ax=axes[1],
        vmin=-3,
        vmax=3,
        center=0,
        xticklabels=xticklabel,
        yticklabels=yticklabel,
        cbar_kws={"label": y_axis},
        cmap=colormap,
    )

    axes[0].set_title(f"{event_pair[0]}")
    axes[0].set_ylabel("Neurons")
    axes[1].set_title(f"{event_pair[1]}")
    fig.text(0.5, 0.04, "Seconds", ha="center")

    fig.savefig(os.path.join(save_path, f"{filename}_heatmaps.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_common_matrix(
    df,
    save_path: str,
    colormap: str,
    per_recording: bool | None = None,
    rec_name: str | None = None,
):
    """Auxiliary function for making a matrix plot of common responsive neurons between events""" # TODO: Add plot title
    mask = np.zeros_like(df, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True
    vmax = np.nanmax(df.to_numpy())
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
    sns.heatmap(df, ax=ax, vmin=0, vmax=vmax, cmap=colormap, annot=True, mask=mask)

    if per_recording:
        save_here = make_dir_save(save_path, rec_name)
        fig.savefig(os.path.join(save_here, f"{rec_name}_common_neurons.png"), dpi=300, bbox_inches="tight")
    else:
        fig.savefig(os.path.join(save_path, f"all_common_neurons.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_per_event_structure_plot(
    df,
    save_path: str,
    event_names: list[str, str],
    per_recording: bool,
    colormap: str,
    rec_name: str | None = None,
):
    """Auxfun for making a heatmap of structure which neurons are from corresponding to events they encode."""
    names = ["_".join(i.split("_")[:2]) for i in event_names if "onsets" in i or "offsets" in i]
    df.rename(columns={key: name for key, name in zip(event_names, names)})

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
    sns.heatmap(df, ax=ax, cmap=colormap, annot=True)

    if per_recording:
        save_here = make_dir_save(save_path, rec_name)
        fig.savefig(os.path.join(save_here, f"{rec_name}_per_event_structure.png"), dpi=300, bbox_inches="tight")
    else:
        fig.savefig(os.path.join(save_path, f"per_event_structure.png"), dpi=300, bbox_inches="tight")
