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
    behavioral_pair: list[str] = None,
    behavior: str = None,
    single: bool = None
    ) -> tuple:
    """Aux for loading specific parts of the data_obj for plotting (electrophysiology_data.pickle)

    Args:
        which (str): options include 'mean_all_rats', 'mean_per_rat', and 'all_rats'

    Returns:
        y_axis (str): name of the plot axis
        subjects (dict/DataFrame): holding firing rate/z-score data for subjects' behavior
        partners (dict/DataFrame): holding firing rate/z-score data for partners' behavior    
    """
    if which == "mean_all_rats":
        if single:
            if z_score:
                y_axis = "z-score"
                subject = data_obj["mean_zscored_events_all_rats"][behavior]
            else:
                y_axis = "fr"
                subject = data_obj["mean_fr_events_all_rats"][behavior]
        else:
            if z_score:
                y_axis = "z-score"
                subjects = data_obj["mean_zscored_events_all_rats"][behavioral_pair[0]]
                partners = data_obj["mean_zscored_events_all_rats"][behavioral_pair[1]]
            else:
                y_axis = "fr"
                subjects = data_obj["mean_fr_events_all_rats"][behavioral_pair[0]]
                partners = data_obj["mean_fr_events_all_rats"][behavioral_pair[1]]
    if which == "mean_per_rat":
        if single:
            if z_score:
                y_axis = "z-score"
                subject = data_obj["mean_zscored_events_per_rat"][behavior]
            else:
                y_axis = "fr"
                subject = data_obj["mean_fr_events_per_rat"][behavior]
        else:
            if z_score:
                y_axis = "z-score"
                subjects = data_obj["mean_zscored_events_per_rat"][behavioral_pair[0]]
                partners = data_obj["mean_zscored_events_per_rat"][behavioral_pair[1]]
            else:
                y_axis = "fr"
                subjects = data_obj["mean_fr_events_per_rat"][behavioral_pair[0]]
                partners = data_obj["mean_fr_events_per_rat"][behavioral_pair[1]]
    if which == "all_rats":
        if single:
            if z_score:
                y_axis = "z-score"
                subject = data_obj["all_zscored_events_per_rat"][behavior]
            else:
                y_axis = "fr"
                subject = data_obj["all_fr_events_per_rat"][behavior]
        else:
            if z_score:
                y_axis = "z-score"
                subjects = data_obj["all_zscored_events_per_rat"][behavioral_pair[0]]
                partners = data_obj["all_zscored_events_per_rat"][behavioral_pair[1]]
            else:
                y_axis = "fr"
                subjects = data_obj["all_fr_events_per_rat"][behavioral_pair[0]]
                partners = data_obj["all_fr_events_per_rat"][behavioral_pair[1]]
    if single:
        return y_axis, subject
    elif single == False:
        return y_axis, subjects, partners
    
def make_paired_scatter(df, filename, behavioral_pair, ax_limit, save_path):
    """Auxiliary for making scatter plots
    """
    result = np.around(linregress([df["subject"], df["partner"]]), 2)
    result = f"slope = {result[0]}\nr_val = {result[2]}\np_val = {result[3]}"

    #Build figure
    sns.set_style("darkgrid")
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6), sharey=True)
    sns.set(style="ticks")
    sns.regplot(data=df, x="subject", y="partner", scatter=True, fit_reg=True, ci=0.95, scatter_kws={"color": "royalblue"}, 
                                                                    line_kws={"color": "orangered", "alpha": 0.8})
    
    #Plotting params
    ax.set_title(filename.capitalize())
    ax.set_xlabel(f"{behavioral_pair[0]}")
    ax.set_ylabel(f"{behavioral_pair[1]}")
    ax.set_xlim(ax_limit[0], ax_limit[1])
    ax.set_ylim(ax_limit[0], ax_limit[1])

    plt.text(ax_limit[0]+1, ax_limit[1]-2, result, fontsize=10, bbox=dict(boxstyle="round", facecolor="orangered", alpha=0.7))
    plt.axvline(0, c="black", ls="--", alpha=0.5)
    plt.axhline(0, c="black", ls="--", alpha=0.5)
    
    fig.savefig(os.path.join(save_path, f"{filename}_reg_plot.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

def make_heatmap(subject, onset_col_idx, colormap, y_axis, filename, save_path, xticklabel=50, yticklabel=25):
    """Auxiliary for making sinlge heatmap
    """   
    sorted_idx = subject.iloc[:,onset_col_idx:].mean(axis=1).sort_values().index
    subject = subject.reindex(sorted_idx).reset_index(drop=True).copy()

    #Build figure
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))#, sharey="all", sharex="all", gridspec_kw={"width_ratios": [0.80, 1]})

    sns.heatmap(subject, ax=ax, vmin=-3, vmax=3, center=0, xticklabels=xticklabel, yticklabels=yticklabel, cbar=True, cbar_kws={"label": y_axis}, cmap=colormap)
    
    ax.set_title(filename)
    ax.set_ylabel("Neurons")
    fig.text(0.5, 0.04, 'Seconds', ha='center')

    fig.savefig(os.path.join(save_path, f"{filename}_heatmaps.png"), dpi=300, bbox_inches="tight")
    plt.close(fig) 

def make_paired_heatmap(subject, partner, behavioral_pair, onset_col_idx, colormap, y_axis, filename, save_path, xticklabel=50, yticklabel=25):
    """Auxiliary for making heatmaps
    """   
    sorted_idx = (subject.iloc[:,onset_col_idx:] + partner.iloc[:,onset_col_idx:]).mean(axis=1).sort_values().index
    subject = subject.reindex(sorted_idx).reset_index(drop=True).copy()
    partner = partner.reindex(sorted_idx).reset_index(drop=True).copy()

    #Build figure
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 8), sharey="all", sharex="all", gridspec_kw={"width_ratios": [0.80, 1]})

    sns.heatmap(subject, ax=axes[0], vmin=-3, vmax=3, center=0, xticklabels=xticklabel, yticklabels=yticklabel, cbar=False, cmap=colormap)
    sns.heatmap(partner, ax=axes[1], vmin=-3, vmax=3, center=0, xticklabels=xticklabel, yticklabels=yticklabel, cbar_kws={"label": y_axis}, cmap=colormap)
    
    axes[0].set_title(f"{behavioral_pair[0]}")
    axes[0].set_ylabel("Neurons")
    axes[1].set_title(f"{behavioral_pair[1]}")
    fig.text(0.5, 0.04, 'Seconds', ha='center')

    fig.savefig(os.path.join(save_path, f"{filename}_heatmaps.png"), dpi=300, bbox_inches="tight")
    plt.close(fig) 

def make_common_matrix(df, save_path, per_animal=None, animal=None):
    """Auxiliary function for making a matrix plot of common responsive neurons between events

    Args:
        df (DataFrame): containing amount of neurons per event pair
        save_path (str): path to a folder where the plot should be saved
        per_animal (bool, optional): Toggles if to plot per animal or for all animals. Defaults to None.
        animal (str, optional): name of the animal. Defaults to None.
    """    
    mask = np.zeros_like(df, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True
    vmax = np.nanmax(df.to_numpy())
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
    sns.heatmap(df, ax=ax, vmin=0, vmax=vmax, cmap='inferno', annot=True, mask=mask)
    
    if per_animal==True:
        save_here = make_dir_save(save_path, animal)
        fig.savefig(os.path.join(save_here, f"{animal}_common_neurons.png"), dpi=300, bbox_inches="tight")
    else:
        fig.savefig(os.path.join(save_path, f"all_common_neurons.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
