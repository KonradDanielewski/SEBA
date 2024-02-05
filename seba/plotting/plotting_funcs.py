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
    per_animal: bool = False,
    percent_all_neurons: bool = True,
    ):
    """Function creating matrix plot of a perctage of common responsive neurons between pairs of events

    Args:
        data_obj: output of structurize_data. Stored in ephys_data.pickle
        save_path: path to which the plots will be saved. Additional subfolders will be created on per-animal basis
        per_animal: Toggles if the regression is done and plotted for all recorded neurons or on per-animal basis. Defaults to False.
        percent_all_neurons: Toggles whether to show values as percentage of all neurons or percentage of responsive neurons. Defaults to True.
    Returns:
        Saves a matrix of the amount of common responsive neurons across event pairs
    """    

    rec_names = data_obj["responsive_units"].keys()
    events = data_obj["responsive_units"][list(rec_names)[0]].keys()

    if per_animal:
        for rec_name in rec_names:
            df = pd.DataFrame(np.nan, index=events, columns=events)
            for event1, event2 in combinations_with_replacement(events, 2):
                if data_obj["responsive_units"][rec_name][event1] == None or data_obj["responsive_units"][rec_name][event2] == None:
                    continue
                df.loc[event2, event1] = len(np.intersect1d(
                    data_obj["responsive_units"][rec_name][event1],
                    data_obj["responsive_units"][rec_name][event2])
                    )
            
            if percent_all_neurons:
                nrns = len(data_obj["unit_ids"][rec_name])
                df = df/nrns*100
            
            auxfun_plotting.make_common_matrix(df, save_path, per_animal=per_animal, animal=rec_name)
    else:
        df = pd.DataFrame(0, index=events, columns=events)
                
        for rec_name in rec_names:
            for event1, event2 in combinations_with_replacement(events, 2):
                if data_obj["responsive_units"][rec_name][event1] == None or data_obj["responsive_units"][rec_name][event2] == None:
                    continue
                count = len(np.intersect1d(data_obj["responsive_units"][rec_name][event1], 
                                        data_obj["responsive_units"][rec_name][event2]))
                df.loc[event2, event1] += count

        if percent_all_neurons:
            nrns = []
            for rec_name in rec_names:
                nrns.append(len(data_obj["unit_ids"][rec_name]))
            df = df/sum(nrns)*100

        auxfun_plotting.make_common_matrix(df, save_path, per_animal=per_animal)

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
        save_path: path to which the plots will be saved. Additional subfolders will be created on per-animal basis
        responsive_only: Toggles whether to plot for all recorded cells or only the ones deemed significantly responsive. Defaults to False
        z_score: if to use z-score or raw firing rate. Defaults to True.
        ylimit: y_axis limit, first value is lower, second is upper bound. Defaults to [-1, 2.5].
    Returns:
        Saves plots to set location, creating subfolders on per-animal basis. Each plot is named using event name and neuron ID
    """    

    sns.set_palette("colorblind")

    pre_event = data_obj["bin_params"]["pre_event"]
    post_event = data_obj["bin_params"]["post_event"]
    bin_size = data_obj["bin_params"]["bin_size"]  

    rec_names = list(data_obj["responsive_units"].keys())
    behaviors = list(data_obj["responsive_units"][rec_names[0]].keys())

    unit_ids = data_obj["unit_ids"]
    responsive_ids = data_obj["responsive_units"]
    
    for behavior in behaviors:
        y_axis, subject = auxfun_plotting.load_dict(which="all_rats", z_score=z_score, data_obj=data_obj, behavior=behavior, single=True)
        for rec_name in rec_names:

            save_here = auxiliary.make_dir_save(save_path, rec_name)

            responsive_cells = set(responsive_ids[rec_name][behavior])
            cells_to_use = set(unit_ids[rec_name]).intersection(responsive_cells)

            for idx, unit_id in enumerate(unit_ids[rec_name]):
                if responsive_only  and unit_id not in cells_to_use:
                    continue
                #Prepare data for proper axes
                col_axis = np.around(np.arange(-abs(pre_event), post_event, bin_size), 2)
                df0 = pd.DataFrame(subject[rec_name][idx]).set_axis(col_axis, axis=1)
                df0 = df0.melt(var_name='Seconds', value_name=y_axis, ignore_index=False)

                #Build figure
                fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 8), sharex=True)
                axes = axes.flatten()

                axes[0].eventplot(data_obj["centered_spike_timestamps"][behavior][rec_name][idx], linelengths=0.7)            
                sns.lineplot(df0, ax=axes[1], x="Seconds", y=y_axis, errorbar="se")
                
                #Plottting params
                axes[0].set_ylabel("Trial #", fontsize=14)
                axes[0].set_title(behavior)
                axes[1].set_xlabel('Seconds', fontsize=14)
                axes[1].set_ylabel(y_axis, fontsize=14)
                axes[1].set_ylim([ylimit[0], ylimit[1]])

                save = auxiliary.make_dir_save(save_here, behavior)
                
                fig.savefig(os.path.join(save, f"{behavior}_psth_{str(unit_id)}.png"), dpi=100, bbox_inches="tight")
                plt.close(fig)


def plot_psths_paired(
    data_obj: dict,
    behavioral_pair: list,
    save_path: str,
    responsive_only: bool = False,
    z_score: bool = True,
    ylimit: list[float, float] = [-1, 2.5],
    ):
    """Function used for plotting psths using either z-score or raw firing rate

    Args:
        data_obj: output of structurize_data function. Object containing ephys data structure
        behavioral_pair: list of two keys to plot together, assumes [subject, partner] order e.g., ["freezing_subject", "freezing_partner"]
        save_path: path to which the plots will be saved. Additional subfolders will be created on per-animal basis
        responsive_only: Toggles whether to plot for all recorded cells or only the ones deemed significantly responsive. Defaults to False
        z_score: if to use z-score or raw firing rate. Defaults to True.
        ylimit: y_axis limit, first value is lower, second is upper bound. Defaults to [-1, 2.5].
    Returns:
        Saves plots to set location, creating subfolders on per-animal basis. Each plot is named using event name and neuron ID
    """    
    sns.set_palette("colorblind")

    pre_event = data_obj["bin_params"]["pre_event"]
    post_event = data_obj["bin_params"]["post_event"]
    bin_size = data_obj["bin_params"]["bin_size"]
    unit_ids = data_obj["unit_ids"]
    responsive_ids = data_obj["responsive_units"]

    filename = behavioral_pair[0] + "_" + behavioral_pair[1]

    y_axis, subjects, partners = auxfun_plotting.load_dict(which="all_rats", z_score=z_score, data_obj=data_obj, behavioral_pair=behavioral_pair, single=False)

    for rec_name in unit_ids.keys():
        save_here = auxiliary.make_dir_save(save_path, rec_name)
        if responsive_only:
            responsive_cells = list(set(responsive_ids[rec_name][behavioral_pair[0]] + responsive_ids[rec_name][behavioral_pair[1]]))

        for idx, unit_id in enumerate(unit_ids[rec_name]):
            if responsive_only and unit_id not in responsive_cells:
                continue

            if subjects[rec_name] == None or partners[rec_name] == None:
                continue
            
            #Prepare data for proper axes
            col_axis = np.around(np.arange(-abs(pre_event), post_event, bin_size), 2)
            df0 = pd.DataFrame(subjects[rec_name][idx]).set_axis(col_axis, axis=1)
            df1 = pd.DataFrame(partners[rec_name][idx]).set_axis(col_axis, axis=1)
            df0 = df0.melt(var_name='Seconds', value_name=y_axis, ignore_index=False)
            df1 = df1.melt(var_name='Seconds', value_name=y_axis, ignore_index=False)

            #Build figure
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6), sharey=True)

            sns.lineplot(df0, ax=ax, x="Seconds", y=y_axis, errorbar="se", label=f"{behavioral_pair[0]}")
            sns.lineplot(df1, ax=ax, x="Seconds", y=y_axis, errorbar="se", label=f"{behavioral_pair[1]}")

            #Plottting params
            ax.set_title(filename.capitalize())
            ax.set_xlabel('Seconds')
            ax.set_ylabel(y_axis)
            ax.set_ylim([ylimit[0], ylimit[1]])

            save = auxiliary.make_dir_save(save_here, filename)

            fig.savefig(os.path.join(save, f"{filename}_psth_{str(unit_id)}.png"), dpi=100, bbox_inches="tight")
            plt.close(fig)

def plot_lin_reg_scatter(
    data_obj: dict,
    behavioral_pair: list,
    save_path: str,
    per_animal: bool = False,
    responsive_only: bool = False,
    ax_limit=[-4, 4],
    z_score: bool = True,
    ):
    """Creates scatter plots with regression results showing linear relationship between neuron
    responses to subjects' and partners' behaviors

    Args:
        data_obj: output of structurize_data function. Object containing ephys data structure
        behavioral_pair: list of two keys to plot together, assumes [subject, partner] order e.g., ["freezing_subject", "freezing_partner"]
        save_path: path to which the plots will be saved. Additional subfolders will be created on per-animal basis
        responsive_only: Toggles whether to plot for all recorded cells or only the ones deemed significantly responsive. Defaults to False
        per_animal: Toggles if the regression is done and plotted for all recorded neurons
            or on per-animal basis. Defaults to False.
        ax_limit: axes limits, assumes 0 centered, even per axis. First value is x axis, second is y axis. Defaults to [-4, 4].
        z_score: required for compatibility. Always True
    Returns:
        Saves a scatter plot to a desired location with results of linear regression containing slope, r-value and p-value of the fit. 
    """    
    filename = behavioral_pair[0] + "_" + behavioral_pair[1]
    rec_names = list(data_obj["responsive_units"].keys())

    subjects, partners = auxfun_plotting.load_dict(which="mean_per_rat", z_score=z_score, data_obj=data_obj, behavioral_pair=behavioral_pair, single=False)[1:3] #y_axis not used

    if per_animal:
        for rec_name in rec_names:
            subject = subjects[rec_name].dropna().mean(axis=1).rename("subject")
            partner = partners[rec_name].dropna().mean(axis=1).rename("partner")
            
            if len(subject) == 0 or len(partner) == 0:
                continue

            if responsive_only :
                responsive_units = np.unique(data_obj["responsive_units"][rec_name][behavioral_pair[0]] + data_obj["responsive_units"][rec_name][behavioral_pair[1]])
                subject = subject.loc[responsive_units]
                partner = partner.loc[responsive_units]

            df = pd.concat([subject, partner], axis=1)
            #Make plots
            save_here = auxiliary.make_dir_save(save_path, rec_name)
            auxfun_plotting.make_paired_scatter(df, filename, behavioral_pair, ax_limit, save_here)

    else:
        sub_combined = []
        par_combined = []

        for rec_name in rec_names:         
            subject = subjects[rec_name].dropna().mean(axis=1).rename("subject")
            partner = partners[rec_name].dropna().mean(axis=1).rename("partner")

            if len(subject) == 0 or len(partner) == 0:
                continue

            if responsive_only:
                responsive_units = np.unique(data_obj["responsive_units"][rec_name][behavioral_pair[0]] + data_obj["responsive_units"][rec_name][behavioral_pair[1]])
                subject = subject.loc[responsive_units]
                partner = partner.loc[responsive_units]

            sub_combined.append(subject)
            par_combined.append(partner)

        subject = pd.concat(sub_combined).reset_index(drop=True)
        partner = pd.concat(par_combined).reset_index(drop=True)
        df = pd.concat([subject, partner], axis=1)

        #Make plots
        auxfun_plotting.make_paired_scatter(df, filename, behavioral_pair, ax_limit, save_path)



def plot_heatmaps(
    data_obj: dict,
    save_path: str,
    per_animal: bool = False,
    responsive_only: bool = False,
    colormap: str = "inferno",
    z_score: bool = True,
    x_tick: int = 50,
    y_tick: int = 25,
    ):
    """Plots paired heatmaps for sorted neurons comparison (how the same neuron responed to self and parters' behavior)

    Args:
        data_obj: output of structurize_data function. Object containing ephys data structure
        save_path: path to which the plots will be saved. Additional subfolders will be created on per-animal basis
        per_animal: Toggles if the regression is done and plotted for all recorded neurons
            or on per-animal basis. Defaults to False.
        responsive_only: Toggles whether to plot for all recorded cells or only the ones deemed significantly responsive. Defaults to False
        colormap: matplotlib colormap used for heatmap plotting. Defaults to "viridis".
        z_score: required for compatibility. Always True
    Returns:
        Saves figures to desired location. If per-animal makes/uses folders with animal name
    """
    pre_event = data_obj["bin_params"]["pre_event"]
    post_event = data_obj["bin_params"]["post_event"]
    bin_size = data_obj["bin_params"]["bin_size"]

    rec_names = list(data_obj["responsive_units"].keys())
    behaviors = list(data_obj["responsive_units"][rec_names[0]].keys())

    onset_col_idx = int(pre_event/bin_size)

    for behavior in behaviors:
        if per_animal == False:
            col_axis = np.around(np.arange(-abs(pre_event), post_event, bin_size), 2)
            if responsive_only :
                y_axis, subjects = auxfun_plotting.load_dict(which="all_rats", z_score=z_score, data_obj=data_obj, behavior=behavior, single=True)

                responsive_subject = []
                
                for rec_name in rec_names:
                    if subjects[rec_name] == None:
                        continue
                    responsive_units = data_obj["responsive_units"][rec_name][behavior]
                    temp_subject = [i.mean(axis=0) for i in subjects[rec_name]]
                    
                    df_subs = pd.DataFrame(temp_subject, index=data_obj["unit_ids"][rec_name]).loc[responsive_units]
                    responsive_subject.append(df_subs)

                subjects = pd.concat(responsive_subject)
                subject = subjects.reset_index(drop=True).set_axis(col_axis, axis=1).copy()
            else:
                y_axis, subject = auxfun_plotting.load_dict(which="mean_all_rats", z_score=z_score, data_obj=data_obj, behavior=behavior, single=True)

            auxfun_plotting.make_heatmap(subject, onset_col_idx, colormap, y_axis, behavior, save_path, x_tick, y_tick)      

        if per_animal :
            y_axis, subjects = auxfun_plotting.load_dict(which="all_rats", z_score=z_score, data_obj=data_obj, behavior=behavior, single=True)
            col_axis = np.around(np.arange(-abs(pre_event), post_event, bin_size), 2)

            for rec_name in rec_names:
                for behavior in behaviors:
                    
                    save_here = auxiliary.make_dir_save(save_path, rec_name)
                
                    if responsive_only :
                        responsive_units = pd.Series(data_obj["responsive_units"][rec_name][behavior], dtype="float64").unique()            
                        subject = (pd.DataFrame(np.array(subjects[rec_name]).mean(axis=1), index=data_obj["unit_ids"][rec_name], columns=col_axis)
                                .loc[responsive_units])

                        auxfun_plotting.make_heatmap(subject, onset_col_idx, colormap, y_axis, behavior, save_here, x_tick, y_tick)
                    else:           
                        subject = pd.DataFrame(np.array(subjects[rec_name]).mean(axis=1), columns=col_axis)

                        auxfun_plotting.make_paired_heatmap(subject, onset_col_idx, colormap, y_axis, behavior, save_here, x_tick, y_tick)

def plot_heatmaps_paired(
    data_obj: dict,
    behavioral_pair: list,
    save_path: str,
    per_animal: bool = False,
    responsive_only: bool = False,
    colormap: str = "inferno",
    z_score: bool = True,
    x_tick: int = 50,
    y_tick: int = 25,
    ):
    """Plots paired heatmaps for sorted neurons comparison (how the same neuron responed to self and parters' behavior)

    Args:
        data_obj: output of structurize_data function. Object containing ephys data structure
        behavioral_pair: list of two keys to plot together, assumes [subject, partner] order e.g., ["freezing_subject", "freezing_partner"]
        save_path: path to which the plots will be saved. Additional subfolders will be created on per-animal basis
        per_animal: Toggles if the regression is done and plotted for all recorded neurons
            or on per-animal basis. Defaults to False.
        responsive_only: Toggles whether to plot for all recorded cells or only the ones deemed significantly responsive. Defaults to False
        colormap: matplotlib colormap used for heatmap plotting. Defaults to "viridis".
        z_score: required for compatibility. Always True
    Returns:
        Saves figures to desired location. If per-animal makes/uses folders with animal name
    """
    # TODO: Fix handling of empty data when repsonsive_only=False
    pre_event = data_obj["bin_params"]["pre_event"]
    post_event = data_obj["bin_params"]["post_event"]
    bin_size = data_obj["bin_params"]["bin_size"]

    onset_col_idx = int(pre_event/bin_size)
    filename = behavioral_pair[0] + "_" + behavioral_pair[1]

    if per_animal:
        y_axis, subjects, partners = auxfun_plotting.load_dict(which="all_rats", z_score=z_score, data_obj=data_obj, behavioral_pair=behavioral_pair, single=False)
        col_axis = np.around(np.arange(-abs(pre_event), post_event, bin_size), 2)

        for rec_name in subjects.keys():
            if subjects[rec_name] == None or partners[rec_name] == None:
                    continue
            save_here = auxiliary.make_dir_save(save_path, rec_name)
        
            if responsive_only:
                responsive_units = pd.Series(data_obj["responsive_units"][rec_name][behavioral_pair[0]] + data_obj["responsive_units"][rec_name][behavioral_pair[1]], dtype="float64").unique()            
                subject = (pd.DataFrame(np.array(subjects[rec_name]).mean(axis=1), index=data_obj["unit_ids"][rec_name], columns=col_axis)
                           .loc[responsive_units])
                partner = (pd.DataFrame(np.array(partners[rec_name]).mean(axis=1), index=data_obj["unit_ids"][rec_name], columns=col_axis)
                           .loc[responsive_units])

                auxfun_plotting.make_paired_heatmap(subject, partner, behavioral_pair, onset_col_idx, colormap, y_axis, filename, save_here, x_tick, y_tick)
            else:           
                subject = pd.DataFrame(np.array(subjects[rec_name]).mean(axis=1), columns=col_axis)
                partner = pd.DataFrame(np.array(partners[rec_name]).mean(axis=1), columns=col_axis)

                auxfun_plotting.make_paired_heatmap(subject, partner, behavioral_pair, onset_col_idx, colormap, y_axis, filename, save_here, x_tick, y_tick)
    else:
        col_axis = np.around(np.arange(-abs(pre_event), post_event, bin_size), 2)
        if responsive_only :
            y_axis, subjects, partners = auxfun_plotting.load_dict(which="all_rats", z_score=z_score, data_obj=data_obj, behavioral_pair=behavioral_pair, single=False)

            responsive_subjects = []
            responsive_partners = []

            for rec_name in subjects:
                if subjects[rec_name] == None or partners[rec_name] == None:
                    continue
                if data_obj["responsive_units"][rec_name][behavioral_pair[0]] == None or data_obj["responsive_units"][rec_name][behavioral_pair[1]] == None:
                    continue
                
                responsive_units = np.unique(data_obj["responsive_units"][rec_name][behavioral_pair[0]] + data_obj["responsive_units"][rec_name][behavioral_pair[1]])
                
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
            y_axis, subjects, partners = auxfun_plotting.load_dict(which="mean_all_rats", z_score=z_score, data_obj=data_obj, behavioral_pair=behavioral_pair, single=False)
        
        sorted_idx = (subjects.iloc[:,onset_col_idx:] + partners.iloc[:,onset_col_idx:]).mean(axis=1).sort_values().index
        subject = subjects.reindex(sorted_idx).reset_index(drop=True).copy()
        partner = partners.reindex(sorted_idx).reset_index(drop=True).copy()

        auxfun_plotting.make_paired_heatmap(subject, partner, behavioral_pair, onset_col_idx, colormap, y_axis, filename, save_path, x_tick, y_tick)

def plot_nrns_per_structure(df: pd.DataFrame, save_path: str):
    """Plot number of responsive neurons per structure. Optional.
    """    
    sns.set_style("darkgrid")
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    
    sns.barplot(
        x=df.index,
        y=df.values,
        palette="colorblind",
        legend=False,
        hue=df.index
        )
    
    bars = ax.bar(df.index, df.values)

    plt.tight_layout(pad=2)
    
    ax.set_title("Per structure summary")
    ax.set_xlabel("Structure")
    ax.set_ylabel("# of neurons")
    ax.bar_label(bars)    
    
    fig.savefig(os.path.join(save_path, "summary_per_structure.png"), dpi=300)
    fig.clf()

def plot_nrns_per_event(df: pd.DataFrame, save_path: str):
    """Plot number of responsive neurons per event. Optional.
    """    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 7))
    
    sns.heatmap(
        df,
        ax=ax,
        xticklabels=df.columns,
        yticklabels=df.index,
        vmin=0,
        vmax=df.max().max(),
        annot=True,
        cbar=True,
        cbar_kws={"label": "# of neurons"},
        cmap="viridis"
        )

    plt.tight_layout(pad=3)

    ax.set_title("Per structure summary") 

    fig.savefig(os.path.join(save_path, "summary_per_structure&behavior.png"), dpi=300)
    fig.clf()
