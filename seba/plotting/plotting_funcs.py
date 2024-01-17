"""
@author: K. Danielewski
"""

import os
import pickle
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from seba.plotting import auxfun_ploting

def plot_common_nrns_matrix(data_obj: dict, save_path: str, per_animal=False, percent_all_neurons=True):
    """
    Function creating matrix plot of a perctage of common responsive neurons between pairs of events

    Args:
        data_obj (dict): output of structurize_data. Stored in ephys_data.pickle
        save_path (str): path to which the plots will be saved. Additional subfolders will be created on per-animal basis
        per_animal (bool, optional): Toggles if the regression is done and plotted for all recorded neurons or on per-animal basis. Defaults to False.
        percent_all_neurons (bool, optional): Toggles whether to show values as percentage of all neurons or percentage of responsive neurons. Defaults to True.
    Returns:
        Saves a matrix of the amount of common responsive neurons across event pairs
    """    

    animals = data_obj["responsive_units"].keys()
    events = data_obj["responsive_units"][list(animals)[0]].keys()

    if per_animal :
        for animal in animals:
            df = pd.DataFrame(np.nan, index=events, columns=events)
            for event in events:
                    
                for event2 in events:
                    per_event2[event2] = len(np.intersect1d(data_obj["responsive_units"][animal][event], data_obj["responsive_units"][animal][event2]))
                df[event] = per_event2
            
            if percent_all_neurons :
                nrns = len(data_obj["units_ids"][animal])
                df = df/nrns*100
            else:
                nrns = []
                for event in data_obj["responsive_units"][animal]:
                    nrns.append(data_obj["responsive_units"][animal][event])
                nrns = pd.Series(sum(nrns, []))
                nrns = list(nrns.unique())
                df = df/sum(nrns)*100
                
                for event in events:
                    nrns.append(len(data_obj["responsive_units"][animal][event]))
                df = df/sum(nrns)*100
            
            auxfun_ploting.make_common_matrix(df, save_path, per_animal=per_animal, animal=animal)

    if per_animal == False:
        df = pd.DataFrame(np.nan, index=events, columns=events)
        for animal in animals:
            temp = pd.DataFrame(np.nan, index=events, columns=events)
            for event in events:
                per_event2 = {key: None for key in events}
                for event2 in events:
                    per_event2[event2] = len(np.intersect1d(data_obj["responsive_units"][animal][event], data_obj["responsive_units"][animal][event2]))
                temp[event] = per_event2
            df = df.add(temp, fill_value=0)

        if percent_all_neurons :
            nrns = []
            for animal in animals:
                nrns.append(len(data_obj["units_ids"][animal]))
            df = df/sum(nrns)*100
        else: 
            nrns = []
            for subject in list(data_obj["responsive_units"].keys()):
                subject_responsive = []
                for behavior in list(data_obj["responsive_units"][subject].keys()):
                    responsive_units = data_obj["responsive_units"][subject][behavior]
                    subject_responsive.append(responsive_units)
                subject_responsive = sum(subject_responsive, [])
                subject_responsive = len(pd.Series(subject_responsive).unique())
                nrns.append(subject_responsive)
            nrns = sum(nrns)
            df = df/nrns*100

        auxfun_ploting.make_common_matrix(df, save_path, per_animal=per_animal)

def plot_psths(data_obj: dict, save_path: str, responsive_only=False, z_score=True, ylimit= [-1, 2.5]):
    """
    Function used for plotting psths using either z-score or raw firing rate

    Args:
        data_obj (dict): output of structurize_data function. Object containing ephys data structure
        save_path (str): path to which the plots will be saved. Additional subfolders will be created on per-animal basis
        responsive_only (bool, optional): Toggles whether to plot for all recorded cells or only the ones deemed significantly responsive. Defaults to False
        z_score (bool, optional): if to use z-score or raw firing rate. Defaults to True.
        ylimit (list, optional): y_axis limit, first value is lower, second is upper bound. Defaults to [2.5, 2.5].
    Returns:
        Saves plots to set location, creating subfolders on per-animal basis. Each plot is named using event name and neuron ID
    """    

    sns.set_palette("colorblind")

    pre_event = data_obj["bin_params"]["pre_event"]
    post_event = data_obj["bin_params"]["post_event"]
    bin_size = data_obj["bin_params"]["bin_size"]  

    subjects = list(data_obj["responsive_units"].keys())
    behaviors = list(data_obj["responsive_units"][subjects[0]].keys())

    unit_ids = data_obj["units_ids"]
    responsive_ids = data_obj["responsive_units"]
    
    for behavior in behaviors:
        y_axis, subject = auxfun_ploting.load_dict(which="all_rats", z_score=z_score, data_obj=data_obj, behavior=behavior, single=True)
        for animal in subjects:

            if not os.path.exists(os.path.join(save_path, animal)):
                os.mkdir(os.path.join(save_path, animal))
                save_here = os.path.join(save_path, animal)
            else:
                save_here = os.path.join(save_path, animal)

            responsive_cells = set(responsive_ids[animal][behavior])
            cells_to_use = set(unit_ids[animal]).intersection(responsive_cells)

            for idx, unit_id in enumerate(unit_ids[animal]):
                if responsive_only  and unit_id not in cells_to_use:
                    continue
                #Prepare data for proper axes
                col_axis = np.around(np.arange(-abs(pre_event), post_event, bin_size), 2)
                df0 = pd.DataFrame(subject[animal][idx]).set_axis(col_axis, axis=1)
                df0 = df0.melt(var_name='Seconds', value_name=y_axis, ignore_index=False)

                #Build figure
                fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 8), sharex=True)
                axes = axes.flatten()

                axes[0].eventplot(data_obj["centered_spike_timestamps"][behavior][animal][idx])            
                sns.lineplot(df0, ax=axes[1], x="Seconds", y=y_axis, errorbar="se")
                

                #Plottting params
                axes[0].set_ylabel("Trial #")
                axes[0].set_title(behavior)
                axes[1].set_xlabel('Seconds')
                axes[1].set_ylabel(y_axis)
                axes[1].set_ylim([ylimit[0], ylimit[1]])

                if not os.path.exists(os.path.join(save_here, behavior)):
                    os.mkdir(os.path.join(save_here, behavior))
                    save = os.path.join(save_here, behavior)
                else:
                    save = os.path.join(save_here, behavior)
                
                fig.savefig(os.path.join(save, f"{behavior}_psth_{str(unit_id)}.png"), dpi=100, bbox_inches="tight")
                plt.close(fig)


def plot_psths_paired(data_obj: dict, behavioral_pair: list, save_path: str, responsive_only=False, z_score=True, ylimit= [-1, 2.5]):
    """
    Function used for plotting psths using either z-score or raw firing rate

    Args:
        data_obj (dict): output of structurize_data function. Object containing ephys data structure
        behavioral_pair (list): list of two keys to plot together, assumes [subject, partner] order e.g., ["freezing_subject", "freezing_partner"]
        save_path (str): path to which the plots will be saved. Additional subfolders will be created on per-animal basis
        responsive_only (bool, optional): Toggles whether to plot for all recorded cells or only the ones deemed significantly responsive. Defaults to False
        z_score (bool, optional): if to use z-score or raw firing rate. Defaults to True.
        ylimit (list, optional): y_axis limit, first value is lower, second is upper bound. Defaults to [2.5, 2.5].
    Returns:
        Saves plots to set location, creating subfolders on per-animal basis. Each plot is named using event name and neuron ID
    """    
    sns.set_palette("colorblind")

    pre_event = data_obj["bin_params"]["pre_event"]
    post_event = data_obj["bin_params"]["post_event"]
    bin_size = data_obj["bin_params"]["bin_size"]  

    filename = behavioral_pair[0] + "_" + behavioral_pair[1]

    y_axis, subjects, partners = auxfun_ploting.load_dict(which="all_rats", z_score=z_score, data_obj=data_obj, behavioral_pair=behavioral_pair, single=False)

    unit_ids = data_obj["units_ids"]
    responsive_ids = data_obj["responsive_units"]

    for animal in unit_ids.keys():

        if not os.path.exists(os.path.join(save_path, animal)):
            os.mkdir(os.path.join(save_path, animal))
            save_here = os.path.join(save_path, animal)
        else:
            save_here = os.path.join(save_path, animal)

        responsive_cells = set(responsive_ids[animal][behavioral_pair[0]] + responsive_ids[animal][behavioral_pair[1]])
        cells_to_use = set(unit_ids[animal]).intersection(responsive_cells)

        for idx, unit_id in enumerate(unit_ids[animal]):
            if responsive_only  and unit_id not in cells_to_use:
                continue
            #Prepare data for proper axes
            col_axis = np.around(np.arange(-abs(pre_event), post_event, bin_size), 2)
            df0 = pd.DataFrame(subjects[animal][idx]).set_axis(col_axis, axis=1)
            df1 = pd.DataFrame(partners[animal][idx]).set_axis(col_axis, axis=1)
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

            if not os.path.exists(os.path.join(save_here, filename)):
                os.mkdir(os.path.join(save_here, filename))
                save = os.path.join(save_here, filename)
            else:
                save = os.path.join(save_here, filename)
            
            fig.savefig(os.path.join(save, f"{filename}_psth_{str(unit_id)}.png"), dpi=100, bbox_inches="tight")
            plt.close(fig)

def plot_lin_reg_scatter(data_obj: dict, behavioral_pair: list, save_path: str, per_animal=False, responsive_only=False, ax_limit=[-4, 4], z_score=True):
    """
    Creates scatter plots with regression results showing linear relationship between neuron
    responses to subjects' and partners' behaviors

    Args:
        data_obj (dict): output of structurize_data function. Object containing ephys data structure
        behavioral_pair (list): list of two keys to plot together, assumes [subject, partner] order e.g., ["freezing_subject", "freezing_partner"]
        save_path (str): path to which the plots will be saved. Additional subfolders will be created on per-animal basis
        responsive_only (bool): Toggles whether to plot for all recorded cells or only the ones deemed significantly responsive. Defaults to False
        per_animal (bool, optional): Toggles if the regression is done and plotted for all recorded neurons
            or on per-animal basis. Defaults to False.
        ax_limit (list, optional): axes limits, assumes 0 centered, even per axis. First value is x axis, second is y axis. Defaults to [-4, 4].
        z_score (bool): required for compatibility. Always True
    Returns:
        Saves a scatter plot to a desired location with results of linear regression containing slope, r-value and p-value of the fit. 
    """    
    filename = behavioral_pair[0] + "_" + behavioral_pair[1]

    subjects, partners = auxfun_ploting.load_dict(which="mean_per_rat", z_score=z_score, data_obj=data_obj, behavioral_pair=behavioral_pair, single=False)[1:3] #y_axis not used

    if per_animal == False:

        sub_combined = []
        par_combined = []

        for animal in subjects.columns.levels[0]:           
            subject = subjects[animal].dropna().mean(axis=1).rename("subject")
            partner = partners[animal].dropna().mean(axis=1).rename("partner")

            if responsive_only :
                responsive_units = pd.Series(data_obj["responsive_units"][animal][behavioral_pair[0]] + data_obj["responsive_units"][animal][behavioral_pair[1]], dtype="float64").unique()
                subject = subject.loc[responsive_units]
                partner = partner.loc[responsive_units]

            sub_combined.append(subject)
            par_combined.append(partner)

        subject = pd.concat(sub_combined).reset_index(drop=True)
        partner = pd.concat(par_combined).reset_index(drop=True)
        df = pd.concat([subject, partner], axis=1)

        #Make plots
        auxfun_ploting.make_paired_scatter(df, filename, behavioral_pair, ax_limit, save_path)

    if per_animal :
        for animal in subjects.columns.levels[0]:
            if not os.path.exists(os.path.join(save_path, animal)):
                save_here = os.mkdir(os.path.join(save_path, animal))
                save_here = os.path.join(save_path, animal)
            else:
                save_here = os.path.join(save_path, animal)                

            subject = subjects[animal].dropna().mean(axis=1).rename("subject")
            partner = partners[animal].dropna().mean(axis=1).rename("partner")

            if responsive_only :
                responsive_units = pd.Series(data_obj["responsive_units"][animal][behavioral_pair[0]] + data_obj["responsive_units"][animal][behavioral_pair[1]], dtype="float64").unique()
                subject = subject.loc[responsive_units]
                partner = partner.loc[responsive_units]

            df = pd.concat([subject, partner], axis=1)
            #Make plots
            auxfun_ploting.make_paired_scatter(df, filename, behavioral_pair, ax_limit, save_here)

def plot_heatmaps(data_obj: dict, save_path: str, per_animal=False, responsive_only=False, colormap = "inferno", z_score=True, x_tick=50, y_tick=25):
    """
    Plots paired heatmaps for sorted neurons comparison (how the same neuron responed to self and parters' behavior)

    Args:
        data_obj (dict): output of structurize_data function. Object containing ephys data structure
        save_path (str): path to which the plots will be saved. Additional subfolders will be created on per-animal basis
        per_animal (bool, optional): Toggles if the regression is done and plotted for all recorded neurons
            or on per-animal basis. Defaults to False.
        responsive_only (bool, optional): Toggles whether to plot for all recorded cells or only the ones deemed significantly responsive. Defaults to False
        colormap (str, optional): matplotlib colormap used for heatmap plotting. Defaults to "viridis".
        z_score (bool): required for compatibility. Always True
    Returns:
        Saves figures to desired location. If per-animal makes/uses folders with animal name
    """
    pre_event = data_obj["bin_params"]["pre_event"]
    post_event = data_obj["bin_params"]["post_event"]
    bin_size = data_obj["bin_params"]["bin_size"]

    subjects = list(data_obj["responsive_units"].keys())
    behaviors = list(data_obj["responsive_units"][subjects[0]].keys())

    onset_col_idx = int(pre_event/bin_size)

    for behavior in behaviors:
        if per_animal == False:
            col_axis = np.around(np.arange(-abs(pre_event), post_event, bin_size), 2)
            if responsive_only :
                y_axis, subjects = auxfun_ploting.load_dict(which="all_rats", z_score=z_score, data_obj=data_obj, behavior=behavior, single=True)

                responsive_subject = []

                for animal in subjects:
                    
                    responsive_units = pd.Series(data_obj["responsive_units"][animal][behavior], dtype="float64").unique()
                    temp_subject = []
                    for i in subjects[animal]:
                        meaned_subs = i.mean(axis=0)
                        temp_subject.append(meaned_subs)
                    
                    df_subs = pd.DataFrame(temp_subject, index=data_obj["units_ids"][animal]).loc[responsive_units]

                    responsive_subject.append(df_subs)

                subjects = pd.concat(responsive_subject)

                subject = subjects.reset_index(drop=True).set_axis(col_axis, axis=1).copy()
            else:
                y_axis, subject = auxfun_ploting.load_dict(which="mean_all_rats", z_score=z_score, data_obj=data_obj, behavior=behavior, single=True)

            auxfun_ploting.make_heatmap(subject, onset_col_idx, colormap, y_axis, behavior, save_path, x_tick, y_tick)      

        if per_animal :
            y_axis, subjects = auxfun_ploting.load_dict(which="all_rats", z_score=z_score, data_obj=data_obj, behavior=behavior, single=True)
            col_axis = np.around(np.arange(-abs(pre_event), post_event, bin_size), 2)

            for animal in subjects:
                for behavior in behaviors:
                    if not os.path.exists(os.path.join(save_path, animal)):
                        save_here = os.mkdir(os.path.join(save_path, animal))
                        save_here = os.path.join(save_path, animal)
                    else:
                        save_here = os.path.join(save_path, animal)
                
                    if responsive_only :
                        responsive_units = pd.Series(data_obj["responsive_units"][animal][behavior], dtype="float64").unique()            
                        subject = (pd.DataFrame(np.array(subjects[animal]).mean(axis=1), index=data_obj["units_ids"][animal], columns=col_axis)
                                .loc[responsive_units])

                        auxfun_ploting.make_heatmap(subject, onset_col_idx, colormap, y_axis, behavior, save_here, x_tick, y_tick)
                    else:           
                        subject = pd.DataFrame(np.array(subjects[animal]).mean(axis=1), columns=col_axis)

                        auxfun_ploting.make_paired_heatmap(subject, onset_col_idx, colormap, y_axis, behavior, save_here, x_tick, y_tick)

def plot_heatmaps_paired(data_obj: dict, behavioral_pair: list, save_path: str, per_animal=False, responsive_only=False, colormap = "inferno", z_score=True, x_tick=50, y_tick=25):
    """
    Plots paired heatmaps for sorted neurons comparison (how the same neuron responed to self and parters' behavior)

    Args:
        data_obj (dict): output of structurize_data function. Object containing ephys data structure
        behavioral_pair (list): list of two keys to plot together, assumes [subject, partner] order e.g., ["freezing_subject", "freezing_partner"]
        save_path (str): path to which the plots will be saved. Additional subfolders will be created on per-animal basis
        per_animal (bool, optional): Toggles if the regression is done and plotted for all recorded neurons
            or on per-animal basis. Defaults to False.
        responsive_only (bool, optional): Toggles whether to plot for all recorded cells or only the ones deemed significantly responsive. Defaults to False
        colormap (str, optional): matplotlib colormap used for heatmap plotting. Defaults to "viridis".
        z_score (bool): required for compatibility. Always True
    Returns:
        Saves figures to desired location. If per-animal makes/uses folders with animal name
    """
    pre_event = data_obj["bin_params"]["pre_event"]
    post_event = data_obj["bin_params"]["post_event"]
    bin_size = data_obj["bin_params"]["bin_size"]

    onset_col_idx = int(pre_event/bin_size)
    filename = behavioral_pair[0] + "_" + behavioral_pair[1]

    if per_animal == False:
        col_axis = np.around(np.arange(-abs(pre_event), post_event, bin_size), 2)
        if responsive_only :
            y_axis, subjects, partners = auxfun_ploting.load_dict(which="all_rats", z_score=z_score, data_obj=data_obj, behavioral_pair=behavioral_pair, single=False)

            responsive_subjects = []
            responsive_partners = []

            for animal in subjects:
                
                responsive_units = pd.Series(data_obj["responsive_units"][animal][behavioral_pair[0]] + data_obj["responsive_units"][animal][behavioral_pair[1]], dtype="float64").unique()
                temp_subject = []
                temp_partner = []
                for i, j  in zip(subjects[animal], partners[animal]):
                    meaned_subs = i.mean(axis=0)
                    meaned_partns = j.mean(axis=0)

                    temp_subject.append(meaned_subs)
                    temp_partner.append(meaned_partns)
                
                df_subs = pd.DataFrame(temp_subject, index=data_obj["units_ids"][animal]).loc[responsive_units]
                df_partns = pd.DataFrame(temp_partner, index=data_obj["units_ids"][animal]).loc[responsive_units]

                responsive_subjects.append(df_subs)
                responsive_partners.append(df_partns)

            subjects = pd.concat(responsive_subjects)
            partners = pd.concat(responsive_partners)

            subjects = subjects.reset_index(drop=True).set_axis(col_axis, axis=1).copy()
            partners = partners.reset_index(drop=True).set_axis(col_axis, axis=1).copy()
        else:
            y_axis, subjects, partners = auxfun_ploting.load_dict(which="mean_all_rats", z_score=z_score, data_obj=data_obj, behavioral_pair=behavioral_pair, single=False)
        
        sorted_idx = (subjects.iloc[:,onset_col_idx:] + partners.iloc[:,onset_col_idx:]).mean(axis=1).sort_values().index
        subject = subjects.reindex(sorted_idx).reset_index(drop=True).copy()
        partner = partners.reindex(sorted_idx).reset_index(drop=True).copy()

        auxfun_ploting.make_paired_heatmap(subject, partner, behavioral_pair, onset_col_idx, colormap, y_axis, filename, save_path, x_tick, y_tick)      

    if per_animal :
        y_axis, subjects, partners = auxfun_ploting.load_dict(which="all_rats", z_score=z_score, data_obj=data_obj, behavioral_pair=behavioral_pair, single=False)
        col_axis = np.around(np.arange(-abs(pre_event), post_event, bin_size), 2)

        for animal in subjects.keys():
            if not os.path.exists(os.path.join(save_path, animal)):
                save_here = os.mkdir(os.path.join(save_path, animal))
                save_here = os.path.join(save_path, animal)
            else:
                save_here = os.path.join(save_path, animal)
        
            if responsive_only :
                responsive_units = pd.Series(data_obj["responsive_units"][animal][behavioral_pair[0]] + data_obj["responsive_units"][animal][behavioral_pair[1]], dtype="float64").unique()            
                subject = (pd.DataFrame(np.array(subjects[animal]).mean(axis=1), index=data_obj["units_ids"][animal], columns=col_axis)
                           .loc[responsive_units])
                partner = (pd.DataFrame(np.array(partners[animal]).mean(axis=1), index=data_obj["units_ids"][animal], columns=col_axis)
                           .loc[responsive_units])

                auxfun_ploting.make_paired_heatmap(subject, partner, behavioral_pair, onset_col_idx, colormap, y_axis, filename, save_here, x_tick, y_tick)
            else:           
                subject = pd.DataFrame(np.array(subjects[animal]).mean(axis=1), columns=col_axis)
                partner = pd.DataFrame(np.array(partners[animal]).mean(axis=1), columns=col_axis)

                auxfun_ploting.make_paired_heatmap(subject, partner, behavioral_pair, onset_col_idx, colormap, y_axis, filename, save_here, x_tick, y_tick)

def neurons_per_structure(data_folder: str | list, data_obj: dict, save_path: str, plot=True):
    """
    Summary of a number of neurons recorded from each structure

    Args:
        data_folder (str): path to a folder containing all ephys data folders
        data_obj (dict): output of structurize_data function. Object containing ephys data structure
        save_path (path): path to which the plots and csv files will be saved
        plot (bool): default True, plots simple bar plot summarizing neurons per structure
    Returns:
        Saves histograms of a number of neurons per structure and csv files with the data 
    """
    try:
        if isinstance(data_folder, list):
            pass
        elif isinstance(data_folder, str):
            data_folder = glob(data_folder + "\\*")
    except:
        print(f"Passed data folder should be either string or a list, {type(data_folder)} was passed")

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

    if plot :
        sns.set_style("darkgrid")
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
        sns.barplot(x=df.index, y=df.values, palette="colorblind", legend=False)
        bars = ax.bar(df.index, df.values)

        plt.tight_layout(pad=2)
        
        ax.set_title("Per structure summary")
        ax.set_xlabel("Structure")
        ax.set_ylabel("# of neurons")
        ax.bar_label(bars)    
        
        fig.savefig(os.path.join(save_path, "summary_per_structure.png"), dpi=300)
        fig.clf()


def neurons_per_event(data_folder: str | list, data_obj:dict, save_path:str, plot=True):
    """
    Summary of a number of neurons per animal, event in a csv, creates csv for each structure
    NOTE: Neurons are repeated if a neuron is responsive to more than one behavior.
    
    Args:
        data_folder (str): path to a folder containing all ephys data folders
        data_obj (dict): output of structurize_data function. Object containing ephys data structure
        save_path (path): path to which the plots and csv files will be saved
        plot (bool): default True, if True creates simple bar plots per strucutre, x axis are events, y axis are neurons
    Returns:
        Saves histograms and data per event to desired location
    """
    try:
        if isinstance(data_folder, list):
            pass
        elif isinstance(data_folder, str):
            data_folder = glob(data_folder + "\\*")
    except:
        print(f"Passed data folder should be either string or a list, {type(data_folder)} was passed")

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

    if plot :
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 7))
        sns.heatmap(df, ax=ax, xticklabels=df.columns, yticklabels=df.index, vmin=0, vmax=df.max().max(), annot=True, cbar=True, cbar_kws={"label": "# of neurons"}, cmap="viridis")

        plt.tight_layout(pad=3)
        
        ax.set_title("Per structure summary") 
        
        fig.savefig(os.path.join(save_path, "summary_per_structure&behavior.png"), dpi=300)
        fig.clf()

        

            
