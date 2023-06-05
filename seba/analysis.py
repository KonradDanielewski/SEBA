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

from seba.ephys import ephys as ep
#from zetapy import getIFR, getZeta


def read_bvs(filepath: str, framerate: int, save_path: str):
    """
    Reads BehaActive behavior annotation data from .bvs files

    Args:
        filepath (str): path to the .bvs file
        framerate (int): framerate of the video
        save_path (str): path to the folder containing animal ephys data

    Returns:
        Creates "events" folder in the save_path. Saves events_readout.csv that contains columns with a frame number and behavior. 
    """    
    
    df = pd.read_csv(filepath)
    behav = df["[BehaViewSeq]"].str.split(" - ", expand=True)
    df = behav[0].str.split(":", expand=True)
    df = df.merge(behav[1], left_index=True, right_index=True)
    
    df = df.iloc[:,0:4].astype(int)
    
    df.iloc[:, 0] *= 3600000
    df.iloc[:, 1] *= 60000
    df.iloc[:, 2] *= 1000
    df.iloc[:, 3] *= 10
    
    milis = df.iloc[:, 0:4].sum(axis=1)
    framerate = 1000/framerate
    milis = milis/framerate
    milis = milis.astype(int)

    events = pd.DataFrame(milis).join(behav[1])
    events.columns = ['frame', 'behavior']
    events = events.set_index("frame")
    if not os.path.exists(os.path.join(save_path, "events")):
        os.mkdir(os.path.join(save_path, "events"))
        save_path = os.path.join(save_path, "events")
    else:
        save_path = os.path.join(save_path, "events")
    events.to_csv(os.path.join(save_path, "events_readout.csv"))

def extract_raw_events_TS_BehaView(data_folder: str, framerate: int, append_CS=False, CS_filename=None, CS_len=None, padding=False, pad_size=None):
    """
    Divides all annotations to specific events with their timestamps synced to ephys data

    Args:
        data_folder (str): path to a folder containing all ephys data folders
        events_readout (str): path to events_readout.csv file
        camera_TTL (str): path to file containing camera frame timestamps
        framerate (int): framerate of the video
        append_CS (bool): whether to append conditioned stimulus onset timestamps. Defaults to False
        CS_len (int): length of the conditioned stimulus in seconds
        padding (bool): Toggles if event data should be 1-padded around it to make conditions more conservative. Applied to before onset and after offset
        pad_size (float): in seconds, how big the padding should be
    Returns:
        In animal folder creates events subfolder where it saves timestamps of onsets and offsets of events and events_filled_conditions.csv
        that contains per frame bool status of events
    """
    data_folder = glob(data_folder + "\\*")
    
    if pad_size is not None:
        pad = str(pad_size)
        pad = pad.replace(".", "_")
    
    for folder in data_folder:  
        cam_TS = pd.read_csv(os.path.join(folder, "cam_TTL.txt"), index_col="TS")
        cam_TTL = cam_TS.index
        events = pd.read_csv(os.path.join(folder, "events", "events_readout.csv"))
        event_names = events["behavior"].unique()

        df_conditions = pd.DataFrame(0, index=range(len(cam_TTL)), columns=event_names)

        for name in event_names:
            starts = events.query("behavior == @name")[::2]
            stops = events.query("behavior == @name")[1::2]

            starts_ser = pd.Series(cam_TTL, index=range(len(cam_TTL)))
            stops_ser = pd.Series(cam_TTL, index=range(len(cam_TTL)))
            starts_ser = starts_ser.iloc[starts["frame"]]
            stops_ser = stops_ser.iloc[stops["frame"]]

            if len(starts_ser) != len(stops_ser):
                stops_ser[len(stops_ser) + 1] = list(cam_TTL)[-1]

            #Save raw events without any conditions
            if len(starts_ser) > 0:
                np.savetxt(os.path.join(folder, f"events\\{name}_onsets.txt"), starts_ser, fmt='%1.6f')
            if len(stops_ser) > 0:
                np.savetxt(os.path.join(folder, f"events\\{name}_offsets.txt"), stops_ser, fmt='%1.6f')
            event_filled_preparation(df_conditions=df_conditions, starts_ser=starts_ser, stops_ser=stops_ser,
                                    framerate=framerate, padding=padding, pad_size=pad_size, method="behaview", name=name)

        df_conditions["TS"] = cam_TTL
        
        try:
            if append_CS == True and os.path.isfile(os.path.join(folder, CS_filename)):
                append_event(folder, CS_filename, cam_TTL, CS_len, df_conditions, framerate, padding, pad_size)
        except FileNotFoundError:
                print("The file CS.txt is not in the events folder. Please move the file to events folder and try again")

        output = df_conditions
        if padding == False:
            output.to_csv(os.path.join(folder, f"events\\events_filled_conditions.csv"), index_label="frame_id")
        else:
            output.to_csv(os.path.join(folder, f"events\\events_filled_conditions{pad}.csv"), index_label="frame_id") 

def read_extract_boris(data_folder:str, boris_output:str, camera_TTL:str, append_CS=False, CS_filename=None, CS_len=None, padding=False, pad_size=None):
    """
    Reads and extracts behavior annotations from Boris

    Args:
        data_folder (str): path to a folder containing all ephys data folders
        boris_output (str): path to output of Boris with behavior annotations
        camera_TTL (str): path to file containing camera frame timestamps 
        append_CS (bool, optional): whether to append conditioned stimulus onset timestamps. Defaults to False
        CS_len (int, optional): length of the conditioned stimulus in seconds
        padding (bool): Toggles if event data should be 1-padded around it to make conditions more conservative. Applied to before onset and after offset
        pad_size (float): in seconds, how big the padding should be
    """
    data_folder = glob(data_folder + "\\*")
    if pad_size is not None:
        pad = str(pad_size)
        pad = pad.replace(".", "_")
    for folder in data_folder:  
        df = pd.read_csv(os.path.join(folder, boris_output), skiprows = 15, usecols=["Time", "Total length", "FPS", 
                                                                            "Subject", "Behavior", "Status"])

        subjects = df["Subject"].unique()
        behaviors = df["Behavior"].unique()
        framerate = int(df["FPS"].unique())
        vid_len = range(int(df["Total length"].unique() * framerate))
        cam_TS = pd.read_csv(os.path.join(folder, camera_TTL), usecols=["TS"])
        cam_TTL = cam_TS["TS"]
        cols = []

        for subject in subjects:
            for behavior in behaviors:
                cols.append(f"{subject}_{behavior}")

        df_conditions = pd.DataFrame(0, index = vid_len, columns = cols)
        events_readout = pd.DataFrame(0, index = vid_len, columns = cols)

        if not os.path.exists(os.path.join(folder, "events")):
            os.mkdir(os.path.join(folder, "events"))
            save_path = os.path.join(folder, "events")
        else:
            save_path = os.path.join(folder, "events")

        for subject in subjects:
            for behavior in behaviors:
                starts_ser = df.query(f"Subject == '{subject}' & Behavior == '{behavior}' & Status == 'START'").copy()
                stops_ser = df.query(f"Subject == '{subject}' & Behavior == '{behavior}' & Status == 'STOP'").copy()

                starts_ser =  cam_TS.iloc[(starts_ser["Time"] * framerate).astype(int)]
                stops_ser = cam_TS.iloc[(stops_ser["Time"] * framerate).astype(int)]

                if len(starts_ser) != len(stops_ser):
                    stops_ser[len(stops_ser) + 1] = cam_TTL.iloc[-1]

                if len(starts_ser) > 0:
                    np.savetxt(os.path.join(save_path, f"{subject}_{behavior}_onsets.txt"), starts_ser, fmt='%1.6f')
                if len(stops_ser) > 0:
                    np.savetxt(os.path.join(save_path, f"{subject}_{behavior}_offsets.txt"), stops_ser, fmt='%1.6f')

                event_filled_preparation(df_conditions=df_conditions, starts_ser=starts_ser, stops_ser=stops_ser, 
                                         framerate=framerate, method="boris", padding=padding, 
                                         pad_size=pad_size, subject=subject, behavior=behavior)

        df_conditions["TS"] = cam_TTL
        try:
            if append_CS == True and os.path.isfile(os.path.join(folder, CS_filename)):     
                append_event(folder, CS_filename, cam_TTL, CS_len, df_conditions, framerate, padding, pad_size)
        except FileNotFoundError:
                print("The file CS.txt is not in the events folder. Please move the file to events folder and try again")

        # drop columns filled with zeros
        output = df_conditions.loc[:, (df_conditions != 0).any(axis=0)]
        output_raw = events_readout.loc[:, (df_conditions != 0).any(axis=0)]
        if padding == False:
            output.to_csv(os.path.join(save_path, f"events_filled_conditions.csv"), index_label="frame_id")
        else:
            output.to_csv(os.path.join(save_path, f"events_filled_conditions{pad}.csv"), index_label="frame_id")
        output_raw.to_csv(os.path.join(save_path, "events_readout.csv"), index_label="frame_id") 

def apply_conditions(data_folder:str, input_event:str, conditions:list, pad_size=None, exclusive=True):
    """
    Function used to apply conditions to events to take only independent instances of an event or only instances when events happened together

    Args:
        data_folder (str): path to a folder containing all ephys data folders
        input_event (str): name of the event file that conditions should be applied to, without file extension
        conditions (list): list of strings containing raw event names (uses names of columns in events_filled_conditions.csv) 
                           first has to be the same as the input_event
        pad_size (float): in seconds, specifies which evet_filled_conditions.csv should be used. Default None, no padding
        exclusive (bool, optional): Toggles if output should be for independent instances or instances when events happened together. Defaults to True.
    Returns:
        Text file saved into the events subfolder for each recorded animal.
    """
    data_folder = glob(data_folder + "\\*")
    if pad_size is not None:
        pad = str(pad_size)
        pad = pad.replace(".", "_")
    for folder in data_folder:    
        query_conditions = []
        if exclusive == True:
            for idx, condition in enumerate(conditions):
                if idx == 0:
                    query_conditions.append(f"{condition} == 1")
                else:
                    query_conditions.append(f"{condition} == 0")
        else:
            for condition in conditions:
                query_conditions.append(f"{condition} == 1")

        query_conditions = " and ".join(query_conditions)
        
        if pad_size == None:
            df = pd.read_csv(os.path.join(folder, "events", "events_filled_conditions.csv"), index_col="TS")
        else:
            df = pd.read_csv(os.path.join(folder, "events", f"events_filled_conditions{pad}.csv"), index_col="TS")
        conditional_event = np.loadtxt(os.path.join(folder, "events", f"{input_event}.txt"))
        condition_met = df.query(query_conditions).index
        output = np.intersect1d(conditional_event, condition_met)
        
        if not os.path.exists(os.path.join(folder, f"events\\conditions{pad}" if pad_size is not None else "events")):
            os.mkdir(os.path.join(folder, f"events\\conditions{pad}" if pad_size is not None else "events"))
            save_path = os.path.join(folder, f"events\\conditions{pad}" if pad_size is not None else "events")
        else:
            save_path = os.path.join(folder, f"events\\conditions{pad}" if pad_size is not None else "events")
        
        if exclusive == True:
            np.savetxt(os.path.join(save_path, input_event + "_exclusive.txt"), output, fmt='%1.6f')
        else:
            name = "_".join(conditions[1:])
            np.savetxt(os.path.join(save_path, input_event + "_" + name + "_together.txt"), output, fmt='%1.6f')

def structurize_data(data_folder:str, pad_size=None, pre_event=1.0, post_event=3.0, bin_size=0.01, save_output=True, save_path=None):
    """
    Creates a datastructure to be used for plotting and further analysis, saves it to pickle to not repeat computations.

    Args:
        data_folder (str): path to a folder containing all ephys data folders
        pre_event (float, optional): pre event time in seconds. Defaults to 1.0.
        post_event (float, optional): post event time in seconds. Defaults to 3.0.
        bin_size (float, optional): size of the bin used for firing rate calculation in seconds. Defaults to 0.01.
        save_output (bool, optional): saves output of the function to a pickle file. Defaults to True.
        save_path (str, optional): path to which the file should be saved. File will be named ephys_data.pickle.

    Returns:
        dict: contains fr(firing rate) and means of fr and z-score and its' means. Used as data structure for all following analyses
    """    

    data_folder = glob(data_folder + "\\*")
    if pad_size is not None:
        pad = str(pad_size)
        pad = pad.replace(".", "_")
    events = [os.path.splitext(filename)[0] for filename in os.listdir(os.path.join(data_folder[0],
                                                                       f"events\\conditions{pad}" if pad_size is not None else "events")) 
                                                                       if filename.endswith(".txt")]
    rats = [i.split("\\")[-1] for i in data_folder]

    mean_zscored_events_per_rat = {key: None for key in events}
    mean_zscored_events_all_rats = {key: None for key in events}
    mean_fr_events_per_rat = {key: None for key in events}
    mean_fr_events_all_rats = {key: None for key in events}

    all_zscored_events_per_rat = {key: None for key in events}
    all_fr_events_per_rat = {key: None for key in events}
    
    units_ids = {key: None for key in rats}

    centered_spike_timestamps = {key: None for key in events}
    spike_timestamps = {key: None for key in rats}

    for event in events:
        #Get all mean z-scores to combine into one df
        
        all_zscs = {key: None for key in rats}
        all_frs = {key: None for key in rats}
        centered_spike_timestamps[event] = {key: None for key in rats}
        
        mean_zscs = []
        mean_frs = []

        for folder, rat in zip(data_folder, rats):
            spks_dir = folder + "\\"
            which_to_use = f"events\\conditions{pad}" if pad_size is not None else "events"
            condition = os.path.join(folder, which_to_use, f"{event}.txt")
            
            #Take only spikes timestamps
            spikes_ts, unit_ids = ep.read_spikes(spks_dir)
            centered_ts = ep.calc_rasters(spikes_ts, condition, pre_event, post_event)
            sampling_out = 1/bin_size
            
            if units_ids.get(rat) is None:
                units_ids[rat] = unit_ids
            else:
                pass

            if spike_timestamps.get(rat) is None:
                spike_timestamps[rat] = spikes_ts
            else:
                pass

            #Take only all_fr and mean_fr
            all_fr, mean_fr = ep.fr_events_binless(centered_ts, 0.1, 4, 30000.0, sampling_out, pre_event, post_event)[0:2]
            
            #Take only all_zsc and mean_zsc
            all_zsc, mean_zsc = ep.zscore_events(all_fr, bin_size, pre_event, post_event)[0:2]
            rat = folder.split("\\")[-1]
            cols = [[rat], np.around(np.arange(-abs(pre_event), post_event, bin_size), 2)]
            index = pd.MultiIndex.from_product(cols, names=["rat", "time"])
            
            all_zscs[rat] = all_zsc
            mean_zsc = pd.DataFrame(mean_zsc, index=unit_ids, columns=index)
            mean_zscs.append(mean_zsc)

            all_frs[rat] = np.array(all_fr)
            mean_fr = pd.DataFrame(mean_fr, index=unit_ids, columns=index)
            mean_frs.append(mean_fr)

            centered_spike_timestamps[event][rat] = centered_ts
        
        #per rat dataframes with level 0 containing rat name
        per_rat_zscore = pd.concat(mean_zscs, axis=1)
        mean_zscored_events_per_rat[event] = per_rat_zscore
        per_rat_fr = pd.concat(mean_frs, axis=1)
        mean_fr_events_per_rat[event] = per_rat_fr
        
        #all rats combined, no unit id kept (for heatmaps, correlations)
        for i in range(len(mean_zscs)):
            mean_zscs[i] = mean_zscs[i].droplevel(0, axis=1)
        mean_zscored_rats = pd.concat(mean_zscs, axis=0)
        mean_zscored_rats = mean_zscored_rats.reset_index(drop=True)
        mean_zscored_events_all_rats[event] = mean_zscored_rats

        for i in range(len(mean_frs)):
            mean_frs[i] = mean_frs[i].droplevel(0, axis=1)
        mean_fr_rats = pd.concat(mean_frs, axis=0)
        mean_fr_rats = mean_fr_rats.reset_index(drop=True)
        mean_fr_events_all_rats[event] = mean_fr_rats

        all_fr_events_per_rat[event] = all_frs
        all_zscored_events_per_rat[event] = all_zscs
    
    bin_params = {"pre_event": pre_event,
                  "post_event": post_event,
                  "bin_size": bin_size,}

    output_keys = ["all_fr_events_per_rat", "all_zscored_events_per_rat", "mean_fr_events_all_rats", "mean_fr_events_per_rat", "mean_zscored_events_all_rats", "mean_zscored_events_per_rat", "centered_spike_timestamps", "spike_timestamps", "units_ids", "bin_params"]
    output_values = [all_fr_events_per_rat, all_zscored_events_per_rat, mean_fr_events_all_rats, mean_fr_events_per_rat, mean_zscored_events_all_rats, mean_zscored_events_per_rat, centered_spike_timestamps, spike_timestamps, units_ids, bin_params]
    output = dict(zip(output_keys, output_values))
    if save_output == True:
        with open(f"{save_path}\\ephys_data_{pad}.pickle" if pad_size is not None else f"{save_path}\\ephys_data.pickle", "wb") as handle:
            pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)


    return output

def get_responsive_units(data_folder:str, data_obj:dict, p_bound=0.05, pad_size=None):
    """
    Perform Wilcoxon signed-rank test to check which neurons responded significantly to specific condition/event

    Args:
        data_folder (str): path to a folder containing all ephys data folders
        data_obj (dict): output of structurize_data. Stored in ephys_data.pickle
        p_bound (float): p-value below which neurons are considered as significantly responsive
        pad_size: int if specified. Padding for event length used in creating the data structure
    Returns:
        responsive_units (dict): dict of responsive units that can be appended to ephys data structure
    """
    pre_event = data_obj["bin_params"]["pre_event"]
    post_event = data_obj["bin_params"]["post_event"]
    bin_size = data_obj["bin_params"]["bin_size"]

    data_folder = glob(data_folder+"\\*")
    if pad_size is not None:
        pad = str(pad_size)
        pad = pad.replace(".", "_")
    rats = [i.split("\\")[-1] for i in data_folder]
    conditions = [os.path.splitext(filename)[0] for filename in os.listdir(os.path.join(data_folder[0], 
                                                                                        f"events\\conditions{pad}" if pad_size is not None else "events")) 
                                                                                        if filename.endswith(".txt")]
    responsive_units = {key: None for key in rats}

    for rat in responsive_units:
        responsive_units[rat] = {key: [] for key in conditions}

    fdr_test = pd.Series(dtype="float64")

    for condition in data_obj["all_fr_events_per_rat"]:
        for rat in data_obj["all_fr_events_per_rat"][condition]:
            for idx, unit_id in enumerate(data_obj["units_ids"][rat]):
                col_axis = np.around(np.arange(-abs(pre_event), post_event, bin_size), 2)
                df = pd.DataFrame(data_obj["all_fr_events_per_rat"][condition][rat][idx]).set_axis(col_axis, axis=1)
                event = df.iloc[:, int(pre_event/bin_size):].mean(axis=1)
                baseline = df.iloc[:, :int(pre_event/bin_size)].mean(axis=1)
                #Take p-value only
                if any(event) == True and len(event) > 10:
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

def update_data_structure(data_obj:dict, responsive_units:dict, save_output=True, save_path=None, pad_size=None):
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
    if data_obj.get("responsive_units") and save_output == True:
        with open(f"{save_path}\\ephys_data_{pad}.pickle" if pad_size is not None else f"{save_path}\\ephys_data.pickle", "wb") as handle:
            pickle.dump(data_obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return data_obj

def responsive_neurons2events(data_folder:str, data_obj:dict):
    """
    Assigns 1 to neuron ids in cluster_info_good.csv that were significantly responsive to event

    Args:
        data_folder (str): path to a folder containing all ephys data folders
        data_obj (dict): output of structurize_data. Stored in ephys_data.pickle
    Returns:
        Overwrites existing cluster_info_good.csv with a new one containing bool columns of events with True assigned to responsive neurons
    """
        
    data_folder = glob(data_folder+"\\*")
    responsive_units = data_obj["responsive_units"]

    for folder, rat in zip(data_folder, responsive_units):
        df = pd.read_csv(os.path.join(folder, "cluster_info_good.csv"), index_col='id')
        col_names = responsive_units[rat].keys()
        for col_name in col_names:
            df[col_name] = np.nan
            df.loc[responsive_units[rat][col_name], col_name] = 1
        df.to_csv(os.path.join(folder, "cluster_info_good.csv"))

def event_filled_preparation(df_conditions, starts_ser, stops_ser, framerate, method, padding, pad_size=None, name=None, subject=None, behavior=None):
    """
    Aux function for creating events_filled_conditions
    """    
    if method == "behaview":
        if padding == True:
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
        if padding == True:
            pad = int(np.around(pad_size*framerate, 0))
            for start, stop in zip(list(starts_ser.index), list(stops_ser.index)):
                if start < pad:
                    df_conditions[f"{subject}_{behavior}"][0:stop+pad] = 1
                if stop+pad > len(df_conditions):
                    df_conditions[f"{subject}_{behavior}"][start-pad:len(df_conditions)] = 1
                else:
                    df_conditions[f"{subject}_{behavior}"][start-pad:stop+pad] = 1
            if len(starts_ser) > len(stops_ser):
                    df_conditions[name][starts_ser.index[-1]-pad:len(df_conditions)] = 1
        else:
            for start, stop in zip(list(starts_ser.index), list(stops_ser.index)):
                df_conditions[f"{subject}_{behavior}"][start:stop] = 1
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
    if padding == True:
        pad = int(np.around(pad_size*framerate, 0))
        for stimulus in CS_starts:
            df_conditions.loc[stimulus[0]-pad:stimulus[0]+(CS_len*framerate)+pad, "CS"] = 1
    else:
        for stimulus in CS_starts:
            df_conditions.loc[stimulus[0]:stimulus[0]+(CS_len*framerate), "CS"] = 1
