"""
@author: K. Danielewski
"""

import os
from glob import glob

import numpy as np
import pandas as pd
from scipy.io import loadmat


def fix_wrong_shank_NP2(data_folder:str):
    """This function fixes wrong shank assignement in cluster_info. Relevant for NP2.0

    Args:
        data_folder (str): path to a directory containing all ephys data directories (here should be all dirs with recordings using neuropixels 2.0)
    Returns:
        Saves a new cluster_info.csv file that contains properly assigned shanks (doesn't overwrite old cluster_info.tsv)

    """

    try:
        if isinstance(data_folder, list):
            pass
        elif isinstance(data_folder, str):
            data_folder = glob(data_folder + "\\*")
    except:
        print(f"Passed data folder should be either string or a list, {type(data_folder)} was passed")

    for folder in data_folder:
        tmp = pd.read_csv(os.path.join(folder, r"cluster_info.tsv"), index_col="ch", sep="/t")
        obj = loadmat(os.path.join(folder, "chanMap.mat"))

        #Create temporary df to store shank and channel info, -1 needed to address matlab 1 based indexing
        channels = np.concatenate(obj["chanMap"]).astype(int) - 1
        shanks = np.concatenate(obj["kcoords"]).astype(int) - 1

        df = pd.DataFrame(data={"channels": channels, "shanks": shanks})
        df = df.set_index("channels")

        #Change bad shank assignement to nan and replace with correct
        tmp["sh"] = np.nan
        tmp["sh"] = tmp["sh"].fillna(df["shanks"])

        tmp.to_csv(os.path.join(folder, r"cluster_info.csv"))

def get_brain_regions(data_folder:str, histology_folder:str, neuropixels_20=True):
    """Adds a column to cluster_info with structure names from herbs matched to good units based on depth

    Args:
        data_folder (str): path to a directory containing all ephys data directories 
        histology_folder (str): path to directory containing directory per animal of herbs probe X.pkl files
        neuropixels_version (bool): default True, neuropixels version. Toggles NP shanks number (1 for 1.0, 4 for 2.0)
    Returns:
        Saves a new cluster_info_good.csv file that contains only the neurons that where deemed to be single units
    """    
    try:
        if isinstance(data_folder, list):
            pass
        elif isinstance(data_folder, str):
            data_folder = glob(data_folder + "\\*")
    except:
        print(f"Passed data folder should be either string or a list, {type(data_folder)} was passed")

    histology_folder = glob(histology_folder+"\\*")
    for cluster_info, histology in zip(data_folder, histology_folder):
        tmp = pd.read_csv(os.path.join(cluster_info, r"cluster_info.csv"))
        good_rows = tmp.query("group == 'good'").copy()

        if neuropixels_20 :
            shanks = range(4)
        else:
            shanks = range(1)

        all_shanks = []

        for shank in shanks:
            obj = pd.read_pickle(histology + f"\\probe {shank}.pkl")
            keys = pd.Series(obj["data"]["region_label"]).unique()
            values = pd.Series(obj["data"]["label_name"]).unique()

            areas = {keys[i]: values[i] for i in range(len(keys))}
            
            temporary = pd.Series(np.append(obj["data"]["sites_label"][0], obj["data"]["sites_label"][1]))
            region_label_per_row = []
            for key in keys:
                a = temporary.loc[temporary == key]
                region_label_per_row.append(a)
            region_label_per_row = pd.concat(region_label_per_row, ignore_index=True)
            if neuropixels_20 :
                #15 um is the distance between NP2.0 contacts, 20um for NP1.0
                if (len(region_label_per_row) % 2) == 0:
                    row_depth = np.arange(0, (obj["data"]["sites_label"][0].shape[0]*2)*15, 15) 
                else:
                    row_depth = np.arange(0, (obj["data"]["sites_label"][0].shape[0]*2)*15+1, 15)
            else:
                if (len(region_label_per_row) % 2) == 0:
                    row_depth = np.arange(0, (obj["data"]["sites_label"][0].shape[0]*2)*20, 20)
                else:
                    row_depth = np.arange(0, (obj["data"]["sites_label"][0].shape[0]*2)*20+1, 20)
            area_name = region_label_per_row.map(areas)

            df = pd.DataFrame(data={"depth": row_depth, "Structure":area_name})
            
            #Add "Structure" column to good_rows for comparison
            temp = good_rows.query("sh == @shank").copy()
            temp["Structure"] = np.nan
            temp = temp.set_index("depth")
            df = df.set_index("depth")
            temp["Structure"] = temp["Structure"].fillna(df["Structure"])
            if neuropixels_20 :
                all_shanks.append(temp)
            else:
                temp = (temp.set_index(temp["cluster_id"])
                        .sort_index()
                        .drop(columns=["cluster_id"]))
                temp.to_csv(os.path.join(cluster_info, r"cluster_info_good.csv"))
        if neuropixels_20 :
            output = pd.concat(all_shanks)
            output = (output.reset_index()
                    .set_index(output["cluster_id"])
                    .sort_index()
                    .drop(columns=["cluster_id"]))
            output.to_csv(os.path.join(cluster_info, r"cluster_info_good.csv"))
