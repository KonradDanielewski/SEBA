"""
@author: K. Danielewski adapted from K. Kondrakiewicz
"""

import os

import numpy as np
import pandas as pd


def read_spikes(
    spks_dir: str,
    sampling_rate: float = 30000.0,
    read_only: str = "good",
) -> (list[np.array], list):
    """Reads spike times as saved by Phy2.

    Args:
        spks_dir: directory with spike times and sorting results
        sampling_rate: ephys recording sampling rate.
        read_only: which cluster type to read ("good" for single units, "mua" for multi unit or "noise" for shit).

    Returns:
        spks_ts: a list of numpy arrays, where all timestamps in each array are from one neuron
        units_id: neuron (cluster) ids
    """
    spks = np.load(os.path.join(spks_dir, "spike_times.npy"))
    clst = np.load(os.path.join(spks_dir, "spike_clusters.npy"))
    clst_group = pd.read_csv(os.path.join(spks_dir, "cluster_group.tsv"), sep="\t")

    units_id = np.array(clst_group.cluster_id[clst_group.group == read_only])  # those clusters contain selected type of units
    spks = 1 / sampling_rate * spks  # convert sample numbers to time stamps

    spks_ts = []
    for nrn in units_id:
        spks_ts.append(spks[clst == nrn])

    return spks_ts, units_id
