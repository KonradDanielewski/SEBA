"""
@author: K. Danielewski adapted from K. Kondrakiewicz
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


def read_spikes(spks_dir:str, sampling_rate = 30000.0, read_only = "good"):
    """
    Reads spike times as saved by Phy2.

    Args:
        spks_dir (str): directory with spike times and sorting results
        sampling_rate (float, optional): Defaults to 30000.0.
        read_only (str, optional): which cluster type to read ("good" for single units, "mua" for multi unit or "noise" for shit). Defaults to "good".

    Returns:
        spks_ts (list): a list of numpy arrays, where all timestamps in each array are from one neuron
        units_id (list): neuron ids
    """
    
    spks = np.load(spks_dir + "spike_times.npy")
    clst = np.load(spks_dir + "spike_clusters.npy")
    clst_group = pd.read_csv(spks_dir + "cluster_group.tsv", sep='\t')
    
    units_id = np.array(clst_group.cluster_id[clst_group.group == read_only]) # those clusters contain selected type of units
    spks = 1/sampling_rate * spks # convert sample numbers to time stamps

    spks_ts = []
    for nrn in units_id:
        spks_ts.append(spks[clst==nrn])
        
    return spks_ts, units_id

        

def calc_rasters(spks_ts:list, events_ts:str, pre_event = 1.0, post_event = 3.0):
    """
    Centers spikes' timestamps on events' timestamps.

    Args:
        spks_ts (list): list of arrays with neuron timestamps (each array is one neuron)
        events_ts (str): path to a file with event timestamps
        pre_event (float, optional): pre event time in seconds. Defaults to 1.0.
        post_event (float, optional): post event time in seconds. Defaults to 3.0.

    Returns:
        centered_ts(list): list of lists of centered arrays
    """    

    centered_ts = []

    events_ts = np.loadtxt(events_ts)
    
    for nrn in spks_ts:
        neuron_ts = []
        for evt in events_ts:
            pos = nrn - evt
            in_window = np.logical_and(pos >= -pre_event, pos <= post_event)
            pos = pos[in_window]
            neuron_ts.append(pos)
        centered_ts.append(neuron_ts)
        
    return centered_ts

def fr_events_binless(centered_ts:list, sigma_sec:float, trunc_gauss = 4, sampling_out = 1000, pre_event = 1.0, post_event = 3.0):
    """
    Calculates firing rates in trials by applying a Gaussian kernel (binless).

    Args:
        centered_ts (list): output of calc_raster function containing spike timestamps.
        sigma_sec (float): width of the Gaussian kernel in seconds.
        trunc_gauss (int, optional): the truncation range of the Gaussian, in units of standard deviation. Defaults to 4.
        sampling_out (int, optional): the desired output sampling rate for the firing rate calculation in Hz. Defaults to 1000.
        pre_event (float, optional): pre event time in seconds. Defaults to 1.0.
        post_event (float, optional): post event time in seconds. Defaults to 3.0.

    Returns:
        all_fr (list): list of arrays, for each neurons contains firing rate for each sample in n_trials x n_time_bins
        mean_fr (array): stores mean firing rate across trials
        sem_fr (array): stores standard error of the mean across trials
        t_vec (array): array of times of bins

    NOTE:
    The `sigma_sec` parameter sets the width of the Gaussian kernel used for smoothing the spike train. The
    `sampling_out` parameter sets the desired output sampling rate for the firing rate calculation. The `pre_event`
    and `post_event` parameters set the time window before and after the event for which the firing rate is
    calculated. The function returns three arrays for each neuron: `all_fr` contains firing rates for all trials,
    `mean_fr` contains the mean firing rate across trials, and `sem_fr` contains the standard error of the mean
    firing rate across trials. The function returns a 1D array of time values in `t_vec`.
    """    

    # Calculate how big is your data etc.
    nunits = len(centered_ts)
    ntrials = len(centered_ts[0])
    nsamples = int(np.round(sampling_out*pre_event + sampling_out*post_event))

    t_vec = np.linspace(-pre_event + 1/sampling_out, post_event, nsamples)


    # Create the gaussian window
    sigma = sigma_sec * sampling_out
    halfwidth = trunc_gauss*sigma # half-width of gaussian window - full width is halfwidth * 2 + 1
    
    gaussian = np.arange(-halfwidth, halfwidth + 1)
    gaussian = 1 / (np.sqrt(2 * np.pi) * sigma) * np.e ** (-np.power(gaussian / sigma, 2) / 2) * sampling_out
    #gaussian = np.exp(-(gaussian/sigma)**2/2) # a simpler formula - gives some weird scaling

    
    # Create empty list/arrays for storing results
    all_fr = []
    mean_fr = np.zeros([nunits, nsamples])
    sem_fr = np.zeros([nunits, nsamples])
    
    # Do the firing rate calculation (convolve binarized spikes with the gaussian)
    for nrn in range(nunits):
        neuron_fr = np.zeros([ntrials, nsamples])
        
        for trl in range(ntrials):
            
            where_spks = centered_ts[nrn][trl] + pre_event
            where_spks = np.array(np.round(where_spks*sampling_out), int) # find spike indices with the new sampling rate
            where_spks[where_spks == nsamples] = where_spks[where_spks == nsamples] - 1 # avoid rounding timestamps to indices bigger than data length

            neuron_fr[trl, where_spks] = 1 # code spikes as 1
            #neuron_fr[trl, :] = np.convolve(gaussian, neuron_fr[trl, :], 'same') # do the convoloution
            neuron_fr[trl, :] = gaussian_filter1d(neuron_fr[trl, :], sigma, mode = 'reflect') * sampling_out
            
        all_fr.append(neuron_fr)
        mean_fr[nrn,:] = np.mean(neuron_fr, 0)
        sem_fr[nrn,:] = np.std(neuron_fr, 0) / np.sqrt(ntrials)
    
    return all_fr, mean_fr, sem_fr, t_vec


def zscore_events(all_fr:list, bin_size:int, pre_event = 1.0, post_event = 3.0):
    """
    Calculates z-score in trials - where baseline is separate for each trial.

    Args:
        all_fr (list): list of arrays, for each neurons contains firing rate for each sample in n_trials x n_time_bins
        bin_size (int): bin size in seconds
        pre_event (float, optional): pre event time in seconds. Defaults to 1.0.
        post_event (float, optional): post event time in seconds. Defaults to 3.0.

    Returns:
        all_zsc (list): n trials x n time bins of calculated firing rate per neuron
        mean_zsc (array): n neurons x n time bins for storing mean fr (across trials)
        sem_zsc (array): n neurons x n time bins for storing standard error (across trials)
    """    
    
    # Calculate how big is your data and bin edges (in sec)
    nunits = len(all_fr)
    ntrials = all_fr[0].shape[0]
    bin_edges = np.arange(pre_event *(-1), post_event + bin_size, bin_size)
    nbins = bin_edges.size - 1
    nbins_pre = int(pre_event/bin_size)
    
    # Create empty list/arrays for storing results
    all_zsc = []
    mean_zsc = np.zeros([nunits, nbins])
    sem_zsc = np.zeros([nunits, nbins])
    
    # Do the z-score calculation
    for nrn in range(nunits):
        neuron_zsc = np.ones([ntrials, nbins])
        
        baseline_mean = np.mean(all_fr[nrn][:,0:nbins_pre], 1)
        baseline_std  = np.std(all_fr[nrn][:,0:nbins_pre], 1)
        # !!! What to do if standard deviation for a given bin == 0 (it's empty) or really small?
        # -> Current solution: set these values manually to 1
        baseline_std[baseline_std < 0.1] = 1
        
        for trl in range(ntrials):
            zsc_in_bins = (all_fr[nrn][trl,:] - baseline_mean[trl]) / baseline_std[trl] 
            neuron_zsc[trl, :] = zsc_in_bins[:]
            
        all_zsc.append(neuron_zsc)
        mean_zsc[nrn,:] = np.mean(neuron_zsc, 0)
        sem_zsc[nrn,:] = np.std(neuron_zsc, 0) / np.sqrt(ntrials)
    
    return all_zsc, mean_zsc, sem_zsc, bin_edges