# SEBA - Simple Ephys-Behavior Analysis


SEBA is a small package that standardizes and automates exploratory analysis of electrophysiological data not only in combination with behavior but also any stimuli with recorded and synchronized timestamps. 

It's based on the output of https://github.com/jenniferColonell/ecephys_spike_sorting for ephys data and https://github.com/JingyiGF/HERBS for probe tracing from histological images.

SEBA can import behavior annotation data from `BehaView` and `BORIS` (`DLC2Action` and `SimBA` will be added in future updates) and based on spike data and timestamps of behavior or stimuli it builds a comprehensive data structure which is then used in visualisation and further analysis.

Expected folder structure is:
1. data_folder: 
   * ephys_recording_1
   * ephys_recording_2
   * ephys_recording_3 
   * ...

Each ephys_recording folder should contain results of an `sglx_multi_run_pipeline.py` run from `ecephys_spike_sorting`.

SEBA uses `HERBS` output to assign to each neuron obtained with `ecephys_spike_sorting` it's location and adds this information to `cluster_info_good.csv` - a file that contains only neurons labeled as good by both kilosort and the user.

Examples of plots that can be created using the package. Users can choose between plotting data for all neurons or only responsive neurons, plot based on z-score or firing rate and for between plotting combined results for all animals or plot data per animal.

## Heatmaps of the neural acitvity per behavior:
<img src="https://github.com/KonradDanielewski/SEBA/assets/54865575/f43688b7-f997-4849-ba88-845ef705f7ef"  width="450" height="500">

## Comparison heatmaps (same neuron response between behaviors/stimuli):
<img src="https://github.com/KonradDanielewski/SEBA/assets/54865575/b650ea51-ee75-4261-a7eb-492e0de02c03"  width="450" height="500"> 

## Comparison PSTH plots:
<img src="https://github.com/KonradDanielewski/SEBA/assets/54865575/15a04763-66fa-4324-bac8-bc91c3db7eaf"  width="550" height="500"> 

## PSTH combined with a raster plot for resposivity verification:
<img src="https://github.com/KonradDanielewski/SEBA/assets/54865575/c58ced4b-7427-420b-964a-0ecac290f083"  width="500" height="500"> 

## Simple linear regression:
<img src="https://github.com/KonradDanielewski/SEBA/assets/54865575/c580548a-7ab0-438e-9557-81368f873815"  width="500" height="500"> 

## Summary heatmap structures vs behaviors/stimuli overlap (how many cells per structure encode a specific behavior):
<img src="https://github.com/KonradDanielewski/SEBA/assets/54865575/45f1f290-e975-4358-b5f1-6b124636fc57"  width="500" height="500"> 

## Matrix visualising overlap of neurons per behavior/stimulus:
<img src="https://github.com/KonradDanielewski/SEBA/assets/54865575/9bd5e311-a930-4d94-876c-676958ebf0fd"  width="500" height="500"> 

### For workflow example check `example_notebook`

# Installation:
`conda env create -n SEBA pip, git` \
`git clone https://github.com/KonradDanielewski/SEBA` \
`cd SEBA` \
`pip install .`


Developed by Konrad Danielewski, partially based on script by Kacper Kondrakiewicz https://github.com/KacperKon/EphysAnalysis
