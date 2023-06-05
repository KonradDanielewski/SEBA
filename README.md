# SEBA - Simple Ephys-Behavior Analysis


SEBA is a small package that standardizes and automates exploratory analysis of electrophysiological data not only in combination with behavior but also any stimuli with recorded and synchronized timestamps. 

It's based on the output of https://github.com/jenniferColonell/ecephys_spike_sorting for ephys data and https://github.com/JingyiGF/HERBS for probe tracing from histological images.

SEBA can import behavior annotation data from BehaView and BORIS (DLC2Action and SIMBA will be added in future updates) and based on spike data and timestamps of behavior or stimuli it builds a comprehensive data structure which is then used in visualisation and further analysis.

Expected data structure is:
1. data_folder: 
   * ephys_recording_1
   * ephys_recording_2
   * ephys_recording_3 
   * ...

Each ephys_recording folder should contain results of an `sglx_multi_run_pipeline.py` run from `ecephys_spike_sorting`.

SEBA uses `HERBS` output to assign to each neuron obtained with `ecephys_spike_sorting` it's location and adds this information to `cluster_info_good.csv` - a file that contains only neurons labeled as good by both kilosort and the user.

For workflow example check `example_notebook`

# Installation:
`conda env create -n SEBA pip, git`
`git clone https://github.com/KonradDanielewski/SEBA`
`cd SEBA`
`pip install .`


Developed by Konrad Danielewski, partially based on script by Kacper Kondrakiewicz https://github.com/KacperKon/EphysAnalysis