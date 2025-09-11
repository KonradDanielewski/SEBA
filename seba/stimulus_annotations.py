# stimulus_annotations.py
import numpy as np
from pynwb import NWBFile
from pynwb.misc import AnnotationSeries

def add_cs_onsets_from_path(nwb: NWBFile, cs_path: str, label: str = "CS") -> NWBFile:
    """
    Load CS onset times (probe seconds) from a text file and add them as a stimulus AnnotationSeries.
    """
    cs_times = np.loadtxt(cs_path)  # seconds
    cs_series = AnnotationSeries(
        name="CS_onset",
        data=[label] * int(len(cs_times)),
        timestamps=np.asarray(cs_times, dtype=float),
        description=f"{label} onset times (probe seconds)",
    )
    nwb.add_stimulus(cs_series)
    return nwb
