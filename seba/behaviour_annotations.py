# behavior_annotations.py
from __future__ import annotations

import numpy as np
import pandas as pd
from pynwb import NWBFile
from pynwb.behavior import BehavioralEvents
from pynwb.misc import AnnotationSeries

def read_bvs(behaviour_path: str) -> pd.DataFrame:
    """
    Read BehaActive .bvs annotations and return a DataFrame indexed by video time (seconds).

    Input:
        behaviour_path: Path to the .bvs CSV-like file containing a column "[BehaViewSeq]".

    Output:
        DataFrame with:
            - index: timestamps in seconds (float)
            - column "behavior": string labels
    """
    try:
        df = pd.read_csv(behaviour_path)
    except FileNotFoundError:
        raise

    behav = df["[BehaViewSeq]"].str.split(" - ", expand=True)
    df = behav[0].str.split(":", expand=True)  # H:M:S:centiseconds (per original code)
    df = df.merge(behav[1], left_index=True, right_index=True)

    df = df.iloc[:, 0:4].astype(int)

    # Convert to milliseconds (preserving original multipliers)
    df.iloc[:, 0] *= 3600000  # hours -> ms
    df.iloc[:, 1] *= 60000    # minutes -> ms
    df.iloc[:, 2] *= 1000     # seconds -> ms
    df.iloc[:, 3] *= 10       # hundredths -> ms

    millis = df.iloc[:, 0:4].sum(axis=1) / 1000.0  # -> seconds

    events = pd.DataFrame(millis).join(behav[1])
    events.columns = ["timestamp", "behavior"]
    events = events.set_index("timestamp")
    return events


def align_bvs_to_probe(
    behaview_events_sec: pd.DataFrame,
    cam_TTL_path: str,
    fps_video: float = 30.0
) -> pd.DataFrame:
    """
    Align BehaView event times (video seconds) to probe time using cam_TTL.

    Args:
        behaview_events_sec: DataFrame with index=video timestamps (sec), column "behavior".
        cam_TTL_path: Path to text file with probe timestamps for each camera frame (sec).
        fps_video: Nominal video fps (default 30.0).

    Returns:
        DataFrame indexed by probe time (sec) with a "behavior" column.
    """
    cam_TTL_values = np.loadtxt(cam_TTL_path)

    # Map video times to nearest frame indices
    frame_idx = np.round(behaview_events_sec.index.values * fps_video).astype(int)
    frame_idx = np.clip(frame_idx, 0, len(cam_TTL_values) - 1)

    # Frame times on probe clock
    t_probe = cam_TTL_values[frame_idx]

    events_aligned = behaview_events_sec.copy()
    events_aligned.index = t_probe
    events_aligned.index.name = "t_probe"
    return events_aligned


def add_bvs_behavior_annotations_from_paths(
    nwb: NWBFile,
    behaviour_path: str,
    cam_TTL_path: str,
    fps_video: float = 30.0
) -> NWBFile:
    """
    Read .bvs annotations, align them to probe time via cam_TTL, and add to the NWB file.

    Creates:
        - processing module: name="behavior", description="behavior events aligned to probe via cam_TTL"
        - BehavioralEvents container: name="behavior"
        - AnnotationSeries: name="behavior_labels"
            data = aligned 'behavior' strings
            timestamps = probe-aligned times (float seconds)

    Returns:
        The same NWBFile (mutated), for fluent chaining.
    """
    behaview_events_sec = read_bvs(behaviour_path)
    bvs_events_probe_sec = align_bvs_to_probe(behaview_events_sec, cam_TTL_path, fps_video)

    beh_mod = nwb.create_processing_module(
        name="behavior", description="behavior events aligned to probe via cam_TTL"
    )

    beh = BehavioralEvents(name="behavior")

    behavior_series = AnnotationSeries(
        name="behavior_labels",
        data=bvs_events_probe_sec["behavior"].astype(str).tolist(),
        timestamps=bvs_events_probe_sec.index.values.astype(float),
        description=".bvs labels aligned to probe time",
    )
    beh.add_timeseries(behavior_series)
    beh_mod.add(beh)

    return nwb
