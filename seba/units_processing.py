# This script preprocesses our raw kilosort/phy data for the purpose of saving it as an .nwb file
import numpy as np
import pandas as pd
import os
from datetime import datetime, timezone
from pynwb import NWBFile

def create_nwbfile_from_bvs_path(behaviour_path: str) -> NWBFile:
    """
    Create an NWBFile using the session start time parsed from a .bvs filename.

    Expects filenames like "YYYY-MM-DD-HHMMSS.bvs" (e.g., "2020-08-15-093318.bvs"),
    where time is interpreted as UTC and made timezone-aware.

    Args:
        behaviour_path (str): Path to the .bvs file whose name encodes the session datetime.

    Returns:
        NWBFile: An NWBFile with:
            - session_description="SEBA session (spikes + behavior + CS)"
            - identifier=f"SEBA-{session_start.isoformat()}"
            - session_start_time set from the filename (UTC)

    Raises:
        ValueError: If the filename cannot be parsed into a date/time.
    """
    name = os.path.basename(behaviour_path)       # "2020-08-15-093318.bvs"
    stem = os.path.splitext(name)[0]              # "2020-08-15-093318"
    parts = stem.split("-")                       # ["2020","08","15","093318"]
    if len(parts) < 4:
        raise ValueError(f"Cannot parse date/time from: {behaviour_path}")
    date_str = "-".join(parts[:3])                # "2020-08-15"
    time_str = parts[3]                           # "093318"
    dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H%M%S")
    session_start = dt.replace(tzinfo=timezone.utc)

    nwb = NWBFile(
        session_description="SEBA session (spikes + behavior + CS)",
        identifier=f"SEBA-{session_start.isoformat()}",
        session_start_time=session_start,
    )
    return nwb

def populate_nwb_units_from_kilosort_files(
    nwb,
    spike_times_path: str,
    spike_clusters_path: str,
    sampling_rate: int,
    cluster_info_path: str,
    UNIT_COL_DESCRIPTIONS: dict | None = None,  # <-- now optional
) -> NWBFile:  # <-- you return nwb, so advertise it
    """
    Populate the NWB Units table from Kilosort/Phy spike & cluster files plus cluster_info metadata.

    Args:
        ...
        UNIT_COL_DESCRIPTIONS (dict[str, str] | None): Optional mapping of column -> description.
            If None or missing keys, falls back to "<col> (no description provided)".
    Returns:
        NWBFile: The same object passed in (mutated).
    """
    # Use empty dict if none provided
    descriptions = UNIT_COL_DESCRIPTIONS or {}

    # --- Load raw spike data ---
    spike_times = np.load(spike_times_path)
    spike_clusters = np.load(spike_clusters_path)

    # --- Build cluster_spike_times_dict ---
    cluster_spike_times_dict = {}
    for cluster in np.unique(spike_clusters):
        idx = np.where(spike_clusters == cluster)[0]
        cluster_spike_times_dict[int(cluster)] = spike_times[idx].flatten() / sampling_rate

    # --- pick_scalar_columns (inlined) ---
    df = pd.read_csv(cluster_info_path)
    meta_cols = []
    for col in df.columns:
        if col == "cluster_id":
            continue
        s = df[col].dropna()
        if len(s) == 0:
            meta_cols.append(col)
            continue
        if s.apply(lambda v: isinstance(v, (int, float, bool, str, np.generic))).all():
            meta_cols.append(col)

    # --- Declare those columns on the NWB Units table ---
    for col in meta_cols:
        desc = descriptions.get(col, f"{col} (no description provided)")
        nwb.add_unit_column(name=col, description=desc)

    # --- For quick lookup of metadata by cluster_id ---
    ci = df.set_index("cluster_id", drop=False)

    def to_python_scalar(v):
        if pd.isna(v):
            return None
        if isinstance(v, np.generic):
            return v.item()
        return v

    # --- Add each unit ---
    for cid in sorted(cluster_spike_times_dict.keys()):
        spikes = np.asarray(cluster_spike_times_dict[cid], dtype=float)
        kwargs = {}
        if cid in ci.index:
            row = ci.loc[cid]
            for col in meta_cols:
                val = row[col]
                if isinstance(val, (list, tuple, np.ndarray, pd.Series)):
                    continue
                kwargs[col] = to_python_scalar(val)
        nwb.add_unit(id=int(cid), spike_times=spikes.tolist(), **kwargs)

    return nwb

