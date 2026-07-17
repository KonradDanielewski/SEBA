"""
@author: K. Danielewski

Publication-ready Plotly plotting functions for SEBA.
All figures are saved as SVG by default.

Requires: plotly, kaleido  (pip install plotly kaleido)
"""

import os

import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy.stats import linregress, t as t_dist

from seba.plotting import plot_data
from seba.utils import auxiliary


# ---------------------------------------------------------------------------
# Publication-ready style defaults
# ---------------------------------------------------------------------------

_FONT = dict(family="Arial", size=14, color="black")

_AXIS = dict(
    showline=True,
    linewidth=1.5,
    linecolor="black",
    mirror=True,
    ticks="outside",
    tickwidth=1.5,
    tickfont=dict(family="Arial", size=12),
    showgrid=False,
    zeroline=False,
)

_LAYOUT = dict(
    font=_FONT,
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=70, r=40, t=60, b=70),
)

# Colorblind-safe palette (Wong 2011)
_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442"]


def _save(fig: go.Figure, path: str, filename: str) -> None:
    """Write figure as SVG to path/filename.png."""
    fig.write_image(os.path.join(path, f"{filename}.svg"))


# ---------------------------------------------------------------------------
# PST plots
# ---------------------------------------------------------------------------

def _make_psth_figure(
    spike_trains: list,
    col_axis: np.ndarray,
    mean: np.ndarray,
    sem: np.ndarray,
    y_axis: str,
    title: str,
    ylimit: list[float, float],
) -> go.Figure:
    """Build a two-panel (raster / PSTH) Plotly figure for one unit."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.4, 0.6],
    )

    # --- Raster ---
    x_rast, y_rast = [], []
    for trial_idx, train in enumerate(spike_trains):
        x_rast.extend(train.tolist())
        y_rast.extend([trial_idx] * len(train))

    fig.add_trace(go.Scatter(
        x=x_rast, y=y_rast,
        mode="markers",
        marker=dict(symbol="line-ns-open", size=7, line=dict(width=1, color="black")),
        showlegend=False,
    ), row=1, col=1)

    # --- PSTH (mean ± SEM shaded band) ---
    fig.add_trace(go.Scatter(
        x=col_axis, y=mean + sem,
        mode="lines", line=dict(width=0), showlegend=False,
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=col_axis, y=mean - sem,
        mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(0,114,178,0.20)",
        showlegend=False,
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=col_axis, y=mean,
        mode="lines", line=dict(color=_COLORS[0], width=2),
        showlegend=False,
    ), row=2, col=1)

    # Event-onset lines (one shape per row via paper y-coordinates)
    for row_frac_y0, row_frac_y1 in [(0.44, 1.0), (0.0, 0.38)]:
        fig.add_shape(
            type="line", x0=0, x1=0,
            y0=row_frac_y0, y1=row_frac_y1, yref="paper", xref="x",
            line=dict(color="black", width=1, dash="dash"),
        )

    fig.update_layout(
        title=dict(text=title, font=_FONT),
        **_LAYOUT,
    )
    fig.update_xaxes(**_AXIS)
    fig.update_yaxes(**_AXIS)
    fig.update_yaxes(title_text="Trial #", row=1, col=1)
    fig.update_yaxes(title_text=y_axis, range=ylimit, row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)

    return fig


def plot_psths(
    data_obj: dict,
    save_path: str,
    responsive_only: bool = False,
    z_score: bool = True,
    ylimit: list[float, float] = [-1, 2.5],
) -> None:
    """Plot per-unit raster + PSTH for each event.

    Args:
        data_obj: output of structurize_data
        save_path: root save directory; per-recording and per-event subfolders are created
        responsive_only: plot only responsive units when True. Currently always uses responsive units
            (non-responsive units have no raster data). Kept for API compatibility.
        z_score: use z-score when True, raw firing rate when False
        ylimit: [lower, upper] y-axis limit for the PSTH panel
    Returns:
        SVG files saved under save_path/<rec_name>/<event>/
    """
    rec_names = data_obj["recording_names"]
    events = data_obj["event_names"]

    for event in events:
        for rec_name in rec_names:
            if data_obj["centered_spike_timestamps"][event][rec_name] is None:
                continue

            units_to_plot = (
                plot_data._responsive_ids(data_obj, rec_name, event)
                if responsive_only
                else list(data_obj["unit_ids"][rec_name])
            )
            if not units_to_plot:
                continue

            save_here = auxiliary.make_dir_save(auxiliary.make_dir_save(save_path, rec_name), event)

            for unit_id in units_to_plot:
                spike_trains, col_axis, mean, sem, y_axis = plot_data.get_psth_unit_data(
                    data_obj, event, rec_name, unit_id, z_score
                )
                fig = _make_psth_figure(spike_trains, col_axis, mean, sem, y_axis, event, ylimit)
                _save(fig, save_here, f"{event}_psth_{unit_id}")


def plot_psths_paired(
    data_obj: dict,
    event_pair: list[str, str],
    save_path: str,
    responsive_only: bool = False,
    z_score: bool = True,
    ylimit: list[float, float] = [-1, 2.5],
) -> None:
    """Plot overlaid PSTHs for a pair of events (e.g. subject vs partner) per unit.

    Args:
        data_obj: output of structurize_data
        event_pair: [subject_event, partner_event]
        save_path: root save directory
        responsive_only: restrict to units responsive to either event
        z_score: use z-score when True, raw firing rate when False
        ylimit: [lower, upper] y-axis limit
    Returns:
        SVG files saved under save_path/<rec_name>/<event_pair>/
    """
    rec_names = data_obj["recording_names"]
    filename = f"{event_pair[0]}_{event_pair[1]}"

    for rec_name in rec_names:
        if (
            data_obj["centered_spike_timestamps"][event_pair[0]][rec_name] is None
            or data_obj["centered_spike_timestamps"][event_pair[1]][rec_name] is None
        ):
            continue

        save_here = auxiliary.make_dir_save(auxiliary.make_dir_save(save_path, rec_name), filename)

        if responsive_only:
            units_to_plot = list(set(
                plot_data._responsive_ids(data_obj, rec_name, event_pair[0])
                + plot_data._responsive_ids(data_obj, rec_name, event_pair[1])
            ))
        else:
            units_to_plot = list(data_obj["unit_ids"][rec_name])

        for unit_id in units_to_plot:
            col_axis, mean_sub, sem_sub, mean_par, sem_par, y_axis = plot_data.get_paired_psth_unit_data(
                data_obj, event_pair, rec_name, unit_id, z_score
            )

            fig = go.Figure()

            for mean, sem, label, color in [
                (mean_sub, sem_sub, event_pair[0], _COLORS[0]),
                (mean_par, sem_par, event_pair[1], _COLORS[1]),
            ]:
                rgba_fill = f"rgba({int(color[1:3], 16)},{int(color[3:5], 16)},{int(color[5:7], 16)},0.20)"
                fig.add_trace(go.Scatter(
                    x=col_axis, y=mean + sem,
                    mode="lines", line=dict(width=0), showlegend=False,
                ))
                fig.add_trace(go.Scatter(
                    x=col_axis, y=mean - sem,
                    mode="lines", line=dict(width=0),
                    fill="tonexty", fillcolor=rgba_fill,
                    showlegend=False,
                ))
                fig.add_trace(go.Scatter(
                    x=col_axis, y=mean,
                    mode="lines", line=dict(color=color, width=2),
                    name=label,
                ))

            fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="black")
            fig.update_layout(
                title=dict(text=filename.replace("_", " ").capitalize(), font=_FONT),
                xaxis_title="Time (s)",
                yaxis_title=y_axis,
                yaxis_range=ylimit,
                legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)", bordercolor="black", borderwidth=1),
                **_LAYOUT,
            )
            fig.update_xaxes(**_AXIS)
            fig.update_yaxes(**_AXIS)

            _save(fig, save_here, f"{filename}_psth_{unit_id}")


# ---------------------------------------------------------------------------
# Regression scatter
# ---------------------------------------------------------------------------

def plot_lin_reg_scatter(
    data_obj: dict,
    event_pair: list[str, str],
    save_path: str,
    per_recording: bool = False,
    responsive_only: bool = False,
    ax_limit: list[float, float] = [-4, 4],
    z_score: bool = True,
) -> None:
    """Scatter plot with linear regression line and 95 % CI for a paired event.

    Args:
        data_obj: output of structurize_data
        event_pair: [subject_event, partner_event]
        save_path: root save directory
        per_recording: one plot per recording when True, combined otherwise
        responsive_only: restrict to responsive units
        ax_limit: [min, max] for both axes
        z_score: always True (kept for API compatibility)
    Returns:
        SVG file(s) saved to save_path
    """
    filename = f"{event_pair[0]}_{event_pair[1]}"
    datasets = plot_data.get_lin_reg_data(data_obj, event_pair, per_recording, responsive_only, z_score)

    for df, rec_name in datasets:
        x = df["subject"].values
        y = df["partner"].values
        result = linregress(x, y)

        x_fit = np.linspace(ax_limit[0], ax_limit[1], 300)
        y_fit = result.slope * x_fit + result.intercept

        # 95 % confidence band
        n = len(x)
        x_mean = x.mean()
        s_res = np.sqrt(((y - (result.slope * x + result.intercept)) ** 2).sum() / (n - 2))
        t_crit = t_dist.ppf(0.975, n - 2)
        ci = t_crit * s_res * np.sqrt(1 / n + (x_fit - x_mean) ** 2 / ((x - x_mean) ** 2).sum())

        fig = go.Figure()

        # CI band
        fig.add_trace(go.Scatter(
            x=x_fit, y=y_fit + ci,
            mode="lines", line=dict(width=0), showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=x_fit, y=y_fit - ci,
            mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(214,94,0,0.20)",
            showlegend=False,
        ))
        # Regression line
        fig.add_trace(go.Scatter(
            x=x_fit, y=y_fit,
            mode="lines", line=dict(color="#D55E00", width=2),
            name="fit",
        ))
        # Data points
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode="markers",
            marker=dict(color=_COLORS[0], size=6, opacity=0.7),
            name="units",
        ))

        # Reference lines
        for val in [0]:
            fig.add_hline(y=val, line_width=1, line_dash="dash", line_color="black", opacity=0.5)
            fig.add_vline(x=val, line_width=1, line_dash="dash", line_color="black", opacity=0.5)

        stats_text = (
            f"slope = {result.slope:.2f}<br>"
            f"r = {result.rvalue:.2f}<br>"
            f"p = {result.pvalue:.3g}"
        )
        fig.add_annotation(
            x=ax_limit[0] + 0.15 * (ax_limit[1] - ax_limit[0]),
            y=ax_limit[1] - 0.1 * (ax_limit[1] - ax_limit[0]),
            text=stats_text,
            showarrow=False,
            bgcolor="rgba(214,94,0,0.15)",
            bordercolor="#D55E00",
            borderwidth=1,
            font=dict(family="Arial", size=11),
            align="left",
        )

        title = filename.replace("_", " ").capitalize()
        if rec_name:
            title = f"{rec_name} — {title}"
        fig.update_layout(
            title=dict(text=title, font=_FONT),
            xaxis_title=event_pair[0],
            yaxis_title=event_pair[1],
            xaxis_range=ax_limit,
            yaxis_range=ax_limit,
            legend=dict(x=0.82, y=0.05, bgcolor="rgba(255,255,255,0.8)", bordercolor="black", borderwidth=1),
            **_LAYOUT,
        )
        fig.update_xaxes(**_AXIS)
        fig.update_yaxes(**_AXIS)

        dest = auxiliary.make_dir_save(save_path, rec_name) if rec_name else save_path
        _save(fig, dest, f"{filename}_reg_plot")


# ---------------------------------------------------------------------------
# Heatmaps
# ---------------------------------------------------------------------------

def _make_heatmap_figure(
    df: pd.DataFrame,
    col_axis: np.ndarray,
    onset_col_idx: int,
    y_axis: str,
    title: str,
    colormap: str,
) -> go.Figure:
    """Build a single heatmap figure."""
    # Map column ticks: show every ~500 ms
    step = max(1, len(col_axis) // 10)
    tick_vals = list(range(0, len(col_axis), step))
    tick_text = [f"{col_axis[i]:.1f}" for i in tick_vals]

    fig = go.Figure(go.Heatmap(
        z=df.values,
        colorscale=colormap,
        zmid=0,
        zmin=-3,
        zmax=3,
        colorbar=dict(title=dict(text=y_axis, side="right"), thickness=15),
        x=list(range(len(col_axis))),
        showscale=True,
    ))

    # Onset line
    fig.add_shape(
        type="line", x0=onset_col_idx, x1=onset_col_idx,
        y0=0, y1=1, yref="paper",
        line=dict(color="white", width=1.5, dash="dash"),
    )

    fig.update_layout(
        title=dict(text=title, font=_FONT),
        xaxis=dict(
            tickmode="array", tickvals=tick_vals, ticktext=tick_text,
            title="Time (s)", **_AXIS,
        ),
        yaxis=dict(title="Neurons", **_AXIS),
        **_LAYOUT,
    )

    return fig


def plot_heatmaps(
    data_obj: dict,
    save_path: str,
    per_recording: bool = False,
    responsive_only: bool = False,
    colormap: str = "RdBu_r",
    z_score: bool = True,
) -> None:
    """Plot single-event heatmaps of neural activity, sorted by post-onset mean.

    Args:
        data_obj: output of structurize_data
        save_path: root save directory
        per_recording: one plot per recording when True
        responsive_only: show only responsive units
        colormap: Plotly colorscale name
        z_score: use z-score (always True; kept for compatibility)
    Returns:
        SVG file(s) saved to save_path
    """
    events = data_obj["event_names"]

    for event in events:
        datasets = plot_data.get_heatmap_data(data_obj, event, per_recording, responsive_only, z_score)
        for df, col_axis, onset_col_idx, y_axis, rec_name in datasets:
            fig = _make_heatmap_figure(df, col_axis, onset_col_idx, y_axis, event, colormap)
            dest = auxiliary.make_dir_save(save_path, rec_name) if rec_name else save_path
            _save(fig, dest, f"{event}_heatmap")


def plot_heatmaps_paired(
    data_obj: dict,
    event_pair: list[str, str],
    save_path: str,
    per_recording: bool = False,
    responsive_only: bool = False,
    colormap: str = "RdBu_r",
    z_score: bool = True,
) -> None:
    """Plot side-by-side heatmaps for a paired event, sorted by combined post-onset activity.

    Args:
        data_obj: output of structurize_data
        event_pair: [subject_event, partner_event]
        save_path: root save directory
        per_recording: one plot per recording when True
        responsive_only: show only responsive units
        colormap: Plotly colorscale name
        z_score: use z-score (always True; kept for compatibility)
    Returns:
        SVG file(s) saved to save_path
    """
    filename = f"{event_pair[0]}_{event_pair[1]}"
    datasets = plot_data.get_paired_heatmap_data(data_obj, event_pair, per_recording, responsive_only, z_score)

    for subj_df, part_df, col_axis, onset_col_idx, y_axis, rec_name in datasets:
        step = max(1, len(col_axis) // 10)
        tick_vals = list(range(0, len(col_axis), step))
        tick_text = [f"{col_axis[i]:.1f}" for i in tick_vals]

        fig = make_subplots(
            rows=1, cols=2,
            shared_yaxes=True,
            horizontal_spacing=0.06,
            subplot_titles=[event_pair[0], event_pair[1]],
        )

        common_heatmap = dict(
            colorscale=colormap,
            zmid=0, zmin=-3, zmax=3,
            x=list(range(len(col_axis))),
            showscale=False,
        )

        fig.add_trace(go.Heatmap(z=subj_df.values, **common_heatmap), row=1, col=1)
        fig.add_trace(go.Heatmap(
            z=part_df.values,
            **{**common_heatmap, "showscale": True,
               "colorbar": dict(title=dict(text=y_axis, side="right"), thickness=15)},
        ), row=1, col=2)

        for col_num in [1, 2]:
            fig.add_shape(
                type="line", x0=onset_col_idx, x1=onset_col_idx,
                y0=0, y1=1, yref="paper", xref=f"x{'' if col_num == 1 else col_num}",
                line=dict(color="white", width=1.5, dash="dash"),
            )

        axis_opts = dict(tickmode="array", tickvals=tick_vals, ticktext=tick_text, **_AXIS)
        fig.update_xaxes(title_text="Time (s)", **axis_opts)
        fig.update_yaxes(title_text="Neurons", **_AXIS, col=1)
        fig.update_layout(**_LAYOUT)

        dest = auxiliary.make_dir_save(save_path, rec_name) if rec_name else save_path
        _save(fig, dest, f"{filename}_heatmap")


# ---------------------------------------------------------------------------
# Common neurons matrix
# ---------------------------------------------------------------------------

def plot_common_nrns_matrix(
    data_obj: dict,
    save_path: str,
    per_recording: bool = False,
    percent_all_neurons: bool = True,
    colormap: str = "inferno",
) -> None:
    """Heatmap matrix of common responsive neurons between event pairs.

    Args:
        data_obj: output of structurize_data
        save_path: root save directory
        per_recording: one plot per recording when True
        percent_all_neurons: show values as % of all neurons when True
        colormap: Plotly colorscale name
    Returns:
        SVG file(s) saved to save_path
    """
    datasets = plot_data.get_common_matrix_data(data_obj, per_recording, percent_all_neurons)
    events = data_obj["event_names"]

    for df, rec_name in datasets:
        # Mask upper triangle (show lower triangle only)
        z = df.values.copy().astype(float)
        mask = np.zeros_like(z, dtype=bool)
        mask[np.triu_indices_from(mask, k=1)] = True
        z[mask] = np.nan

        colorbar_title = "% neurons" if percent_all_neurons else "# neurons"

        fig = go.Figure(go.Heatmap(
            z=z,
            x=events,
            y=events,
            colorscale=colormap,
            zmin=0,
            zmax=np.nanmax(z),
            colorbar=dict(title=dict(text=colorbar_title, side="right"), thickness=15),
            text=np.where(mask, "", np.round(z, 1).astype(str)),
            texttemplate="%{text}",
            textfont=dict(size=11),
        ))

        title = f"{rec_name} — common neurons" if rec_name else "All recordings — common neurons"
        fig.update_layout(
            title=dict(text=title, font=_FONT),
            xaxis=dict(title="Event", **_AXIS),
            yaxis=dict(title="Event", autorange="reversed", **_AXIS),
            **_LAYOUT,
        )

        dest = auxiliary.make_dir_save(save_path, rec_name) if rec_name else save_path
        fname = f"{rec_name}_common_neurons" if rec_name else "all_common_neurons"
        _save(fig, dest, fname)


# ---------------------------------------------------------------------------
# Neurons per structure (bar)
# ---------------------------------------------------------------------------

def plot_nrns_per_structure(df: pd.DataFrame, save_path: str) -> None:
    """Bar chart of responsive neuron counts per brain structure.

    Args:
        df: Series/DataFrame with structure names as index and counts as values
        save_path: directory to save the SVG
    Returns:
        SVG saved to save_path/summary_per_structure.png
    """
    structures = df.index.tolist()
    values = df.values.tolist()

    colors = (_COLORS * ((len(structures) // len(_COLORS)) + 1))[:len(structures)]

    fig = go.Figure(go.Bar(
        x=structures,
        y=values,
        marker_color=colors,
        text=[str(int(v)) for v in values],
        textposition="outside",
        textfont=dict(size=11),
    ))

    fig.update_layout(
        title=dict(text="Neurons per structure", font=_FONT),
        xaxis=dict(title="Structure", **_AXIS),
        yaxis=dict(title="# neurons", **_AXIS),
        bargap=0.3,
        **_LAYOUT,
    )

    _save(fig, save_path, "summary_per_structure")


# ---------------------------------------------------------------------------
# Structure × event heatmap
# ---------------------------------------------------------------------------

def plot_neurons_per_event_structure(
    data_folder: str | list,
    data_obj: dict,
    save_path: str,
    colormap: str = "inferno",
    per_recording: bool = False,
) -> None:
    """Heatmap of neuron counts per brain structure × event.

    Args:
        data_folder: path(s) to ephys data directories
        data_obj: output of structurize_data
        save_path: root save directory
        colormap: Plotly colorscale name
        per_recording: one plot per recording when True
    Returns:
        SVG file(s) saved to save_path
    """
    from seba.utils.auxiliary import check_data_folder
    data_folder = check_data_folder(data_folder)
    event_names = data_obj["event_names"]

    # Shorten event names for readability (strip _onsets / _offsets suffix)
    short_names = ["_".join(e.split("_")[:2]) if ("onsets" in e or "offsets" in e) else e for e in event_names]

    datasets = plot_data.get_per_event_structure_data(data_folder, data_obj, per_recording)

    for df, rec_name in datasets:
        df = df.rename(columns=dict(zip(event_names, short_names)))

        fig = go.Figure(go.Heatmap(
            z=df.values,
            x=df.columns.tolist(),
            y=df.index.tolist(),
            colorscale=colormap,
            colorbar=dict(title=dict(text="# neurons", side="right"), thickness=15),
            text=df.values.round(0).astype(int).astype(str),
            texttemplate="%{text}",
            textfont=dict(size=11),
        ))

        title = f"{rec_name} — structure × event" if rec_name else "All recordings — structure × event"
        fig.update_layout(
            title=dict(text=title, font=_FONT),
            xaxis=dict(title="Event", **_AXIS),
            yaxis=dict(title="Structure", autorange="reversed", **_AXIS),
            **_LAYOUT,
        )

        dest = auxiliary.make_dir_save(save_path, rec_name) if rec_name else save_path
        fname = f"{rec_name}_per_event_structure" if rec_name else "per_event_structure"
        _save(fig, dest, fname)
