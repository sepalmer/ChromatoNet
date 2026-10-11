"""Plotting helpers for ChromatoNet simulations.

Typical usage example:

    fig, axes = plot_helper.plot_snapshots('my_run', n_frames=5)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from matplotlib import collections as mpl_collections
import matplotlib.pyplot as plt
import numpy as np
import torch

import calc_helper as ch

CHROMA_COLOR = '#CC2222'

# Colorblind-safe colors for probed chromatophores.
PROBE_COLORS = ('#000000', '#0173B2', '#DE8F05', '#029E73', '#D55E00',
                '#56636E', '#CC78BC', '#CA9161')

# Colorblind-safe colors for the muscles of a probed chromatophore.
MUSCLE_COLORS = ('#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9',
                 '#000000')


def _draw_chromatophores(ax, frame, radius_scale, radius_floor, color):
    """Draws one frame of chromatophores as circles sized by radius."""
    patches = [plt.Circle((c[0], c[1]), c[2] * radius_scale + radius_floor)
               for c in frame]
    ax.add_collection(mpl_collections.PatchCollection(
        patches, facecolors=color, edgecolors='none'))


def _limits(x, padding):
    """Returns plot limits around all chromatophore positions in a run."""
    return (np.nanmin(x[:, :, 0]) - padding, np.nanmax(x[:, :, 0]) + padding,
            np.nanmin(x[:, :, 1]) - padding, np.nanmax(x[:, :, 1]) + padding)


def _clean_axes(ax):
    """Removes the top and right spines."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_snapshots(
    name: str,
    n_frames: int = 5,
    times: Sequence[float] | None = None,
    start_time: float = 1.0,
    end_time: float | None = None,
    overlay: Callable[[plt.Axes], None] | None = None,
    radius_scale: float = 2.5,
    radius_floor: float = 0.07,
    padding: float = 0.3,
    color: str = CHROMA_COLOR,
):
    """Plots a row of snapshots of a saved 2D simulation.

    Args:
        name: The name passed to ChromatoNet.run().
        n_frames: Number of evenly spaced snapshots (used if times is None).
        times: Specific times to show; the nearest saved frame is used.
        start_time: Start of the range for evenly spaced snapshots.
        end_time: End of that range (defaults to the end of the run).
        overlay: Optional function ax -> None that draws on every panel,
            for example the outline of a stiff region.
        radius_scale: Circle radius is radius_scale * radius + radius_floor,
            so closed chromatophores still show as small dots.
        radius_floor: See radius_scale.
        padding: Space around the lattice.
        color: Chromatophore color.

    Returns:
        fig, axes
    """
    t, x, _, _ = ch.load_run(name)
    if times is None:
        t_end = t[-1] if end_time is None else min(end_time, t[-1])
        times = np.linspace(max(start_time, t[0]), t_end, n_frames)
    xl, xh, yl, yh = _limits(x, padding)

    n = len(times)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.0 * (yh - yl)
                                            / (xh - xl)), squeeze=False)
    axes = axes[0]
    for i, target in enumerate(times):
        frame_idx = int(np.argmin(np.abs(t - target)))
        ax = axes[i]
        _draw_chromatophores(ax, x[frame_idx], radius_scale, radius_floor,
                             color)
        if overlay is not None:
            overlay(ax)
        ax.set_xlim(xl, xh)
        ax.set_ylim(yl, yh)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        label = f'{t[frame_idx]:.0f}s'
        if i == 0:
            label = 't = ' + label
        ax.text(0.97, 0.97, label, transform=ax.transAxes, ha='right',
                va='top', fontsize=12, fontweight='bold', fontstyle='italic')
    fig.tight_layout()
    return fig, axes


def plot_chain_spacetime(
    name: str,
    n_rows: int = 20,
    start_time: float = 1.0,
    end_time: float | None = None,
    size_scale: float = 400.0,
    color: str = CHROMA_COLOR,
):
    """Plots a 1D chain run as a space-time diagram.

    Each row is one moment (time increases downward) and each dot is a
    chromatophore, sized by its radius.

    Args:
        name: The name passed to ChromatoNet.run().
        n_rows: Number of moments to show.
        start_time: First moment.
        end_time: Last moment (defaults to the end of the run).
        size_scale: Dot area scale.
        color: Chromatophore color.

    Returns:
        fig, ax
    """
    t, x, _, _ = ch.load_run(name)
    t_end = t[-1] if end_time is None else min(end_time, t[-1])
    fig, ax = plt.subplots(figsize=(9, 6))
    for target in np.linspace(max(start_time, t[0]), t_end, n_rows):
        idx = int(np.argmin(np.abs(t - target)))
        sizes = size_scale * (x[idx, :, 2] * 2.5 + 0.07)**2
        ax.scatter(x[idx, :, 0], np.full(x.shape[1], t[idx]), s=sizes,
                   color=color, edgecolors='none')
    ax.invert_yaxis()
    ax.set_xlabel('chromatophore position (x)')
    ax.set_ylabel('time (s)')
    _clean_axes(ax)
    return fig, ax


def plot_traces(
    net,
    name: str,
    chromat_indices: Sequence[int],
    start_time: float = 0.0,
    end_time: float | None = None,
):
    """Plots chromatophore radii and the voltages of their muscles.

    Each probed chromatophore gets one color, used for its dot on the layout
    panel, its radius trace and the voltage traces of all its muscles.
    Works for 1D chains and 2D lattices.

    Args:
        net: The ChromatoNet used for the run.
        name: The name passed to run().
        chromat_indices: Chromatophores to probe (up to 8).
        start_time: Start of the time window.
        end_time: End of the time window (defaults to the end of the run).

    Returns:
        fig, (ax_layout, ax_radius, ax_voltage)
    """
    t, x, v, z = ch.load_run(name)
    mask = t >= start_time
    if end_time is not None:
        mask &= t <= end_time
    t, x, v = t[mask], x[mask], v[mask]

    chain = ch.is_chain(z)
    if chain:
        fig = plt.figure(figsize=(10, 7))
        gs = fig.add_gridspec(3, 1, height_ratios=[0.6, 2, 2])
        ax_map = fig.add_subplot(gs[0])
        ax_r = fig.add_subplot(gs[1])
        ax_v = fig.add_subplot(gs[2], sharex=ax_r)
    else:
        fig = plt.figure(figsize=(13, 5.5))
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.6])
        ax_map = fig.add_subplot(gs[:, 0])
        ax_r = fig.add_subplot(gs[0, 1])
        ax_v = fig.add_subplot(gs[1, 1], sharex=ax_r)

    # Layout: all chromatophores in gray, probed ones in color.
    ax_map.scatter(z[:, 0], z[:, 1], s=15, color='lightgray', zorder=1)
    for idx, color in zip(chromat_indices, PROBE_COLORS):
        ax_map.scatter(z[idx, 0], z[idx, 1], s=150, color=color,
                       edgecolors='black', zorder=3)
        ax_map.annotate(str(idx), z[idx], xytext=(0, -12),
                        textcoords='offset points', ha='center', va='top')
        ax_r.plot(t, x[:, idx, 2], color=color, lw=2)
        for m_idx, _ in ch.connected_muscles(net, idx):
            ax_v.plot(t, v[:, m_idx, 2], color=color, lw=1.2, alpha=0.8)
    ax_map.set_aspect('equal')
    if chain:
        ax_map.set_ylim(-1.0, 0.5)
    ax_map.axis('off')

    ax_r.set_ylabel('radius')
    ax_v.set_ylabel('voltage')
    ax_v.set_xlabel('time (s)')
    plt.setp(ax_r.get_xticklabels(), visible=False)
    for ax in (ax_r, ax_v):
        _clean_axes(ax)
    fig.tight_layout()
    return fig, (ax_map, ax_r, ax_v)

def plot_probe(
    net,
    name: str,
    chromat_idx: int,
    window: float = 3.0,
    start_time: float = 0.0,
    end_time: float | None = None,
):
    """Plots one chromatophore, its muscles, its radius and muscle voltages.

    Left: the network around the chromatophore (red). Each of its muscles,
    and the chromatophore at the other end, has its own color. Right: the
    chromatophore's radius (top) and the voltage of each muscle in the
    matching color (bottom).

    Args:
        net: The ChromatoNet used for the run.
        name: The name passed to run().
        chromat_idx: Index of the chromatophore to probe.
        window: Half-width of the network panel around the chromatophore.
        start_time: Start of the time window.
        end_time: End of the time window (defaults to the end of the run).

    Returns:
        fig, (ax_network, ax_radius, ax_voltage)
    """
    t, x, v, z = ch.load_run(name)
    mask = t >= start_time
    if end_time is not None:
        mask &= t <= end_time
    t, x, v = t[mask], x[mask], v[mask]

    muscles = ch.connected_muscles(net, chromat_idx)
    ends = net.y_to_x[net.y_to_m].cpu().numpy()

    fig = plt.figure(figsize=(13, 5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.6])
    ax_net = fig.add_subplot(gs[:, 0])
    ax_r = fig.add_subplot(gs[0, 1])
    ax_v = fig.add_subplot(gs[1, 1], sharex=ax_r)

    # Network around the probed chromatophore: everything else in gray.
    for a, b in ends:
        ax_net.plot(z[[a, b], 0], z[[a, b], 1], color='0.85', lw=1,
                    zorder=1)
    ax_net.scatter(z[:, 0], z[:, 1], s=30, color='0.6', zorder=2)
    for i, (m_idx, other) in enumerate(muscles):
        color = MUSCLE_COLORS[i % len(MUSCLE_COLORS)]
        a, b = ends[m_idx]
        ax_net.plot(z[[a, b], 0], z[[a, b], 1], color=color, lw=4, zorder=3)
        ax_net.scatter(z[other, 0], z[other, 1], s=120, color=color,
                       edgecolors='black', zorder=4)
        ax_v.plot(t, v[:, m_idx, 2], color=color, lw=1.8,
                  label=f'muscle to #{other}')
    ax_net.scatter(z[chromat_idx, 0], z[chromat_idx, 1], s=250,
                   color=CHROMA_COLOR, edgecolors='black', zorder=5)
    ax_net.annotate(f'#{chromat_idx}', z[chromat_idx], xytext=(0, 14),
                    textcoords='offset points', ha='center',
                    fontweight='bold')
    ax_net.set_xlim(z[chromat_idx, 0] - window, z[chromat_idx, 0] + window)
    ax_net.set_ylim(z[chromat_idx, 1] - window, z[chromat_idx, 1] + window)
    ax_net.set_aspect('equal')
    ax_net.axis('off')

    ax_r.plot(t, x[:, chromat_idx, 2], color=CHROMA_COLOR, lw=2)
    ax_r.set_ylabel('radius')
    plt.setp(ax_r.get_xticklabels(), visible=False)
    ax_v.set_ylabel('voltage')
    ax_v.set_xlabel('time (s)')
    ax_v.legend(loc='upper right', fontsize=9, frameon=False)
    for ax in (ax_r, ax_v):
        _clean_axes(ax)
    fig.tight_layout()
    return fig, (ax_net, ax_r, ax_v)

def plot_network(net, show_gap_junctions: bool = True,
                 dot_size: float = 200.0, ax=None):
    """Draws a network at rest: chromatophores, muscles and gap junctions.

    Chromatophores are red, muscles solid blue lines, gap junctions dotted
    blue lines between the midpoints of the muscles they connect (the pairs
    counted in nc_m), and skin springs short green lines below each
    chromatophore.

    Args:
        net: A ChromatoNet (or ChromatoNet24).
        show_gap_junctions: Whether to draw gap junctions.
        dot_size: Chromatophore marker area.
        ax: Axes to draw on (a new figure is made if None).

    Returns:
        fig, ax
    """
    z = net.z.cpu().numpy()
    ends = net.y_to_x[net.y_to_m].cpu().numpy()
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))
    else:
        fig = ax.get_figure()

    for px, py in z:
        ax.plot([px, px], [py - 0.25, py - 0.05], color='green', lw=3,
                zorder=2)
    for a, b in ends:
        ax.plot(z[[a, b], 0], z[[a, b], 1], color='#0173B2', lw=1.5,
                alpha=0.8, zorder=1)

    if show_gap_junctions:
        mids = (z[ends[:, 0]] + z[ends[:, 1]]) / 2
        m_to_y = net.m_to_y.cpu().numpy()
        drawn = set()
        for m, neighbor_y in enumerate(net.muscle_tempgv_neighbors):
            for n in m_to_y[np.asarray(neighbor_y, dtype=int)]:
                pair = (min(m, int(n)), max(m, int(n)))
                if pair not in drawn:
                    drawn.add(pair)
                    ax.plot(mids[list(pair), 0], mids[list(pair), 1],
                            color='#0173B2', ls=':', lw=1.5, alpha=0.7,
                            zorder=1)

    ax.scatter(z[:, 0], z[:, 1], s=dot_size, color=CHROMA_COLOR,
               edgecolors='black', zorder=3)
    ax.set_aspect('equal')
    ax.axis('off')
    return fig, ax


def plot_current(iapp, position: tuple[float, float], t_max: float,
                 n_points: int = 1000, ax=None):
    """Plots an applied current over time at one position.

    Args:
        iapp: Applied current function iapp(positions, t), as passed to
            ChromatoNet.
        position: (x, y) where the current is evaluated.
        t_max: End time in seconds.
        n_points: Number of time points.
        ax: Axes to draw on (a new figure is made if None).

    Returns:
        fig, ax
    """
    pos = torch.tensor([position], dtype=torch.float32)
    ts = np.linspace(0, t_max, n_points)
    current = [float(iapp(pos, torch.tensor(t_val, dtype=torch.float32))[0])
               for t_val in ts]
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 2.5))
    else:
        fig = ax.get_figure()
    ax.plot(ts, current, color='black', lw=2)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('applied current')
    _clean_axes(ax)
    fig.tight_layout()
    return fig, ax


def plot_param_map(net, key: str, title: str | None = None, ax=None):
    """Shows where a parameter differs from its usual value.

    Positions with the most common value are small gray dots; all others
    are larger colored dots, with one legend entry per value. ks and kc are
    per chromatophore; other parameters are per muscle, at the muscle
    midpoints.

    Args:
        net: A ChromatoNet.
        key: Parameter name, for example 'kp' or 'ks'.
        title: Optional title.
        ax: Axes to draw on (a new figure is made if None).

    Returns:
        fig, ax
    """
    z = net.z.cpu().numpy()
    values = net.cached_params[key].cpu().numpy()
    if key in ('ks', 'kc'):
        pos = z
        values = values[:, 0]
    else:
        ends = net.y_to_x[net.y_to_m].cpu().numpy()
        pos = (z[ends[:, 0]] + z[ends[:, 1]]) / 2
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.get_figure()

    unique_vals, counts = np.unique(values, return_counts=True)
    base = unique_vals[np.argmax(counts)]
    is_base = values == base
    ax.scatter(pos[is_base, 0], pos[is_base, 1], s=12, color='lightgray',
               label=f'{key} = {base:g}')
    others = [val for val in unique_vals if val != base]
    for i, val in enumerate(others):
        sel = values == val
        ax.scatter(pos[sel, 0], pos[sel, 1], s=45,
                   color=MUSCLE_COLORS[i % len(MUSCLE_COLORS)],
                   label=f'{key} = {val:g}')
    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), frameon=False)
    if title is not None:
        ax.set_title(title)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.tight_layout()
    return fig, ax


def plot_comparison(
    names: Sequence[str],
    titles: Sequence[str],
    time: float,
    radius_scale: float = 2.5,
    radius_floor: float = 0.07,
    padding: float = 0.3,
    color: str = CHROMA_COLOR,
):
    """Plots one snapshot per run, side by side, at the same time.

    Args:
        names: Run names passed to ChromatoNet.run().
        titles: One title per panel.
        time: Time to show; the nearest saved frame is used.
        radius_scale: See plot_snapshots.
        radius_floor: See plot_snapshots.
        padding: Space around the lattice.
        color: Chromatophore color.

    Returns:
        fig, axes
    """
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.8), squeeze=False)
    axes = axes[0]
    for ax, name, title in zip(axes, names, titles):
        t, x, _, _ = ch.load_run(name)
        frame_idx = int(np.argmin(np.abs(t - time)))
        _draw_chromatophores(ax, x[frame_idx], radius_scale, radius_floor,
                             color)
        xl, xh, yl, yh = _limits(x, padding)
        ax.set_xlim(xl, xh)
        ax.set_ylim(yl, yh)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'{title} (t = {t[frame_idx]:.0f}s)')
    fig.tight_layout()
    return fig, axes