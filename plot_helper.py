"""Plotting helpers for ChromatoNet simulations."""

import numpy as np
import matplotlib.pyplot as plt


def load_run(name):
    """Load a saved simulation.

    Returns:
        t: saved times, shape (n_times,)
        x: chromatophore data [x, y, radius], shape (n_times, n_chromatophores, 3)
        v: muscle data [x, y, voltage] at the muscle midpoint, shape (n_times, n_muscles, 3)
        z: chromatophore rest positions, shape (n_chromatophores, 2)
    """
    t = np.loadtxt(f"time_{name}.csv", ndmin=1)
    x = np.loadtxt(f"x_pos_{name}.csv", delimiter=' ', ndmin=2)
    v = np.loadtxt(f"v_pos_{name}.csv", delimiter=' ', ndmin=2)
    z = np.loadtxt(f"z_pos_{name}.csv", delimiter=' ', ndmin=2)
    return t, x.reshape(len(x), -1, 3), v.reshape(len(v), -1, 3), z


def plot_probe(net, name, chromat_idx, window=4.0, figsize=(12, 4)):
    """Probe one chromatophore: network diagram, its radius, and the voltages of its muscles.

    Colors match across panels: the probed chromatophore and its radius share
    one color, and each attached muscle and its voltage trace share another.

    Args:
        net: the chromatophore_network used for the run
        name: the `name` passed to run()
        chromat_idx: index of the chromatophore to probe
        window: half-width of the diagram around the probed chromatophore (None shows everything)
        figsize: figure size

    Returns:
        fig, (ax_net, ax_radius, ax_voltage)
    """
    t, x, v, z = load_run(name)

    # Muscles attached to the probed chromatophore, and the chromatophore at the other end of each
    y_idx = net.x_to_y_tensors[chromat_idx]
    probe_muscles = net.m_to_y[y_idx].cpu().numpy()
    other_chromats = net.y_to_x[net.y_to_y[y_idx]].cpu().numpy()

    # The two chromatophores at the ends of every muscle
    ends = net.y_to_x[net.y_to_m].cpu().numpy()

    chromat_color = '#CC2222'
    muscle_colors = plt.cm.tab10(np.arange(len(probe_muscles)) % 10)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.5])
    ax_net = fig.add_subplot(gs[:, 0])
    ax_r = fig.add_subplot(gs[0, 1])
    ax_v = fig.add_subplot(gs[1, 1], sharex=ax_r)

    # Network diagram: all muscles in grey, the probed muscles in color
    for a, b in ends:
        ax_net.plot(z[[a, b], 0], z[[a, b], 1], color='0.85', lw=1, zorder=1)
    for m, c in zip(probe_muscles, muscle_colors):
        a, b = ends[m]
        ax_net.plot(z[[a, b], 0], z[[a, b], 1], color=c, lw=3, zorder=2)
    ax_net.scatter(z[:, 0], z[:, 1], s=20, color='0.5', zorder=3)
    ax_net.scatter(z[chromat_idx, 0], z[chromat_idx, 1], s=150, color=chromat_color, zorder=4)
    ax_net.annotate(str(chromat_idx), z[chromat_idx], xytext=(0, 10),
                    textcoords='offset points', ha='center')
    if window is not None:
        ax_net.set_xlim(z[chromat_idx, 0] - window, z[chromat_idx, 0] + window)
        ax_net.set_ylim(z[chromat_idx, 1] - window, z[chromat_idx, 1] + window)
    ax_net.set_aspect('equal')
    ax_net.axis('off')

    # Radius of the probed chromatophore
    ax_r.plot(t, x[:, chromat_idx, 2], color=chromat_color, label=f"chromatophore {chromat_idx}")
    ax_r.set_ylabel("radius")
    ax_r.legend(loc='upper right', fontsize=8)
    plt.setp(ax_r.get_xticklabels(), visible=False)

    # Voltage of each attached muscle
    for m, other, c in zip(probe_muscles, other_chromats, muscle_colors):
        ax_v.plot(t, v[:, m, 2], color=c, label=f"muscle {chromat_idx}–{other}")
    ax_v.set_xlabel("time (s)")
    ax_v.set_ylabel("voltage")
    ax_v.legend(loc='upper right', fontsize=8)

    fig.tight_layout()
    return fig, (ax_net, ax_r, ax_v)