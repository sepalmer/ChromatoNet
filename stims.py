"""Stimulus and set-up functions for ChromatoNet.

Initial voltages (vi_*) are passed to chromatophore_network.set_VI(). They take
the muscle positions x, a tensor of shape (n_muscles, 2), and return one
voltage per muscle.

Applied currents (iapp_*) are passed as the iapp argument of
chromatophore_network. They take the muscle positions x and the time t, and
return one current per muscle.

Initial positions (xi_*) are passed to set_XI() or reset_net(xi=...). They take
one rest position z (a NumPy array of length 2) and return the starting position.

Parameter maps (param_*) are used in PINITS to make a parameter position
dependent, for example a stiff region. They take one position and return a value.

Keyword arguments set the shape. To change them, wrap the function:
    net.set_VI(lambda x: stims.vi_edge(x, edge='bottom', width=2.0))
    iapp=lambda x, t: stims.iapp_edge_periodic(x, t, frequency=0.02)
    PINITS={'kp': lambda p: stims.param_disk(p, center=(14., 12.), radius=3., inside=100., outside=0.1)}
"""

import numpy as np
import torch


def _edge_distance(x, edge):
    """Distance used for the edge Gaussian: x for left/right, y for bottom/top."""
    if edge == 'left':
        return x[:, 0]
    elif edge == 'right':
        return -x[:, 0]
    elif edge == 'bottom':
        return x[:, 1]
    elif edge == 'top':
        return -x[:, 1]
    else:
        raise ValueError("edge must be 'left', 'right', 'top', or 'bottom'")


# Initial voltages

def vi_gaussian(x, center=(0., 0.), width=1.0, base=-0.3, amplitude=1.0):
    """Gaussian depolarization around a point. The default center is the lower-left corner."""
    c = torch.tensor(center, dtype=torch.float32, device=x.device)
    return base + amplitude * torch.exp(-((x - c)**2).sum(dim=1) / (2 * width**2))


def vi_disk(x, center=(0., 0.), radius=1.0, base=-0.266, amplitude=6.0):
    """Uniform depolarization of all muscles within `radius` of a point."""
    c = torch.tensor(center, dtype=torch.float32, device=x.device)
    inside = torch.sqrt(((x - c)**2).sum(dim=1)) < radius
    return base + amplitude * inside.to(torch.float32)


def vi_rectangle(x, xlim, ylim, base=-0.3, amplitude=1.0):
    """Uniform depolarization of all muscles inside a rectangle, e.g. a narrow strip."""
    inside = ((x[:, 0] >= xlim[0]) & (x[:, 0] <= xlim[1]) &
              (x[:, 1] >= ylim[0]) & (x[:, 1] <= ylim[1]))
    return base + amplitude * inside.to(torch.float32)


def vi_edge(x, edge='left', width=1.0, base=-0.3, amplitude=1.0):
    """Gaussian depolarization falling off from one edge of the lattice."""
    distance = _edge_distance(x, edge)
    return base + amplitude * torch.exp(-(distance**2) / (2 * width**2))


# Applied currents

def iapp_edge_periodic(x, t, edge='left', width=1.0, frequency=0.0125, base=0.15, amplitude=0.5):
    """Sinusoidal current along one edge: base + amplitude * edge Gaussian * cos(2 pi f t)."""
    distance = _edge_distance(x, edge)
    width_t = torch.tensor(width, dtype=torch.float32, device=x.device)
    spatial = torch.exp(-(distance**2) / (2 * width_t**2))

    freq = torch.tensor(frequency, dtype=torch.float32, device=x.device)
    temporal = torch.cos(2 * np.pi * freq * t)

    return base + amplitude * spatial * temporal


def iapp_gaussian_periodic(x, t, center=(0., 0.), width=1.0, frequency=0.0125, base=0.15, amplitude=0.5):
    """Sinusoidal current around a point: base + amplitude * Gaussian * cos(2 pi f t).

    frequency=0 gives a constant local current.
    """
    c = torch.tensor(center, dtype=torch.float32, device=x.device)
    width_t = torch.tensor(width, dtype=torch.float32, device=x.device)
    spatial = torch.exp(-((x - c)**2).sum(dim=1) / (2 * width_t**2))

    freq = torch.tensor(frequency, dtype=torch.float32, device=x.device)
    temporal = torch.cos(2 * np.pi * freq * t)

    return base + amplitude * spatial * temporal


# Initial positions

def xi_displace(z, center=(0., 0.), radius=0.5, shift=(-0.25, 0.)):
    """Shift chromatophores within `radius` of `center` by `shift`; others start at rest.

    The default pulls the first chromatophore of a chain to the left, which
    stretches the first muscle past the calcium threshold.
    """
    z = np.asarray(z, dtype=np.float32)
    if np.linalg.norm(z - np.asarray(center, dtype=np.float32)) < radius:
        return z + np.asarray(shift, dtype=np.float32)
    return z


# Parameter maps (for PINITS)

def param_disk(pos, center, radius, inside, outside):
    """Return `inside` within `radius` of `center` and `outside` elsewhere."""
    if np.linalg.norm(np.asarray(pos) - np.asarray(center)) < radius:
        return inside
    return outside


def param_rectangle(pos, xlim, ylim, inside, outside):
    """Return `inside` within the rectangle xlim x ylim and `outside` elsewhere."""
    in_x = xlim[0] <= pos[0] <= xlim[1]
    in_y = ylim[0] <= pos[1] <= ylim[1]
    return inside if (in_x and in_y) else outside