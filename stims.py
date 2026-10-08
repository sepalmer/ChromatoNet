"""Stimulus and set-up functions for ChromatoNet.

Initial voltages (vi_*) are passed to ChromatoNet.set_vi(). They take the
muscle positions x, a tensor of shape (n_muscles, 2), and return one voltage
per muscle.

Applied currents (iapp_*) are passed as the iapp argument of ChromatoNet.
They take the muscle positions x and the time t, and return one current per
muscle.

Initial positions (xi_*) are passed to set_xi() or reset_net(xi=...). They
take one rest position z, a NumPy array of shape (2,), and return the
starting position.

Parameter maps (param_*) are used in params to make a parameter position
dependent, for example a stiff region. They take one position and return a
value.

Keyword arguments set the shape. To change them, wrap the function:

    net.set_vi(lambda x: stims.vi_edge(x, edge='bottom', width=2.0))
    iapp=lambda x, t: stims.iapp_edge_periodic(x, t, frequency=0.02)
    params={'kp': lambda p: stims.param_disk(
        p, center=(14., 12.), radius=3., inside=100., outside=0.1)}
"""

from __future__ import annotations

import numpy as np
import torch


def _edge_distance(x: torch.Tensor, edge: str) -> torch.Tensor:
    """Returns the distance used for the edge Gaussian.

    Args:
        x: Muscle positions, shape (n_muscles, 2).
        edge: 'left', 'right', 'bottom' or 'top'.

    Raises:
        ValueError: If edge is not recognized.
    """
    if edge == 'left':
        return x[:, 0]
    if edge == 'right':
        return -x[:, 0]
    if edge == 'bottom':
        return x[:, 1]
    if edge == 'top':
        return -x[:, 1]
    raise ValueError(
        f"Unknown edge: {edge!r}. Use 'left', 'right', 'top' or 'bottom'.")


# Initial voltages.

def vi_gaussian(
    x: torch.Tensor,
    center: tuple[float, float] = (0., 0.),
    width: float = 1.0,
    base: float = -0.3,
    amplitude: float = 1.0,
) -> torch.Tensor:
    """Returns a Gaussian depolarization around a point.

    Args:
        x: Muscle positions, shape (n_muscles, 2).
        center: Center of the Gaussian. The default is the lower-left corner.
        width: Standard deviation of the Gaussian.
        base: Voltage far from the center.
        amplitude: Extra voltage at the center.

    Returns:
        One voltage per muscle, shape (n_muscles,).
    """
    c = torch.tensor(center, dtype=torch.float32, device=x.device)
    return base + amplitude * torch.exp(
        -((x - c)**2).sum(dim=1) / (2 * width**2))


def vi_disk(
    x: torch.Tensor,
    center: tuple[float, float] = (0., 0.),
    radius: float = 1.0,
    base: float = -0.266,
    amplitude: float = 6.0,
) -> torch.Tensor:
    """Returns a uniform depolarization of all muscles within a disk.

    Args:
        x: Muscle positions, shape (n_muscles, 2).
        center: Center of the disk.
        radius: Radius of the disk.
        base: Voltage outside the disk.
        amplitude: Extra voltage inside the disk.

    Returns:
        One voltage per muscle, shape (n_muscles,).
    """
    c = torch.tensor(center, dtype=torch.float32, device=x.device)
    inside = torch.sqrt(((x - c)**2).sum(dim=1)) < radius
    return base + amplitude * inside.to(torch.float32)


def vi_rectangle(
    x: torch.Tensor,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    base: float = -0.3,
    amplitude: float = 1.0,
) -> torch.Tensor:
    """Returns a uniform depolarization of all muscles inside a rectangle.

    Args:
        x: Muscle positions, shape (n_muscles, 2).
        xlim: (min, max) x of the rectangle.
        ylim: (min, max) y of the rectangle.
        base: Voltage outside the rectangle.
        amplitude: Extra voltage inside the rectangle.

    Returns:
        One voltage per muscle, shape (n_muscles,).
    """
    inside = ((x[:, 0] >= xlim[0]) & (x[:, 0] <= xlim[1]) &
              (x[:, 1] >= ylim[0]) & (x[:, 1] <= ylim[1]))
    return base + amplitude * inside.to(torch.float32)


def vi_edge(
    x: torch.Tensor,
    edge: str = 'left',
    width: float = 1.0,
    base: float = -0.3,
    amplitude: float = 1.0,
) -> torch.Tensor:
    """Returns a Gaussian depolarization falling off from one edge.

    Args:
        x: Muscle positions, shape (n_muscles, 2).
        edge: 'left', 'right', 'bottom' or 'top'.
        width: Standard deviation of the Gaussian.
        base: Voltage far from the edge.
        amplitude: Extra voltage at the edge.

    Returns:
        One voltage per muscle, shape (n_muscles,).
    """
    distance = _edge_distance(x, edge)
    return base + amplitude * torch.exp(-(distance**2) / (2 * width**2))


# Applied currents.

def iapp_edge_periodic(
    x: torch.Tensor,
    t: torch.Tensor,
    edge: str = 'left',
    width: float = 1.0,
    frequency: float = 0.0125,
    base: float = 0.15,
    amplitude: float = 0.5,
) -> torch.Tensor:
    """Returns a sinusoidal current along one edge.

    The current is base + amplitude * edge Gaussian * cos(2 pi f t).

    Args:
        x: Muscle positions, shape (n_muscles, 2).
        t: Time.
        edge: 'left', 'right', 'bottom' or 'top'.
        width: Standard deviation of the edge Gaussian.
        frequency: Frequency of the sinusoid, in Hz.
        base: Current everywhere.
        amplitude: Amplitude of the sinusoid at the edge.

    Returns:
        One current per muscle, shape (n_muscles,).
    """
    distance = _edge_distance(x, edge)
    width_t = torch.tensor(width, dtype=torch.float32, device=x.device)
    spatial = torch.exp(-(distance**2) / (2 * width_t**2))

    freq = torch.tensor(frequency, dtype=torch.float32, device=x.device)
    temporal = torch.cos(2 * np.pi * freq * t)

    return base + amplitude * spatial * temporal


def iapp_gaussian_periodic(
    x: torch.Tensor,
    t: torch.Tensor,
    center: tuple[float, float] = (0., 0.),
    width: float = 1.0,
    frequency: float = 0.0125,
    base: float = 0.15,
    amplitude: float = 0.5,
) -> torch.Tensor:
    """Returns a sinusoidal current around a point.

    The current is base + amplitude * Gaussian * cos(2 pi f t). frequency=0
    gives a constant local current.

    Args:
        x: Muscle positions, shape (n_muscles, 2).
        t: Time.
        center: Center of the Gaussian.
        width: Standard deviation of the Gaussian.
        frequency: Frequency of the sinusoid, in Hz.
        base: Current everywhere.
        amplitude: Amplitude of the sinusoid at the center.

    Returns:
        One current per muscle, shape (n_muscles,).
    """
    c = torch.tensor(center, dtype=torch.float32, device=x.device)
    width_t = torch.tensor(width, dtype=torch.float32, device=x.device)
    spatial = torch.exp(-((x - c)**2).sum(dim=1) / (2 * width_t**2))

    freq = torch.tensor(frequency, dtype=torch.float32, device=x.device)
    temporal = torch.cos(2 * np.pi * freq * t)

    return base + amplitude * spatial * temporal


# Initial positions.

def xi_displace(
    z: np.ndarray,
    center: tuple[float, float] = (0., 0.),
    radius: float = 0.5,
    shift: tuple[float, float] = (-0.25, 0.),
) -> np.ndarray:
    """Returns the starting position, shifted if near a point.

    Chromatophores within `radius` of `center` are shifted by `shift`; all
    others start at rest. The default pulls the first chromatophore of a
    chain to the left, which stretches the first muscle past the calcium
    threshold.

    Args:
        z: One rest position, shape (2,).
        center: Center of the shifted region.
        radius: Radius of the shifted region.
        shift: Displacement applied inside the region.

    Returns:
        The starting position as float32, shape (2,).
    """
    z = np.asarray(z, dtype=np.float32)
    if np.linalg.norm(z - np.asarray(center, dtype=np.float32)) < radius:
        return z + np.asarray(shift, dtype=np.float32)
    return z


# Parameter maps (for params).

def param_disk(
    pos: np.ndarray,
    center: tuple[float, float],
    radius: float,
    inside: float,
    outside: float,
) -> float:
    """Returns `inside` within a disk and `outside` elsewhere.

    Args:
        pos: One position, shape (2,).
        center: Center of the disk.
        radius: Radius of the disk.
        inside: Value inside the disk.
        outside: Value outside the disk.
    """
    if np.linalg.norm(np.asarray(pos) - np.asarray(center)) < radius:
        return inside
    return outside


def param_rectangle(
    pos: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    inside: float,
    outside: float,
) -> float:
    """Returns `inside` within a rectangle and `outside` elsewhere.

    Args:
        pos: One position, shape (2,).
        xlim: (min, max) x of the rectangle.
        ylim: (min, max) y of the rectangle.
        inside: Value inside the rectangle.
        outside: Value outside the rectangle.
    """
    in_x = xlim[0] <= pos[0] <= xlim[1]
    in_y = ylim[0] <= pos[1] <= ylim[1]
    return inside if (in_x and in_y) else outside