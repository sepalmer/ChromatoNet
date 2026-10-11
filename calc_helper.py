"""Analysis helpers for ChromatoNet simulations.

These functions read the files written by ChromatoNet.run() and compute the
quantities used in the paper: when chromatophores open, wave speed, reach,
thickness and density, power spectra and peak frequencies. The wave and
frequency measures are written for 2D lattices.

Typical usage example:

    t, x, v, z = calc_helper.load_run('my_run')
    xs, mean_t, std_t = calc_helper.time_to_open(t, x, z)
"""

from __future__ import annotations

import numpy as np


def load_run(
    name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads a saved simulation.

    Args:
        name: The name passed to ChromatoNet.run().

    Returns:
        A tuple (t, x, v, z):
            t: Saved times, shape (n_times,).
            x: [x, y, radius] per chromatophore, shape
                (n_times, n_chromatophores, 3).
            v: [x, y, voltage] per muscle at its midpoint, shape
                (n_times, n_muscles, 3).
            z: Chromatophore rest positions, shape (n_chromatophores, 2).
    """
    t = np.loadtxt(f'time_{name}.csv', ndmin=1)
    x = np.loadtxt(f'x_pos_{name}.csv', delimiter=' ', ndmin=2)
    v = np.loadtxt(f'v_pos_{name}.csv', delimiter=' ', ndmin=2)
    z = np.loadtxt(f'z_pos_{name}.csv', delimiter=' ', ndmin=2)
    return t, x.reshape(len(x), -1, 3), v.reshape(len(v), -1, 3), z


def is_chain(z: np.ndarray) -> bool:
    """Returns True if the rest positions form a 1D chain."""
    return np.ptp(z[:, 1]) < 0.1 * np.ptp(z[:, 0])


def interior_indices(z: np.ndarray) -> np.ndarray:
    """Returns the chromatophores not on the edges of the bounding box."""
    xs, ys = z[:, 0], z[:, 1]
    on_edge = ((xs == xs.min()) | (xs == xs.max()) |
               (ys == ys.min()) | (ys == ys.max()))
    return np.where(~on_edge)[0]


def chromats_along_x(
    z: np.ndarray,
    n: int,
    row_fraction: float = 0.5,
    start: float = 0.1,
    end: float = 0.9,
) -> list[int]:
    """Returns n chromatophores evenly spaced along x at one height.

    Args:
        z: Rest positions, shape (n_chromatophores, 2).
        n: Number of chromatophores.
        row_fraction: Height as a fraction of the lattice (0 is the bottom).
        start: Start of the x range, as a fraction of the lattice width.
        end: End of the x range, as a fraction of the lattice width.

    Returns:
        The index of the chromatophore nearest each target point.
    """
    x_min, x_max = z[:, 0].min(), z[:, 0].max()
    y_min, y_max = z[:, 1].min(), z[:, 1].max()
    target_y = y_min + row_fraction * (y_max - y_min)
    indices = []
    for frac in np.linspace(start, end, n):
        target_x = x_min + frac * (x_max - x_min)
        dist = np.hypot(z[:, 0] - target_x, z[:, 1] - target_y)
        indices.append(int(np.argmin(dist)))
    return indices


def connected_muscles(net, chromat_idx: int) -> list[tuple[int, int]]:
    """Returns the muscles attached to one chromatophore.

    Args:
        net: A ChromatoNet (or ChromatoNet24).
        chromat_idx: Index of the chromatophore.

    Returns:
        A list of (muscle index, index of the chromatophore at the other end).
    """
    y_idx = net.x_to_y_tensors[chromat_idx]
    muscles = net.m_to_y[y_idx].cpu().numpy()
    others = net.y_to_x[net.y_to_y[y_idx]].cpu().numpy()
    return [(int(m), int(o)) for m, o in zip(muscles, others)]


def time_to_open(
    t: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    threshold_fraction: float = 0.9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns when chromatophores first open, averaged per x position.

    A chromatophore opens when its radius first exceeds threshold_fraction
    of its own maximum. Edge chromatophores are left out.

    Args:
        t: Saved times.
        x: Chromatophore data from load_run.
        z: Rest positions.
        threshold_fraction: Fraction of each chromatophore's maximum radius.

    Returns:
        A tuple (unique_x, mean_times, std_times). Times are NaN at x
        positions where no chromatophore opened.
    """
    valid = interior_indices(z)
    valid_z = z[valid]
    unique_x = np.unique(valid_z[:, 0])
    mean_times, std_times = [], []
    for x_val in unique_x:
        times_at_x = []
        for ci in valid[valid_z[:, 0] == x_val]:
            radius = x[:, ci, 2]
            opened = np.where(radius > threshold_fraction * np.max(radius))[0]
            if len(opened) > 0:
                times_at_x.append(t[opened[0]])
        mean_times.append(np.mean(times_at_x) if times_at_x else np.nan)
        std_times.append(np.std(times_at_x) if times_at_x else np.nan)
    return unique_x, np.array(mean_times), np.array(std_times)


def wave_velocity(
    unique_x: np.ndarray,
    mean_times: np.ndarray,
) -> tuple[float, float]:
    """Returns the wave speed from a linear fit of opening time to position.

    Position is normalized to [0, 1], so the speed is in lattice widths per
    second. The error is a rough estimate from the fit residuals.

    Args:
        unique_x: x positions from time_to_open.
        mean_times: Mean opening times from time_to_open.

    Returns:
        A tuple (velocity, error). Both are NaN if there are fewer than two
        points or the wave does not move forward.
    """
    x_norm = (unique_x - unique_x.min()) / np.ptp(unique_x)
    valid = ~np.isnan(mean_times)
    if valid.sum() < 2:
        return np.nan, np.nan
    coeffs = np.polyfit(x_norm[valid], mean_times[valid], 1)
    slope = coeffs[0]
    if slope <= 0:
        return np.nan, np.nan
    velocity = 1.0 / slope
    residuals = mean_times[valid] - np.polyval(coeffs, x_norm[valid])
    slope_error = np.std(residuals) / np.sqrt(valid.sum())
    return velocity, velocity**2 * slope_error


def wave_reach(
    x: np.ndarray,
    z: np.ndarray,
    reach_threshold: float = 0.9,
) -> float:
    """Returns how far across the lattice the wave reaches, in percent.

    An x position counts as reached if the radius averaged over its
    chromatophores exceeds reach_threshold of the overall maximum radius at
    any time. Edge chromatophores are left out.

    Args:
        x: Chromatophore data from load_run.
        z: Rest positions.
        reach_threshold: Fraction of the overall maximum radius.
    """
    valid = interior_indices(z)
    valid_z = z[valid]
    threshold = reach_threshold * np.max(x[:, valid, 2])
    unique_x = np.unique(valid_z[:, 0])
    max_x_reached = unique_x.min()
    for x_val in unique_x:
        column = valid[valid_z[:, 0] == x_val]
        if np.max(np.mean(x[:, column, 2], axis=1)) > threshold:
            max_x_reached = x_val
    return 100 * (max_x_reached - unique_x.min()) / np.ptp(unique_x)


def wave_thickness(
    t: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    absolute_threshold: float = 0.01,
) -> tuple[float, float]:
    """Returns the wave thickness when the wave crosses the middle.

    At the time the wave reaches the middle x position (90% opening
    threshold), measures the x extent of open chromatophores (radius above
    absolute_threshold) in each row.

    Args:
        t: Saved times.
        x: Chromatophore data from load_run.
        z: Rest positions.
        absolute_threshold: Radius above which a chromatophore counts as open.

    Returns:
        A tuple (mean, std) over rows, or (NaN, NaN) if no row has two open
        chromatophores.
    """
    unique_x, mean_times, _ = time_to_open(t, x, z, threshold_fraction=0.9)
    x_norm = (unique_x - unique_x.min()) / np.ptp(unique_x)
    mid_idx = np.argmin(np.abs(x_norm - 0.5))
    time_idx = np.argmin(np.abs(t - mean_times[mid_idx]))

    valid = interior_indices(z)
    valid_z = z[valid]
    thicknesses = []
    for y_val in np.unique(valid_z[:, 1]):
        row = valid[valid_z[:, 1] == y_val]
        open_x = z[row, 0][x[time_idx, row, 2] > absolute_threshold]
        if len(open_x) > 1:
            thicknesses.append(np.ptp(open_x))
    if not thicknesses:
        return np.nan, np.nan
    return np.mean(thicknesses), np.std(thicknesses)


def wave_density(
    t: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    window_size: int = 4,
    start_time: float = 0.0,
    end_time: float | None = None,
) -> tuple[float, float]:
    """Returns how densely chromatophores open around the wave peak.

    At each time, finds the x position with the largest column-averaged
    radius and averages the column radii within window_size positions of it.

    Args:
        t: Saved times.
        x: Chromatophore data from load_run.
        z: Rest positions.
        window_size: Number of x positions on each side of the peak.
        start_time: Start of the time window.
        end_time: End of the time window (defaults to the end of the run).

    Returns:
        A tuple (mean, std) over time.
    """
    mask = t >= start_time
    if end_time is not None:
        mask &= t <= end_time
    valid = interior_indices(z)
    valid_z = z[valid]
    unique_x = np.unique(valid_z[:, 0])
    columns = [valid[valid_z[:, 0] == x_val] for x_val in unique_x]
    densities = []
    for frame in x[mask]:
        col_radii = np.array([np.mean(frame[col, 2]) for col in columns])
        peak = np.argmax(col_radii)
        lo = max(0, peak - window_size)
        hi = min(len(unique_x), peak + window_size + 1)
        densities.append(np.mean(col_radii[lo:hi]))
    return np.mean(densities), np.std(densities)


def peak_frequency(signal: np.ndarray, t: np.ndarray) -> float:
    """Returns the frequency (Hz) with the most power in a signal."""
    freqs = np.fft.fftfreq(len(signal), t[1] - t[0])
    pos = freqs > 0
    if not np.any(pos):
        return 0.0
    power = np.abs(np.fft.fft(signal)[pos])**2
    return freqs[pos][np.argmax(power)]


def power_spectrum(
    t: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    max_freq: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns the radius power spectrum averaged over interior chromatophores.

    Args:
        t: Saved times.
        x: Chromatophore data from load_run.
        z: Rest positions.
        max_freq: Highest frequency to return, in Hz (None for all).

    Returns:
        A tuple (freqs, power) for positive frequencies in Hz.
    """
    valid = interior_indices(z)
    freqs = np.fft.fftfreq(len(t), t[1] - t[0])
    pos = freqs > 0
    power = np.zeros(pos.sum())
    for ci in valid:
        power += np.abs(np.fft.fft(x[:, ci, 2])[pos])**2
    power /= len(valid)
    freqs = freqs[pos]
    if max_freq is not None:
        keep = freqs <= max_freq
        freqs, power = freqs[keep], power[keep]
    return freqs, power


def measured_frequency(
    t: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    start_time: float = 0.0,
    end_time: float | None = None,
) -> tuple[float, float]:
    """Returns the mean and spread of each chromatophore's peak frequency.

    Args:
        t: Saved times.
        x: Chromatophore data from load_run.
        z: Rest positions.
        start_time: Start of the time window.
        end_time: End of the time window (defaults to the end of the run).

    Returns:
        A tuple (mean, std) in Hz over interior chromatophores.
    """
    mask = t >= start_time
    if end_time is not None:
        mask &= t <= end_time
    freqs = [peak_frequency(x[mask, ci, 2], t[mask])
             for ci in interior_indices(z)]
    return np.mean(freqs), np.std(freqs)


def wilson_ci(p: float, n: int, z_score: float = 1.96) -> tuple[float, float]:
    """Returns the Wilson score interval for a proportion p from n trials."""
    if n == 0:
        return 0.0, 0.0
    denom = 1 + z_score**2 / n
    center = (p + z_score**2 / (2 * n)) / denom
    spread = (z_score * np.sqrt((p * (1 - p) + z_score**2 / (4 * n)) / n)
              / denom)
    return max(0.0, center - spread), min(1.0, center + spread)