"""Animates a saved ChromatoNet simulation.

Reads the x_pos_{name}.csv and time_{name}.csv files written by
ChromatoNet.run(steps, name=name) and draws each chromatophore as a circle
whose size follows its radius.

In a notebook, displaying the animation with
plt.rcParams['animation.html'] = 'html5', or saving it to a video file,
requires ffmpeg. 'jshtml' works without it.

Typical usage example:

    anim = animate_module.animate('my_run')
"""

from __future__ import annotations

from matplotlib import animation
from matplotlib import collections as mpl_collections
import matplotlib.pyplot as plt
import numpy as np


def animate(name_scheme: str, speedup: float = 10) -> animation.FuncAnimation:
    """Returns a matplotlib animation of the run saved under name_scheme.

    Args:
        name_scheme: The `name` passed to ChromatoNet.run().
        speedup: How many times faster than simulated time the animation
            plays.
    """
    # Time between frames in ms, from the saved time points.
    try:
        time_data = np.loadtxt(f'time_{name_scheme}.csv', delimiter=' ')
        tstep = 1000*(time_data[1]-time_data[0])/speedup
    except (OSError, IndexError, ValueError):
        print('Could not load time data, using default timestep')
        tstep = 50  # Default 50 ms between frames.

    # One row per saved time, [x, y, radius] per chromatophore.
    x = np.loadtxt(f'x_pos_{name_scheme}.csv', delimiter=' ', ndmin=2)
    xrs = x.reshape((len(x), int(len(x[0])/3), 3))

    # Plot limits with a margin of 1, ignoring any NaN values.
    xl = np.nanmin(xrs[:, :, 0])-1
    xh = np.nanmax(xrs[:, :, 0])+1
    yl = np.nanmin(xrs[:, :, 1])-1
    yh = np.nanmax(xrs[:, :, 1])+1

    fig = plt.figure(figsize=(10, 10*(yh-yl)/(xh-xl)))
    ax = plt.axes(xlim=(xl, xh), ylim=(yl, yh))
    plt.axis('off')

    # Radius + 0.05 so closed chromatophores still show as small dots.
    patches = []
    for c in xrs[0]:
        patches.append(plt.Circle((c[0], c[1]), c[2] + 0.05))

    collection = mpl_collections.PatchCollection(patches, facecolors='red')

    def init():
        ax.add_collection(collection)
        return collection,

    def animate_frame(i):
        patches = []
        for c in xrs[i]:
            patches.append(plt.Circle((c[0], c[1]), c[2] + 0.05))
        collection.set_paths(patches)
        return collection,

    anim = animation.FuncAnimation(fig, animate_frame,
                                   init_func=init,
                                   frames=len(xrs),
                                   interval=tstep,  # Time between frames, ms.
                                   blit=True)

    return anim