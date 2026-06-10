import numpy as np
from matplotlib import pyplot as plt

from trace.core import TrajectoryManager
from trace.visuals.utils import env_frame


def grid_trajectories(manager:TrajectoryManager, abstr_frame:bool=False,
                      title: str|None=None, alpha: float|int=0.1, color: str='red'):

    frame = plt.imread('plots/sketches/dst_frame_abstr.png') if abstr_frame else env_frame(manager.metadata['env_id'])
    h, w = np.array(manager.metadata['observations_high'])[:2]
    fig, ax = plt.subplots()
    ax.imshow(frame, extent=(0, w, 0, h), origin='lower')

    for obs, acs, _ in zip(*manager.conditioning_features(per_point=False)):

        if len(obs) < 2: continue

        x_obs, y_obs = [], []
        x_acs, y_acs = [], []

        for coords, a in zip(obs, acs):
            y, x = coords[:2]
            x_obs.append(x)
            y_obs.append(y)

            if a == 0:
                x_acs.append(x)
                y_acs.append(y)

        ax.plot(x_obs, y_obs, alpha=alpha, linewidth=1.5, color=color)

        if 'minetrain' in manager.metadata['env_id']:
            ax.scatter(x_acs, y_acs, marker='x', color='blue', s=20)

    ax.set(xlim=(0, w), ylim=(h, 0), aspect='equal')
    ax.axis('off')
    if title: ax.set_title(title)
    fig.tight_layout(rect=(0, 0, 1, 1))
    return fig


def quantization_grid(obs:list, ranges:list, metadata:dict, abstr_frame:bool=False):
    frame = plt.imread('plots/sketches/dst_frame_abstr.png') if abstr_frame else env_frame(metadata['env_id'])
    h, w = np.array(metadata['observations_high'])[:2]
    fig, ax = plt.subplots()
    ax.imshow(frame, extent=(0, w, 0, h), origin='lower')
    x, y = [coords[0] for traj in obs for coords in traj], [coords[1] for traj in obs for coords in traj]


    ax.scatter(x, y, marker='o', color='yellow', s=1, alpha=0.1)
    ax.vlines(ranges[0], 0, 1, linestyle='dashed', color='blue', linewidth=0.5)
    ax.hlines(ranges[1], 0, 1, linestyle='dashed', color='blue', linewidth=0.5)
    ax.set(xlim=(0, w), ylim=(h, 0), aspect='equal')
    plt.axis('off')
    fig.tight_layout()
    return fig
