import os
os.chdir('../')

from core import TrajectoryManager
from trace.visuals import grid_trajectories, temporal_alignment, decision_tree, boxplot



def cluster_graphs(manager:TrajectoryManager, ind:int|str='u', color:str='blue', max_len:int|None=None, directory:str|None=None):
    obs, acs, rew = manager.conditioning_features(pad=None, flatten=True)
    metadata = manager.metadata
    time_range = (0, max_len) if max_len is not None else None

    title = f'Cluster {ind}: {len(manager)} trajectories'
    figs = {
        f"trajectory{ind}.png": grid_trajectories(obs, acs, title=title, color=color),
        f'temporal{ind}.png': temporal_alignment(acs, metadata['actions'], time_range=time_range, title=title),
        f'decision_tree{ind}.png': decision_tree(obs, acs, metadata, title=title),
        f'boxplot{ind}.png': boxplot(rew, title=title),
    }

    if directory:
        for filename, fig in figs.items():
            fig.savefig(os.path.join(directory, filename))
    return figs
