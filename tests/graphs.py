import os
os.chdir('../')
import numpy as np
from yaml import safe_load

from trace.core import TrajectoryManager
from trace.visuals import temporal_alignment, decision_tree, boxplot
from visuals import grid_trajectories


def cluster_graphs(manager:TrajectoryManager, ind:int|str='u', color:str='blue', max_len:int|None=None, directory:str|None=None):
    obs, acs, rew = manager.conditioning_features(per_point=False)
    metadata = manager.metadata
    time_range = (0, max_len) if max_len is not None else None

    title = f'Cluster {ind}: {len(manager)} trajectories'
    figs = {
        f"trajectory{ind}.png": grid_trajectories(manager, title=title, color=color, abstr_frame=False),
        f'temporal{ind}.png': temporal_alignment(acs, metadata['actions'], time_range=time_range, title=title),
        f'decision_tree{ind}.png': decision_tree(obs, acs, metadata, title=title),
        f'boxplot{ind}.png': boxplot(rew, title=title),
    }

    if directory:
        for filename, fig in figs.items():
            fig.savefig(os.path.join(directory, filename))
    return figs


def integration():
    from trace.clustering.cached_labels.minetrain_k_medoids_kl import labels
    policy, config, metric = 'ground_truth', 'minetrain', 'kl'
    colors = safe_load(open(f'trace/configs/colors.yaml'))['warm']
    universal = TrajectoryManager(config).load(policy)

    graph_directory = f"plots/{universal.metadata['file_prefix']}/{policy}/{metric}"
    for file in os.listdir(graph_directory):
        path = os.path.join(graph_directory, file)
        if os.path.isfile(path): os.remove(path)

    assert len(labels) == len(universal), f'Length mismatch {len(labels)} and {len(universal)}'
    clusters = [universal.subset(np.array(labels) == l) for l in range(max(np.array(labels)) + 1)]

    for i, manager in enumerate([*clusters, universal]):
        ind = i if i < len(clusters) else 'u'
        cluster_graphs(manager, ind=ind, directory=graph_directory, color=colors[i])


if __name__ == '__main__':
    integration()
