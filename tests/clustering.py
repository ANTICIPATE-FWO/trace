import os
os.chdir('../')


from collections.abc import Callable

from trace.behavior import EmpiricalDistribution, distance_matrix
from trace.core import TrajectoryManager


def behavior_clustering(manager:TrajectoryManager, cluster:Callable, k:int, metric:str='agreement', save:bool=True):
    obs, acs, _ = manager.conditioning_features(pad=None, flatten=True)
    behavior_models = [EmpiricalDistribution(manager.metadata, alpha=0.5).fit(o, a) for o, a in zip(obs, acs)]
    features = distance_matrix(behavior_models, metric=metric, norm=True)
    labels = cluster(features, k=k, metric='precomputed')

    if save:
        prefix, func = manager.metadata['file_prefix'], cluster.__name__
        with open(f'trace/clustering/cached_labels/{prefix}_{func}_{metric}.py', 'w') as f:
            f.write('labels = ')
            f.write(str(labels.tolist()))

    return [manager.subset(labels == l) for l in range(k)], labels