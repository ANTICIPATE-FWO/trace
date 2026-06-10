import os
os.chdir('../')

import numpy as np
np.set_printoptions(precision=4)

from trace.core import TrajectoryManager
from trace.behavior import decisiveness, quantize


def ablation(universal:TrajectoryManager, labels:list|np.ndarray):
    labels, k = np.array(labels), max(labels) + 1
    rnd_labels = np.asarray(np.random.randint(0, k, size=len(labels)))

    cluster_report([universal], 'universal')
    cluster_report([universal.subset(labels == l)for l in range(k)], method='trace')
    cluster_report([universal.subset(rnd_labels == l)for l in range(k)], method='random')


def cluster_report(clusters:list[TrajectoryManager], method:str):
    print(f"\n{method.upper()}")
    cluster_vars, cluster_ents = [], []
    for i, cluster in enumerate(clusters):
        obs, acs, _ = cluster.conditioning_features(per_point=False)
        obs = quantize(obs, method='uni', bins=[100, 100, 6, 10, 10, 90, 90])

        cluster_vars.append(np.mean(1000*decisiveness(obs, acs, cluster.metadata)))
        cluster_ents.append(np.mean(1000*decisiveness(obs, acs, cluster.metadata, entropy=True)))

        print(f"\tCluster {i}:\n\t\tVariance: {cluster_vars[-1]:.5f}\tEntropy: {cluster_ents[-1]:.5f}")
    print("\tAverage")
    print(f"\t\tVariance: {np.mean(cluster_vars):.5f}\tEntropy: {np.mean(cluster_ents):.5f}")


if __name__ == '__main__':
    from trace.clustering.cached_labels import minetrain_k_medoids_kl
    ablation(
        universal=TrajectoryManager('minetrain').load('ground_truth'),
        labels=minetrain_k_medoids_kl.labels,
    )