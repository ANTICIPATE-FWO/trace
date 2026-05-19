import os
os.chdir('../')

import numpy as np
from trace.core import TrajectoryManager
from trace.behavior import decisiveness

def ablation(labels:list|np.ndarray, metadata:dict, filepath:str):
    labels, k = np.array(labels), max(labels) + 1
    rnd_labels = np.random.randint(0, k, size=len(labels))
    universal = TrajectoryManager(metadata).load(filepath, filtering=False)

    cluster_report([universal], 'universal')
    cluster_report([universal.subset(labels == l)for l in range(k)], 'trace')
    cluster_report([universal.subset(rnd_labels == l)for l in range(k)], 'random')


def cluster_report(clusters:list[TrajectoryManager], method:str):
    print(f"\n{method.upper()}")

    cluster_vars, cluster_ents = [], []

    for i, cluster in enumerate(clusters):
        obs, acs, _ = cluster.conditioning_features(pad=None, flatten=True)

        cluster_vars.append(np.mean(decisiveness(obs, acs, cluster.metadata)[0]))
        cluster_ents.append(np.mean(decisiveness(obs, acs, cluster.metadata, entropy=True)[0]))

        print(f"\tCluster {i}:\n\t\tVariance: {cluster_vars[-1]:.3f}\tEntropy: {cluster_ents[-1]:.3f}")

    print("\tAverage")
    print(f"\t\tVariance: {np.mean(cluster_vars):.3f}\tEntropy: {np.mean(cluster_ents):.3f}")


def example():
    from yaml import safe_load
    labels = []
    metadata = safe_load(open('trace/configs/minetrain.yaml', 'r'))
    filepath = 'data/minetrain_ground_truth.json'
    ablation(labels, metadata, filepath)


if __name__ == '__main__':
    example()

