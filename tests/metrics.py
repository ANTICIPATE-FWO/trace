import os
os.chdir('../')

from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


from trace.core import TrajectoryManager
from trace.behavior import decisiveness, EmpiricalDistribution

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

        cluster_vars.append(np.mean(decisiveness(obs, acs, cluster.metadata)[0]))
        cluster_ents.append(np.mean(decisiveness(obs, acs, cluster.metadata, entropy=True)[0]))

        print(f"\tCluster {i}:\n\t\tVariance: {cluster_vars[-1]:.3f}\tEntropy: {cluster_ents[-1]:.3f}")
    print("\tAverage")
    print(f"\t\tVariance: {np.mean(cluster_vars):.3f}\tEntropy: {np.mean(cluster_ents):.3f}")


def overlap_analysis(policies:list|np.ndarray):
    common_states = defaultdict(int)

    for i in range(len(policies)):
        vi = set(policies[i].get_visited())

        for j in range(i+1, len(policies)):
            vj = set(policies[j].get_visited())
            common_states[len(vi & vj)] += 1

    x = sorted(common_states.keys())
    y = [common_states[k] for k in x]

    plt.bar(x, y)

    plt.xlabel('Number of common states')
    plt.ylabel('Number of trajectory pairs')
    plt.title('Overlap in pairs of trajectories')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('plots/minetrain/overlap.png')


def example():
    from yaml import safe_load
    from trace.clustering.cached_labels.dst_k_medoids_kl import labels
    metadata = safe_load(open('trace/configs/minetrain.yaml', 'r'))
    manager = TrajectoryManager(metadata).load('data/minetrain_ground_truth.json', filtering=True)
    obs, acs, _ = manager.conditioning_features(pad=None, flatten=True)

    policies = [EmpiricalDistribution(metadata).fit(o, a) for o,a in zip(obs, acs)]

    overlap_analysis(policies)




if __name__ == '__main__':
    example()

