import numpy as np
np.set_printoptions(threshold=10000, suppress=True)

import os
os.chdir("..")

from trace.core import  TrajectoryManager
from trace.clustering import (k_means, k_medoids, cluster_connections, gaussian_mixture, dirichlet_process_mixture,
                              cluster_connections, aggregate_policies)
from trace.visuals import sankey, cluster_scatter, tsne_transform, grid_arrows
from trace.behavior import BayesianPolicy, frobenius

# config
graph_labels = [('TSNE of Conditioned Policies (Frobenius distance)', 'Dimension 1', 'Dimension 2'),
                ('Total reward of episode', 'Treasure', 'Time'),]
save = False
filepath  = "data/38_dst_ipro.json"
env_id = 'deep-sea-treasure-v0'
cluster=k_medoids


def main():
    manager = TrajectoryManager(env_id).load(filepath)
    rewards = manager.accrued_reward()
    obs_seq, ac_seq = manager.policy_data(flatten=False, pad=None)
    policies = [BayesianPolicy(env_id, alpha=0.5).fit(obs, acs)for obs, acs in zip(obs_seq, ac_seq)]
    distance = frobenius(policies)


    labels, centers = [], []
    for i, data in enumerate([distance, rewards]):
        l, c = cluster(data, k=3)
        if data.shape[1] > 2: data, c = tsne_transform(data, c)

        labels.append(l)
        centers.append(c)
        fig = cluster_scatter(data, l, c, color_id=i, graph_labels=graph_labels[i])
        if save: fig.savefig(f"plots/scatter{i}.png")
        fig.show()

    sankey_fig = sankey(*cluster_connections(labels))
    if save: sankey_fig.write_image(f"plots/sankey.png")
    sankey_fig.show()

    c_obs, c_ac = aggregate_policies(ac_seq, obs_seq, labels[0])
    for i in range(len(c_obs)):
        policy = BayesianPolicy(env_id, alpha=0.5).fit(c_obs[i], c_ac[i])
        grid_arrows(policy, title=f'Behavior Cluster {i}')


if __name__ == "__main__":
    main()

