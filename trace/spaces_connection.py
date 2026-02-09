import numpy as np
np.set_printoptions(threshold=10000, suppress=True)

import os
os.chdir("..")

from trace.core import  TrajectoryManager, env_metadata
from trace.clustering import (k_means, k_medoids, cluster_connections, gaussian_mixture, dirichlet_process_mixture,
                              cluster_connections, aggregate_policies)
from trace.visuals import sankey, cluster_scatter, tsne_transform, grid_arrows
from trace.behavior import BayesianDSTPolicy

# config
graph_labels = [('TSNE of Conditioned Policies (Frobenius distance)', 'Dimension 1', 'Dimension 2'),
                ('Total reward of episode', 'Treasure', 'Time'),]
save = False
filepath  = "data/38_dst_ipro.json"
env_id = 'deep-sea-treasure-v0'
cluster=k_medoids


def main():
    h, w = env_metadata[env_id]['observations_dim']
    obs_space = np.meshgrid(np.arange(h), np.arange(w))

    manager = TrajectoryManager(env_id=env_id).load(filepath)
    rewards = manager.accrued_reward()
    ac_seq = manager.sequence(key='actions', flatten=False, pad=None)
    obs_seq = manager.sequence(key='observations', flatten=False, pad=None)
    policies = [
        BayesianDSTPolicy(obs_space=obs_space, num_actions=4, alpha=0.5).fit(obs, acs) # todo metadata inside
        for obs, acs in zip(obs_seq, ac_seq)
    ]
    distance = np.array([[np.linalg.norm(p1.prob_matrix() - p2.prob_matrix()) for p2 in policies] for p1 in policies])


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
        policy = BayesianDSTPolicy(obs_space=obs_space, num_actions=4, alpha=0.5).fit(c_obs[i], c_ac[i])
        grid_arrows(policy, title=f'Behavior Cluster {i}')


if __name__ == "__main__":
    main()

