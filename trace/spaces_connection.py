import numpy as np
np.set_printoptions(threshold=10000)
import os
os.chdir("..")

from typing import List
from trace.core import  TrajectoryManager, env_metadata
from trace.clustering import k_means, k_medoids, cluster_connections, gaussian_mixture, dirichlet_process_mixture
from trace.visuals import sankey, cluster_scatter, tsne_transform, grid_arrows
from trace.behavior import behavior_report, reward_report, BayesianDSTPolicy, wasserstein_dist, l2_cost


def multi_graph(data_3d, graph_labels: list, save: bool = False):
    labels, centers, plt_figs = [], [], []
    for i, data in enumerate(data_3d):
        #l, c = k_means(data, k=3)
        l, c = k_medoids(data, k=3)
        #l, c = gaussian_mixture(data, k_max=10)
        #l, c = dirichlet_process_mixture(data, k_max=10)


        if data.shape[1] > 2: data, c = tsne_transform(data, c)

        labels.append(l)
        centers.append(c)
        plt_figs.append(cluster_scatter(data, l, c, color_id=i, graph_labels=graph_labels[i]))

    sankey_fig = sankey(*cluster_connections(labels))

    if save:
        for i, fig in enumerate(plt_figs): fig.savefig(f"plots/scatter{i}.png")
        sankey_fig.write_image(f"plots/sankey.png")
    else:
        for fig in plt_figs: fig.show()
        sankey_fig.show()
    return labels, centers


def main():
    graph_labels = [
        ('TSNE of Conditioned Policies (Frobenius distance)', 'Dimension 1', 'Dimension 2'),
        ('Total reward of episode', 'Treasure', 'Time'),
    ]
    filepath = "data/38_dst_ipro.json"
    env_id = 'deep-sea-treasure-v0'
    actions = env_metadata[env_id]['actions']
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

    cost = l2_cost(list(actions.values()))
    distance = np.array([[np.linalg.norm(p1.prob_matrix() - p2.prob_matrix()) for p2 in policies] for p1 in policies])

    print(f'Distance features: {distance.shape} Reward features: {rewards.shape}')

    labels, centers = multi_graph([distance, rewards], graph_labels, save=False)
    return
    for c_id in range(np.max(labels[0]) + 1):
        c_ac = [ac for i, ac in enumerate(ac_seq) if labels[0][i] == c_id]
        c_obs = [obs for i, obs in enumerate(obs_seq) if labels[0][i] == c_id]

        policy = BayesianDSTPolicy(obs_space=obs_space, num_actions=4, alpha=0.5)
        policy.fit(c_obs, c_ac)
        grid_arrows(policy, title=f"Behavior Cluster {c_id}")
        #all_probs = np.array([[policy.action_probs([x, y]) for x in range(12)] for y in range(12)])
        #print(f"Behavior Cluster {c_id} variance: {np.var(all_probs)}")


if __name__ == "__main__":
    main()

