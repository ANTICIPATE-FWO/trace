import numpy as np
np.set_printoptions(threshold=10000)
import os
os.chdir("..")

from trace.core import  TrajectoryManager
from trace.clustering import k_means, cluster_connections, gaussian_mixture, dirichlet_process_mixture
from trace.visuals import sankey, cluster_scatter, tsne_transform, grid_arrows
from trace.behavior import behavior_report, reward_report, BayesianDSTPolicy


def multi_graph(data_3d, graph_labels: list, save: bool = False):
    labels, centers, plt_figs = [], [], []
    for i, data in enumerate(data_3d):
        #l, c = k_means(data, k=3)
        #l, c = gaussian_mixture(data, k_max=10)
        l, c = dirichlet_process_mixture(data, k_max=10)


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
        ('TSNE of Action Sequences', 'Dimension 1', 'Dimension 2'),
        ('Total reward of episode', 'Treasure', 'Time'),
    ]
    #filepath = "data/38_dst_ipro.json"
    #filepath = "data/2_mc_ipro.json"
    filepath = "data/dst_ground_truth.json"

    manager = TrajectoryManager(env_id='deep-sea-treasure-v0').load(filepath)

    rewards = manager.rewards_ep()
    action_dist = manager.distribution()
    ac_seq = manager.sequence(key='actions')
    obs_seq = manager.sequence(key='observations')
    print(f"Action Dist: {action_dist.shape} Action Seq: {ac_seq.shape} Rewards: {rewards.shape}")

    labels, centers = multi_graph([ac_seq, rewards], graph_labels, save=False)

    for c_id in range(np.max(labels[0]) + 1):
        c_ac = ac_seq[labels[0] == c_id]
        c_obs = obs_seq[labels[0] == c_id]
        policy = BayesianDSTPolicy(num_actions=4, alpha=0.5)
        policy.fit(c_obs, c_ac)
        grid_arrows(policy, 12, 12, title=f"Behavior Cluster {c_id}")
        all_probs = np.array([[policy.action_probs([x, y]) for x in range(12)] for y in range(12)])
        print(f"Behavior Cluster {c_id} variance: {np.var(all_probs)}")


if __name__ == "__main__":
    main()

