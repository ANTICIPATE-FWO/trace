import numpy as np
np.set_printoptions(threshold=10000)
from json import load
import os
os.chdir("..")

from policy_explainability.utils import filter_traj, rewards_per_episode
from policy_explainability.clustering import k_means, cluster_connections, gaussian_mixture
from policy_explainability.visuals import sankey, cluster_scatter, tsne_transform
from policy_explainability.behavior import policy_dist, policy_sequence


def multi_graph(data_3d, graph_labels: list, save: bool = False):
    labels, centers, plt_figs = [], [], []
    for i, data in enumerate(data_3d):
        #l, c = k_means(data, k=4)
        l, c = gaussian_mixture(data, k_max=4)

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


def main():
    graph_labels = [
        ('TSNE of Action Sequences', 'Dimension 1', 'Dimension 2'),
        ('Total reward of episode', 'Treasure', 'Time'),
    ]
    filepath = "data/38_dst_ipro.json"
    with open(filepath, "r") as f: trajectories = filter_traj(load(f))

    rewards = rewards_per_episode(trajectories)
    action_dist = policy_dist(trajectories, normalize=True)
    action_sequence = policy_sequence(trajectories)
    print(action_sequence)
    print(f"Action Dist: {action_dist.shape} Action Seq: {action_sequence.shape} Rewards: {rewards.shape}")
    multi_graph([action_sequence, rewards], graph_labels, save=False)


if __name__ == "__main__":
    main()