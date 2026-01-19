import numpy as np
from json import load
import os
os.chdir("..")

from policy_explainability.analysis import filter_traj, rewards_per_episode, policy_dist
from policy_explainability.utils import k_means, cluster_connections
from policy_explainability.visuals import sankey, cluster_scatter


def multi_graph(data_3d, graph_labels: list, save: bool = False):
    labels, centers = [], []
    for data in data_3d:
        l, c = k_means(data, k=3)
        labels.append(l)
        centers.append(c)
    plt_figs = [
        cluster_scatter(data_3d[i], labels[i], centers[i], color_id=i, graph_labels=graph_labels[i])
        for i in range(len(data_3d))
    ]
    sankey_fig = sankey(*cluster_connections(labels))

    if save:
        for i, fig in enumerate(plt_figs): fig.savefig(f"plots/scatter{i}.png")
        sankey_fig.write_image(f"plots/sankey.png")
    else:
        for fig in plt_figs: fig.show()
        sankey_fig.show()


def main():
    graph_labels = [
        ('Normalized action distribution', 'Action 1 (down)', 'Action 3 (right)'),
        ('Total reward of episode', 'Treasure', 'Time'),
    ]
    filepath = "data/38_dst_ipro.json"
    with open(filepath, "r") as f: trajectories = filter_traj(load(f))

    rewards = rewards_per_episode(trajectories)
    action_dist = policy_dist(trajectories, normalize=True)[:, [1,3]] # only action 1 and 3 for dst
    multi_graph(np.stack([action_dist, rewards]), graph_labels, save=True)


if __name__ == "__main__":
    main()