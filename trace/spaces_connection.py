import numpy as np
np.set_printoptions(threshold=10000)
import os
os.chdir("..")

from trace.core import  TrajectoryManager
from trace.clustering import k_means, cluster_connections, gaussian_mixture, dirichlet_process_mixture
from trace.visuals import sankey, cluster_scatter, tsne_transform
from trace.behavior import behavior_report, reward_report


def multi_graph(data_3d, graph_labels: list, save: bool = False):
    labels, centers, plt_figs = [], [], []
    for i, data in enumerate(data_3d):
        #l, c = k_means(data, k=4)
        #l, c = gaussian_mixture(data, k_max=4)
        l, c = dirichlet_process_mixture(data, k_max=5)


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
    filepath = "data/38_dst_ipro.json"
    #filepath = "data/2_mc_ipro.json"


    manager = TrajectoryManager(actions=[0,1,2,3])
    manager.load_file(filepath)

    rewards = manager.rewards_ep()
    action_dist = manager.distribution()
    action_sequence = manager.sequence()
    print(f"Action Dist: {action_dist.shape} Action Seq: {action_sequence.shape} Rewards: {rewards.shape}")


    return
    labels, centers = multi_graph([action_sequence, rewards], graph_labels, save=False)

    for space, report in enumerate([behavior_report, reward_report]):
        for c_id in range(np.max(labels[space]) + 1):
            c_ac = action_sequence[labels[space] == c_id]
            c_r = rewards[labels[space] == c_id]
            report(c_ac, c_r, c_id + 1)

if __name__ == "__main__":
    main()