import numpy as np
np.set_printoptions(threshold=10000, suppress=True)

import os
os.chdir("..")

from trace.core import  TrajectoryManager
from trace.clustering import (k_means, k_medoids, spectral, gaussian_mixture, dirichlet_mixture,
                              cluster_connections, aggregate_policies)
from trace.visuals import sankey, cluster_scatter, colors, grid_map, grid_arrows
from trace.behavior import BayesianPolicy, frobenius, overlap

# config
graph_labels = [('TSNE of Conditioned Policies (Frobenius distance)', 'Dimension 1', 'Dimension 2'),
                ('Total reward of episode', 'Treasure', 'Time'),]
save, show = False, True
filepath  = "data/38_dst_ipro.json"
env_id = 'deep-sea-treasure-v0'
plot_directory = "plots/frobenius_conditioned"
cluster_functions= [spectral, k_medoids]
k=2


def main():
    #todo clean up fig saving and showing
    manager = TrajectoryManager(env_id).load(filepath)
    rewards = manager.accrued_reward()
    obs_seq, ac_seq = manager.policy_data(flatten=False, pad=None)

    policies = [BayesianPolicy(env_id, alpha=0.5).fit(obs, acs)for obs, acs in zip(obs_seq, ac_seq)]
    print(f'Initialized {len(policies)} policies')


    labels= []
    for i, data in enumerate([overlap(policies), rewards]):
        labels.append(cluster_functions[i](data, k=k))

        fig = cluster_scatter(data, labels[-1], color_id=i, graph_labels=graph_labels[i])
        if save: fig.savefig(os.path.join(plot_directory, f"scatter{i}.png"))
        if show: fig.show()

    sankey_fig = sankey(*cluster_connections(labels))
    if save: sankey_fig.write_image(os.path.join(plot_directory, "sankey.png"))
    if show: sankey_fig.show()

    if env_id != 'deep-sea-treasure-v0': return
    c_obs, c_ac = aggregate_policies(obs_seq, ac_seq, labels[0])

    for c in range(len(c_obs)):
        policy = BayesianPolicy(env_id, alpha=0.5).fit(c_obs[c], c_ac[c])
        title = f'Behavior Cluster {c + 1}: {len(c_obs[c])} episodes'
        color = colors[0][c]
        figs = [grid_map(c_obs[c], title=title, color=color),
                grid_arrows(policy, title=title, color=color)]
        for i, fig in enumerate(figs):
            if save: fig.savefig(os.path.join(plot_directory, f"grid-cluster{c}-{i}.png"))
            if show: fig.show()

    un_obs, un_ac = aggregate_policies(obs_seq, ac_seq, [0]*len(ac_seq))
    universal_policy = BayesianPolicy(env_id, alpha=0.5).fit(un_obs[0], un_ac[0])
    fig = grid_arrows(universal_policy, title=f'Universal Policy: {len(un_obs[0])} episodes')

    if save: fig.savefig(os.path.join(plot_directory, "grid_universal.png"))
    if show: fig.show()


if __name__ == "__main__":
    main()

