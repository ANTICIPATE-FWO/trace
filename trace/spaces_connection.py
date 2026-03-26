import numpy as np
np.set_printoptions(threshold=10000, suppress=True)

import os
os.chdir("..")

from trace.core import  TrajectoryManager, colors, aggregate_policies
from trace.clustering import k_means, k_medoids, spectral, gaussian_mixture, dirichlet_mixture, cluster_connections
from trace.visuals import sankey, cluster_scatter, grid_trajectories, grid_arrows, temporal_alignment, decision_tree
from trace.behavior import BayesianPolicy, distance_matrix

# config
metric = 'agreement'
k=2

filepath  = "data/38_dst_ipro.json"
env_id = 'deep-sea-treasure-v0'
plot_directory = f"plots/{metric}"
cluster_functions= [k_medoids, k_medoids]

save, show = True, False
graph_labels = [(f'TSNE of Conditioned Policies ({metric} metric)', '', ''),
                ('Total reward of episode', 'Treasure', 'Time'),]


def main():
    manager = TrajectoryManager(env_id).load(filepath)
    rewards = manager.accrued_reward()
    obs_seq, ac_seq = manager.policy_data(flatten=False, pad=None)

    policies = [BayesianPolicy(env_id, alpha=0.5).fit(obs, acs) for obs, acs in zip(obs_seq, ac_seq)]
    behavior_features = distance_matrix(policies, metric=metric)
    print(f'\rInitialized {len(policies)} policies')

    labels, figs = [], []
    for i, data in enumerate([behavior_features, rewards]):
        labels.append(cluster_functions[i](data, k=k))
        figs.append((cluster_scatter(data, labels[-1], color_id=i, graph_labels=graph_labels[i]), f"scatter{i}.png"))
    figs.append((sankey(*cluster_connections(labels)), "sankey.png"))

    if env_id != 'deep-sea-treasure-v0': return

    c_obs, c_ac = aggregate_policies(obs_seq, ac_seq, labels[0])
    for c in range(len(c_obs)):
        policy = BayesianPolicy(env_id, alpha=0.5).fit(c_obs[c], c_ac[c])
        title = f'Behavior Cluster {c + 1}: {len(c_obs[c])} episodes'
        color = colors[0][c]
        figs.extend([
            (grid_trajectories(c_obs[c], title=title, color=color, alpha=0.01), f"trajectory{c}.png"),
            (grid_arrows(policy, title=title, color=color), f"arrows{c}.png"),
            (temporal_alignment(c_ac[c], env_id, title=title), f'temporal_alignment{c}.png'),
            (decision_tree(c_obs[c], c_ac[c], env_id, title=title), f'decision_tree{c}.png')
        ])
    obs_univ, acs_univ = aggregate_policies(obs_seq, ac_seq, [0] * len(policies))
    policy_univ = BayesianPolicy(env_id, alpha=0.5).fit(obs_univ[0], acs_univ[0])

    title = f'Universal Policy: {len(obs_univ[0])} episodes'
    figs.extend([
        (grid_trajectories(obs_univ[0], title=title, color='white'), "trajectory_universal.png"),
        (grid_arrows(policy_univ, title=title), "arrows_universal.png"),
        (temporal_alignment(acs_univ[0], env_id, title=title), "temporal_alignment_universal.png"),
        (decision_tree(obs_univ[0], acs_univ[0], env_id, title=title), "decision_tree_universal.png")
    ])

    if save:
        for fig, filename in figs:
            path = os.path.join(plot_directory, filename)
            if hasattr(fig, "write_image"): fig.write_image(path)
            elif hasattr(fig, "savefig"): fig.savefig(path)
            else: raise TypeError(f"Unsupported figure type for {filename}: {type(fig)}")
    if show:
        for fig, _ in figs: fig.show()

if __name__ == "__main__":
    main()

