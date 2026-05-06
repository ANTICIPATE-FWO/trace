import numpy as np
np.set_printoptions(threshold=10000, suppress=True)

import os
os.chdir("..")

from yaml import safe_load

from trace.core import  TrajectoryManager, aggregate_policies, tree_features, traj_dict
from trace.clustering import k_means, k_medoids, spectral, gaussian_mixture, dirichlet_mixture, cluster_connections
from trace.visuals import sankey, cluster_scatter, grid_trajectories, grid_arrows, temporal_alignment, decision_tree, minecart_trajectories
from trace.behavior import BayesianPolicy, distance_matrix

# config
filepath  = "data/minetrain_ipro.json" # 38_dst_ipro.json
config, method, metric = 'minetrain', 'ipro', 'frobenius'
k, cluster_functions= 3, [k_medoids, k_medoids]
save, show = True, False

# global params
metadata = safe_load(open(f"trace/configs/{config}.yaml", "r"))
colors = safe_load(open("trace/configs/colors.yaml", "r"))
plot_directory = f"plots/{metadata['file_prefix']}/{method}/{metric}"


def main():
    # Loading
    manager = TrajectoryManager(metadata).load(filepath)
    observations, actions, reward_features = manager.policy_data(flatten=False, pad=None)

    if method == 'ground_truth':
        rewards = manager.sequence(key='rewards')
        unique = list(set(str(r) for r in reward_features))
        labels = [unique.index(str(r)) for r in reward_features]
        observations, actions, reward_features = aggregate_policies(observations, actions, rewards, labels)
        manager = TrajectoryManager(metadata).load(traj_dict(observations, actions, reward_features))
        observations, actions, reward_features = manager.policy_data(flatten=False, pad=None)

    max_len = max(len(episode) for point in observations for episode in point)
    policies = [BayesianPolicy(metadata, alpha=0.5).fit(obs, acs) for obs, acs in zip(observations, actions)]
    behavior_features = distance_matrix(policies, metric=metric)

    # Clustering
    cluster_config = [(k_medoids, behavior_features),
                      (k_medoids, reward_features)]
    labels = [cluster(feature, k=k) for cluster, feature in cluster_config]
    cluster_obs, cluster_acs, _ = aggregate_policies(observations, actions, reward_features, labels[0])
    cluster_policies = [BayesianPolicy(metadata, alpha=0.5).fit(obs, acs) for obs, acs in zip(cluster_obs, cluster_acs)]
    universal_obs, universal_acs, _ = aggregate_policies(observations, actions, reward_features, [0] * len(policies))
    universal_policy = BayesianPolicy(metadata, alpha=0.5).fit(universal_obs[0], universal_acs[0])

    # Plotting
    # todo add boxplot reward
    figs = [(cluster_scatter(behavior_features, labels[0], colors=colors['warm']), f"behavior_scatter.png"),
            (sankey(*cluster_connections(labels), colors), "sankey.png")]

    traj_frame = grid_trajectories if 'deep-sea-treasure' in metadata['env_id'] else minecart_trajectories
    for c_id in range(k):
        title = f'Behavior Cluster {c_id + 1}: {len(cluster_obs[c_id])} episodes'
        color = colors['warm'][c_id]
        figs.extend([
            #(grid_trajectories(c_obs[c], title=title, color=color, alpha=0.01), f"trajectory{c}.png"),
            (traj_frame(cluster_obs[c_id], cluster_acs[c_id], title=title, color=color), f"trajectory{c_id}.png"),
            (temporal_alignment(cluster_acs[c_id], metadata['actions'], time_range=(0, max_len), title=title), f'temporal_alignment{c_id}.png'),
            (decision_tree(*tree_features(cluster_obs[c_id], cluster_acs[c_id]), metadata, title=title), f'decision_tree{c_id}.png')
        ])

    title = f'Universal Policy: {len(universal_obs[0])} episodes'
    figs.extend([
        (traj_frame(universal_obs[0], universal_acs[0], title=title, color='white'), "trajectory_universal.png"),
        (temporal_alignment(universal_acs[0], metadata['actions'], time_range=(0, max_len), title=title), "temporal_alignment_universal.png"),
        (decision_tree(*tree_features(universal_obs[0], universal_acs[0]), metadata, title=title), "decision_tree_universal.png")
    ])


    # Saving & showing
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

