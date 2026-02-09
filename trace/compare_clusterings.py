import numpy as np
np.set_printoptions(threshold=10000, suppress=True)
import os
os.chdir("..")
import random
from scipy.spatial.distance import jensenshannon

from trace.core import  TrajectoryManager, env_metadata
from trace.clustering import k_means, k_medoids, cluster_connections, aggregate_policies
from trace.visuals import sankey, cluster_scatter, tsne_transform, grid_arrows
from trace.behavior import BayesianDSTPolicy

# config
graph_labels = [('TSNE of Conditioned Policies (Frobenius distance)', 'Dimension 1', 'Dimension 2'),
                ('Total reward of episode', 'Treasure', 'Time'),]
save = False
filepath  = "data/dst_ground_truth.json"
env_id = 'deep-sea-treasure-v0'
cluster=k_medoids

def action_clarity_sparse(prob_matrix):
    var = np.var(prob_matrix, axis=2)
    visited = prob_matrix.var(axis=2) != 0

    if not np.any(visited):
        return 0.0

    decisiveness = var[visited].mean()
    sparsity = 1.0 - visited.mean()

    return decisiveness * sparsity

def main():
    h, w = env_metadata[env_id]['observations_dim']
    obs_space = np.meshgrid(np.arange(h), np.arange(w))

    print('loading file')
    manager = TrajectoryManager(env_id=env_id).load(filepath)

    rewards = manager.accrued_reward()
    ac_seq = manager.sequence(key='actions', flatten=False, pad=None)
    obs_seq = manager.sequence(key='observations', flatten=False, pad=None)
    #dists = manager.distribution(key='actions')
    print('making policies')
    policies = [BayesianDSTPolicy(obs_space=obs_space, num_actions=4, alpha=0.5).fit(obs, acs) # todo metadata inside
                for obs, acs in zip(obs_seq, ac_seq)]
    print('calculating distance')
    P = np.stack([p.prob_matrix().ravel() for p in policies])
    norms = np.sum(P * P, axis=1)
    dist2 = norms[:, None] + norms[None, :] - 2.0 * P @ P.T
    dist2 = np.maximum(dist2, 0.0)
    distance = np.sqrt(dist2)
    print('starting clustering')
    labels, centers = [], []
    for i, data in enumerate([distance, rewards]):
        l, c = cluster(data, k=3)
        labels.append(l)
        centers.append(c)
    print('starting aggregation')
    c_obs, c_ac = aggregate_policies(ac_seq, obs_seq, labels[0])
    c_pols = [BayesianDSTPolicy(obs_space=obs_space, num_actions=4, alpha=0.5).fit(obs, acs) for obs, acs in zip(c_obs, c_ac)]
    print('starting random clustering')
    rnd_labels = [random.randint(0, 2) for _ in range(len(ac_seq))]
    rnd_obs, rnd_ac = aggregate_policies(ac_seq, obs_seq, rnd_labels)
    rnd_pols = [BayesianDSTPolicy(obs_space=obs_space, num_actions=4, alpha=0.5).fit(obs, ac)
                    for obs, ac in zip(rnd_obs, rnd_ac)]

    print(np.var(rnd_pols[2].prob_matrix(), axis=2))
    print()
    print(np.var(c_pols[2].prob_matrix(), axis=2))

    for p1, p2 in zip(c_pols, rnd_pols):
        c1, c2 = action_clarity_sparse(p1.prob_matrix()), action_clarity_sparse(p2.prob_matrix())
        print(f'True: {c1:.2f} Random: {c2:.2f}')

    return
    true_score = demultiplexing_score(c_pols)
    rnd_score = demultiplexing_score(rnd_pols)

    print(f"True clustering JS:   {true_score:.4f}")
    print(f"Random clustering JS: {rnd_score:.4f}")
    print(f"Ratio: {true_score / (rnd_score + 1e-8):.2f}x")


if __name__ == "__main__":
    main()

