import numpy as np
np.set_printoptions(threshold=10000, suppress=True, precision=3)
import os
os.chdir("..")
from json import load

from trace.core import  TrajectoryManager, env_metadata, synthetic_stochastic_points
from trace.clustering import k_means, k_medoids, gaussian_mixture, cluster_connections, aggregate_policies
from trace.behavior import BayesianPolicy, frobenius
from trace.visuals import grid_arrows

# config
graph_labels = [('TSNE of Conditioned Policies (Frobenius distance)', 'Dimension 1', 'Dimension 2'),
                ('Total reward of episode', 'Treasure', 'Time'),]
save = False
filepath  = "data/dst_ground_truth.json"
env_id = 'deep-sea-treasure-v0'
cluster=gaussian_mixture
k = 3

def sparse_action_clarity(prob_matrix):
    var = np.var(prob_matrix, axis=2)
    visited = prob_matrix.var(axis=2) != 0

    if not np.any(visited): return 0.0

    decisiveness = var[visited].mean()
    sparsity = 1.0 - visited.mean()

    return decisiveness * sparsity

def sparse_action_clarity_entropy(prob_matrix, eps=1e-12):
    probs = prob_matrix / (prob_matrix.sum(axis=2, keepdims=True) + eps)
    entropy = -np.sum(probs * np.log(probs + eps), axis=2)
    visited = prob_matrix.var(axis=2) != 0

    if not np.any(visited): return 0.0

    decisiveness = (1.0 - entropy[visited]).mean()
    sparsity = 1.0 - visited.mean()

    return decisiveness * sparsity


def main():
    ground_truth = load(open(filepath, 'rb'))
    manager = TrajectoryManager(env_id).load(synthetic_stochastic_points(ground_truth))
    obs_seq, ac_seq = manager.policy_data(flatten=False, pad=None)

    #dists = manager.distribution(key='actions')
    policies = [BayesianPolicy(env_id, alpha=0.5).fit(obs, acs) for obs, acs in zip(obs_seq, ac_seq)]

    # can be used for behavioral clustering
    ac_seq_pad = manager.sequence(key='actions', flatten=True, pad=-1)
    distance = frobenius(policies)

    cl_labels, _ = cluster(distance, k=k)
    cl_pols = [BayesianPolicy(env_id, alpha=0.5).fit(obs, acs)
               for obs, acs in zip(*aggregate_policies(ac_seq, obs_seq, cl_labels))]

    rnd_labels = np.random.randint(0, k, size=len(cl_labels))
    rnd_pols = [BayesianPolicy(env_id, alpha=0.5).fit(obs, ac)
        for obs, ac in zip(*aggregate_policies(ac_seq, obs_seq, rnd_labels))]

    obs_seq, ac_seq = manager.policy_data(flatten=True, pad=None)
    universal_pol = BayesianPolicy(env_id, alpha=0.5).fit(obs_seq, ac_seq)

    print(np.var(rnd_pols[-1].prob_matrix(), axis=2))
    print()
    print(np.var(cl_pols[-1].prob_matrix(), axis=2))
    print()

    def metric_report(func):
        universal_score = func(universal_pol.prob_matrix())
        true_score = [func(pol.prob_matrix()) for pol in cl_pols]
        rnd_score = [func(pol.prob_matrix()) for pol in rnd_pols]

        print(f"Without clustering: {universal_score:.2f}")
        print(f"True clustering:    {np.array(true_score)}")
        print(f"Random clustering:  {np.array(rnd_score)}")

    print('Variance')
    metric_report(sparse_action_clarity)
    print('Entropy')
    metric_report(sparse_action_clarity_entropy)


if __name__ == "__main__":
    main()

