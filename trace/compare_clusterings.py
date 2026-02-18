import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(threshold=10000, suppress=True, precision=3)
import os
os.chdir("..")
from json import load

from trace.core import  TrajectoryManager, env_metadata, synthetic_stochastic_points
from trace.clustering import k_means, k_medoids, spectral, gaussian_mixture, cluster_connections, aggregate_policies
from trace.behavior import BayesianPolicy, frobenius, overlap
from trace.visuals import grid_arrows

# config
graph_labels = [('TSNE of Conditioned Policies (Frobenius distance)', 'Dimension 1', 'Dimension 2'),
                ('Total reward of episode', 'Treasure', 'Time'),]
save = False
filepath  = "data/38_dst_ipro.json"
env_id = 'deep-sea-treasure-v0'
cluster=spectral
k = 2

def sparse_action_clarity(policy):
    total_states = np.prod([len(axis) for axis in policy.obs_space], dtype=int)
    if (visited_states := len(policy.counts)) == 0: return 0.0

    variances = [np.var(policy.action_probs(s)) for s in policy.counts.keys()]
    decisiveness = np.mean(variances)
    sparsity = 1.0 - (visited_states / total_states)

    return decisiveness * sparsity



def sparse_action_clarity_entropy(policy, eps=1e-12):
    total_states = np.prod([len(axis) for axis in policy.obs_space], dtype=int)
    if (visited_states := len(policy.counts)) == 0: return 0.0

    max_entropy = np.log(policy.num_actions)
    decisiveness = []
    for s in policy.counts.keys():
        p = policy.action_probs(s)
        h = -np.sum(p * np.log(p + eps))
        decisiveness.append(1.0 - h/max_entropy)

    sparsity = 1.0 - (visited_states / total_states)

    return np.mean(decisiveness) * sparsity



def main():
    manager = TrajectoryManager(env_id).load(filepath)
    obs_seq, ac_seq = manager.policy_data(flatten=False, pad=None)
    #dists = manager.distribution(key='actions')
    policies = [BayesianPolicy(env_id, alpha=0.5).fit(obs, acs) for obs, acs in zip(obs_seq, ac_seq)]


    cl_labels = cluster(overlap(policies), k=k)
    cl_pols = [BayesianPolicy(env_id, alpha=0.5).fit(obs, acs)
               for obs, acs in zip(*aggregate_policies(obs_seq, ac_seq, cl_labels))]

    rnd_labels = np.random.randint(0, k, size=len(cl_labels))
    rnd_pols = [BayesianPolicy(env_id, alpha=0.5).fit(obs, ac)
        for obs, ac in zip(*aggregate_policies(obs_seq, ac_seq, rnd_labels))]

    obs_seq, ac_seq = manager.policy_data(flatten=True, pad=None)
    universal_pol = BayesianPolicy(env_id, alpha=0.5).fit(obs_seq, ac_seq)

    def metric_report(func):
        universal_score = func(universal_pol)
        true_score = [func(pol) for pol in cl_pols]
        rnd_score = [func(pol) for pol in rnd_pols]

        print(f"Without clustering:  {universal_score:.2f}")
        print(f"Gaussian clustering: {np.array(true_score)}")
        print(f"Random clustering:   {np.array(rnd_score)}")

    print('Variance')
    metric_report(sparse_action_clarity)
    print('Entropy')
    metric_report(sparse_action_clarity_entropy)




if __name__ == "__main__":
    main()

