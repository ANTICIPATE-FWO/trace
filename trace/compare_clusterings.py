import numpy as np
np.set_printoptions(threshold=10000, suppress=True, precision=3)
import os
os.chdir("..")

from trace.core import  TrajectoryManager, env_metadata, synthetic_stochastic_points
from trace.clustering import k_means, k_medoids, spectral, gaussian_mixture, cluster_connections, aggregate_policies
from trace.behavior import BayesianPolicy, distance_matrix
from trace.networks import component_labels

# config
filepath  = "data/38_dst_ipro.json"
env_id = 'deep-sea-treasure-v0'
cluster=k_medoids
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

    features = distance_matrix(policies, metric='agreement', smoothing=False)


    nx_labels = component_labels(features, threshold=0.3)
    nx_pols = [BayesianPolicy(env_id, alpha=0.5).fit(obs, acs)
               for obs, acs in zip(*aggregate_policies(obs_seq, ac_seq, nx_labels))]

    cl_labels = cluster(features, k=k, metric='precomputed')
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
        nx_score = [func(pol) for pol in nx_pols]
        rnd_score = [func(pol) for pol in rnd_pols]

        print(f"Without clustering:  {universal_score:.2f}")
        print(f"True clustering   : {np.array(true_score)} -> {np.mean(true_score): .2f}")
        print(f"Network components: {np.array(nx_score)} -> {np.mean(nx_score): .2f}")
        print(f"Random clustering :   {np.array(rnd_score)} -> {np.mean(rnd_score): .2f}")

    print('-' * 20)
    print('Variance')
    metric_report(sparse_action_clarity)
    print('-' * 20)
    print('Entropy')
    metric_report(sparse_action_clarity_entropy)




if __name__ == "__main__":
    main()

