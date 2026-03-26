import numpy as np
import warnings
import os
from beautifultable import BeautifulTable

from trace.core import TrajectoryManager, aggregate_policies, tree_features
from trace.clustering import k_medoids
from trace.behavior import BayesianPolicy, distance_matrix, tree_rules, component_labels
from trace.visuals import decision_tree

os.chdir('../')
np.set_printoptions(threshold=10000, suppress=True, precision=3)
warnings.filterwarnings("ignore")

# config
filepath  = "data/38_dst_ipro.json"
env_id = 'deep-sea-treasure-v0'
cluster=k_medoids
k = 2


def sparse_action_clarity(obs:list|np.ndarray, acs:list|np.ndarray, env_id:str):
    policy = BayesianPolicy(env_id, alpha=0.5).fit(obs, acs)
    total_states = np.prod([len(axis) for axis in policy.obs_space], dtype=int)
    if (visited_states := len(policy.counts)) == 0: return 0.0

    variances = [np.std(policy.action_probs(s)) for s in policy.counts.keys()]
    decisiveness = np.mean(variances)
    sparsity = 1.0 - (visited_states / total_states)

    #return decisiveness / visited_states
    return decisiveness * sparsity



def sparse_action_clarity_entropy(obs:list|np.ndarray, acs:list|np.ndarray, env_id:str, eps:float=1e-12):
    policy = BayesianPolicy(env_id, alpha=0.5).fit(obs, acs)
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
    policies = [BayesianPolicy(env_id, alpha=0.5).fit(obs, acs)
                for obs, acs in zip(obs_seq, ac_seq)]

    features = distance_matrix(policies, metric='agreement', smoothing=False)

    nx_labels = (component_labels(features, threshold=0.3), "network")
    cl_labels = (cluster(features, k=k, metric='precomputed'), cluster.__name__)

    rnd_labels = (np.random.randint(0, k, size=len(policies)), 'random')
    uni_labels =([0] * len(policies), 'universal')

    table = BeautifulTable()
    table.column_headers = ["Method", "Variance", "Entropy"]
    for labels, method in [nx_labels, cl_labels, rnd_labels, uni_labels]:
        var, entr = [], []
        for obs, acs in zip(*aggregate_policies(obs_seq, ac_seq, labels)):
            var.append(sparse_action_clarity(obs, acs, env_id))
            entr.append(sparse_action_clarity_entropy(obs, acs, env_id))
        table.append_row([method, np.array(var), np.array(entr)])
    print(table)



if __name__ == "__main__":
    main()

