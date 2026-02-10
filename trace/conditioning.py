import os
os.chdir('../')

import numpy as np
np.set_printoptions(suppress=True)

from trace.core import TrajectoryManager
from trace.behavior import BayesianPolicy



def main():

    env_id, filepath = "deep-sea-treasure-v0", "data/dst_ground_truth.json"
    manager = TrajectoryManager(env_id).load(filepath)

    obs_seq = manager.sequence(key='observations', pad=None)
    act_seq = manager.sequence(key='actions', pad=None)
    policy = BayesianPolicy(env_id, alpha=0.5)

    policy.fit(obs_seq, act_seq)

    test_obs = [2,3]
    probs_matrix = policy.prob_matrix()
    print("Action probabilities:", policy.action_probs(test_obs), probs_matrix[*test_obs])
    print("Sampled action:", policy.act(test_obs))



if __name__ == "__main__":
    main()