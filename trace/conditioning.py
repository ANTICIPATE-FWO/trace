import os
os.chdir('../')

import numpy as np
np.set_printoptions(suppress=True)

from trace.core import TrajectoryManager
from trace.behavior import BayesianDSTPolicy



def main():

    env_id, filepath = "deep-sea-treasure-v0", "data/dst_ground_truth.json"
    manager = TrajectoryManager(env_id).load(filepath)

    obs_seq = manager.sequence(key='observations', pad=None)
    act_seq = manager.sequence(key='actions')
    policy = BayesianDSTPolicy(num_actions=4, alpha=0.5)

    policy.fit(obs_seq, act_seq)
    np.meshgrid(np.arange(12), np.arange(12))

    test_obs = [0,5]
    print("Action probabilities:", policy.action_probs(test_obs))
    print("Sampled action:", policy.act(test_obs))



if __name__ == "__main__":
    main()