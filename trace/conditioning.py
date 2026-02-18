import os
os.chdir('../')

import numpy as np
np.set_printoptions(suppress=True)

from trace.core import TrajectoryManager
from trace.behavior import BayesianPolicy



def main():

    env_id, filepath = "deep-sea-treasure-v0", "data/38_dst_ipro.json"
    manager = TrajectoryManager(env_id).load(filepath)

    obs_seq = manager.sequence(key='observations', pad=None, flatten=True)
    act_seq = manager.sequence(key='actions', pad=None, flatten=True)
    policy = BayesianPolicy(env_id, alpha=0.5)


    policy.fit(obs_seq, act_seq)

    #test_obs = [0.0, 0.0, 0.0, 0.7, 0.7, 0.0, 0.0]
    test_obs=[0,0]
    print("Action probabilities:", policy.action_probs(test_obs))
    print("Sampled action:", policy.act(test_obs))


if __name__ == "__main__":
    main()