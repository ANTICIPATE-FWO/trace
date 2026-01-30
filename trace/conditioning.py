import os
os.chdir('../')

import numpy as np
np.set_printoptions(suppress=True)

from trace.core import TrajectoryManager
from trace.behavior import BayesianDSTPolicy

def main():
    filepath = "data/38_dst_ipro.json"
    env_id = "deep-sea-treasure-v0"

    manager = TrajectoryManager(env_id).load(filepath)
    obs_seq = manager.sequence(key='observations')
    act_seq = manager.sequence(key='actions')

    policy = BayesianDSTPolicy(
        num_actions=4,
        alpha=0.5  # weak prior
    )

    policy.fit(obs_seq, act_seq)

    test_obs = obs_seq[1, 1]

    print("Action probabilities:", policy.action_probs(test_obs))
    print("Sampled action:", policy.act(test_obs))
    # todo clustering based on conditioned policies


if __name__ == "__main__":
    main()