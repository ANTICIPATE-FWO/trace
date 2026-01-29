import os
os.chdir('../')

import numpy as np
np.set_printoptions(suppress=True)

from trace.utils import TrajectoryManager
from trace.behavior import BayesianDSTPolicy

def main():
    filepath = "data/38_dst_ipro.json"
    actions = [0, 1, 2, 3]

    manager = TrajectoryManager(actions).load(filepath)
    obs_seq = manager.sequence(key='observations')
    act_seq = manager.sequence(key='actions')

    policy = BayesianDSTPolicy(
        num_actions=len(actions),
        alpha=0.5  # weak prior
    )

    policy.fit(obs_seq, act_seq)

    test_obs = obs_seq[0, 0]

    print("Action probabilities:", policy.action_probs(test_obs))
    print("Sampled action:", policy.act(test_obs))
    # todo clustering based on conditioned policies


if __name__ == "__main__":
    main()