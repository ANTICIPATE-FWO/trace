import os
os.chdir('../')

import numpy as np
np.set_printoptions(suppress=True)

from trace.core import TrajectoryManager
from trace.clustering import aggregate_policies

synthetic_dst_data = [
    # first pareto point
    [{
      "observations": [[0, 0], [1, 0]],
      "actions": [1],
      "rewards": [0.7, -1]
    }],

    # second pareto point
    [{
      "observations": [[0, 0], [1, 0], [1, 1], [2, 1]],
      "actions": [1, 3, 1],
      "rewards": [8.2, -3]
    }, {
      "observations": [[0, 0], [0, 1], [1, 1], [2, 1]],
      "actions": [3, 1, 1],
      "rewards": [8.2, -3]
    }]
]
num_points = 2
num_episodes = [1, 2]

def aggregation():
    manager = TrajectoryManager(env_id='deep-sea-treasure-v0').load(synthetic_dst_data)
    ac_seq = manager.sequence(flatten=False, pad=None)
    obs_seq = manager.sequence(flatten=False, pad=None)
    c_obs, c_ac = aggregate_policies(ac_seq, obs_seq, [0] * num_points)
    assert len(c_obs) == len(c_ac) == sum(num_episodes)


def sequence():
    manager = TrajectoryManager(env_id='deep-sea-treasure-v0').load(synthetic_dst_data)
    ac_seq = manager.sequence(flatten=False, pad=None)
    assert len(ac_seq) == num_points
    for episode, num in zip(ac_seq, num_episodes):
        assert len(episode) == num
    ac_seq = manager.sequence(flatten=True, pad=None)
    assert len(ac_seq) == sum(num_episodes)
    max_len = max(len(ac) for ac in ac_seq)
    ac_seq = manager.sequence(flatten=False, pad=-1)
    assert np.array(ac_seq).shape == (num_points, len(num_episodes), max_len)
    ac_seq = manager.sequence(flatten=True, pad=-1)
    assert np.array(ac_seq).shape == (sum(num_episodes), max_len)


def distribution():
    manager = TrajectoryManager(env_id='deep-sea-treasure-v0').load(synthetic_dst_data)
    assert manager.distribution(flatten=True).shape == (3,4)
    dists = manager.distribution(flatten=False)
    assert len(dists) == num_points
    for episode, num in zip(dists, num_episodes):
        assert len(episode) == num
        for dist in episode: assert dist.shape == (4,)


def main():
    pass


if __name__ == '__main__':
    distribution()
    sequence()
    aggregation()