from typing import Any, List
import numpy as np

NUM_ACTIONS = 4  # actions: 0,1,2,3 for dst setting

def episode_dist(episode: List[Any], normalize: bool = True):
    if len(episode) < 1: raise ValueError('Episode must have at least one action')

    counts = np.zeros(NUM_ACTIONS, dtype=np.float32)
    for a in episode: counts[a] += 1
    if normalize: counts /= counts.sum()
    return counts


def policy_dist(trajectories: List[Any], normalize: bool = True):
    return np.array([
        episode_dist(episode['actions'], normalize=normalize)
        for point in trajectories for episode in point
    ])
