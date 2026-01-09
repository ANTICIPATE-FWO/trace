from json import load
from collections import Counter
import numpy as np

def scalarized_reward(trajectory, gamma=0.99):
    returns = []

    for traj in trajectory:
        rewards = np.asarray(traj["rewards"], dtype=np.float32)
        timesteps = rewards.shape[0]

        discounts = gamma ** np.arange(timesteps, dtype=np.float32)
        discounted_return = (rewards * discounts[:, None]).sum(axis=0)

        returns.append(discounted_return)

    return np.mean(returns, axis=0)




filepath = "/Users/emilypalaska/Documents/morl-baselines/policies/data.json"

with open(filepath, "r") as f:
    trajectories = load(f)

point1, point2 = trajectories[0], trajectories[1]

print(f'Pareto point 1: {scalarized_reward(point1)}\n---------------')

ep_ids = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]
for ep_id in ep_ids:
    print(f'\nEpisode {ep_id}')
    counter = Counter(point1[ep_id]['actions'])
    for k, v in counter.items(): print(f'\t{k}: {v}')

print(f'\nPareto point 2 {scalarized_reward(point2)}\n---------------')
ep_ids = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]
for ep_id in ep_ids:
    print(f'\nEpisode {ep_id}')
    counter = Counter(point2[ep_id]['actions'])
    for k, v in counter.items(): print(f'\t{k}: {v}')

