import os
os.chdir('..')

import numpy as np
np.set_printoptions(suppress=True)

from yaml import safe_load
from trace.core import TrajectoryManager

env_id = 'minecart-v0'
traj_filepaths = ['data/3.2_minecart_ipro.json', 'data/3_minecart_ipro.json', 'data/2m_minecart_ipro.json']
metadata_filepath = "trace/configs/environments.yaml"

def pareto_filter(points: np.ndarray):
    is_dominated = np.zeros(len(points), dtype=bool)

    for i in range(len(points)):
        if is_dominated[i]: continue
        for j in range(len(points)):
            if i == j: continue
            # j dominates i
            if np.all(points[j] >= points[i]) and np.any(points[j] > points[i]):
                is_dominated[i] = True
                break

    return points[~is_dominated]


def main():
    metadata = safe_load(open(metadata_filepath, "r"))[env_id]
    managers = [TrajectoryManager(metadata).load(file) for file in traj_filepaths]

    rewards = np.vstack([manager.accrue() for manager in managers])
    for file, manager in zip(traj_filepaths, managers): print(f'{file}: {manager.accrue().shape[0]} points')

    filtered = pareto_filter(rewards)
    print(f'Filtering: {rewards.shape[0]} -> {filtered.shape[0]} points')

    for manager in managers:
        print(manager.accrue())
        print()

    print(filtered)


if __name__ == '__main__':
    main()