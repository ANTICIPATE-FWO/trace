import os
os.chdir('../')

import numpy as np
np.set_printoptions(suppress=True, precision=4)

from yaml import safe_load
from trace.policies import path_playout, initialize_setting
from trace.core import TrajectoryManager


def integration(path:list, steps:list):
    _, env = initialize_setting(safe_load(open('trace/configs/minetrain.yaml')), minetrain=True)
    trajectory = path_playout(env, path, steps)

    print(f'path: {path}, mine steps: {steps}')
    for o, a, r in zip(trajectory['observations'], trajectory['actions'], trajectory['rewards']):
        print('-' * 100)
        print('observation: ', o)
        print('action: ', a)
        print('reward: ', r)


def ipro_debug():
    from visuals import grid_trajectories
    from trace.policies.auxiliary import perform_action

    sample_index = 25

    manager = TrajectoryManager('minetrain').load('ipro')
    grid_trajectories(manager.subset([i==sample_index for i in range(len(manager))]), alpha=1, color='red').show()

    _, env = initialize_setting(manager.metadata, minetrain=True)
    obs, _ = env.reset()

    trajectory, terminated = {'observations': [obs.tolist()], 'actions': [], 'rewards': []}, False
    action_sample = manager.sequence('actions', flatten=True)[sample_index]
    for a, action in enumerate(action_sample):
        env, terminated = perform_action(env, action, trajectory)

    manager = TrajectoryManager(manager.metadata).load([[trajectory]])
    grid_trajectories(manager, alpha=1, color='blue').show()

    print('Playout completed\n')



if __name__ == '__main__':
    ipro_debug()




