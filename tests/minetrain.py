import os
os.chdir('../')

import numpy as np
np.set_printoptions(suppress=True, precision=4)

from yaml import safe_load
from trace.policies import path_playout, initialize_setting


def integration(path:list, steps:list):
    _, env = initialize_setting(safe_load(open('trace/configs/minetrain.yaml')))
    trajectory = path_playout(env, path, steps)

    print(f'path: {path}, mine steps: {steps}')
    for o, a, r in zip(trajectory['observations'], trajectory['actions'], trajectory['rewards']):
        print('-' * 100)
        print('observation: ', o)
        print('action: ', a)
        print('reward: ', r)
