import os

from collections import Counter

os.chdir('../')

import numpy as np

from trace.behavior import EmpiricalDistribution, equal_quantization, uniform_quantization
from trace.core import TrajectoryManager

def integration():
    manager = TrajectoryManager('dst-conc').load('data/dst_ground_truth.json', split=True)
    obs, acs, _ = manager.conditioning_features(per_point=False)
    print(obs[0])
    return
    for dim in range(len(obs[0])):
        feature = [t[dim] for trajectory in obs for t in trajectory]
        d = equal_quantization(feature, bins=5)
        counter = Counter(d)
        for key, value in counter.items():
            print(key, value)
        print()



if __name__ == '__main__':
    integration()
