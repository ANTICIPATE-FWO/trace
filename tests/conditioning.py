import os
os.chdir('../')

import numpy as np

from trace.behavior import EmpiricalDistribution, quantize, distance_matrix
from trace.core import TrajectoryManager
from trace.visuals import heatmap, overlap_bar, distribution_bar

def overlap_actions(models:list[EmpiricalDistribution]):
    common_state_actions = []

    for i in range(len(models)):
        vi = set(models[i].get_visited())

        for j in range(i, len(models)):
            assert models[i].feature_mask == models[j].feature_mask, \
                f'State feature mask mismatch'
            vj = set(models[j].get_visited())

            common_state_actions.append(np.sum(
                [models[i].counts[v] + models[j].counts[v] for v in vi&vj]
            ))

    return common_state_actions



def quantization_comparison(manager:TrajectoryManager):
    og_obs, acs, _ = manager.conditioning_features(per_point=False)

    print('Unique values per feature')
    for n in range(len(og_obs[0][0])):
        values = [coords[n] for trajectory in og_obs for coords in trajectory]
        print(f'\t{n}: {len(set(values))}')
    print()

    bins = [10, 10, 6, 10, 10, 90, 90]
    all_obs = [og_obs, quantize(og_obs, method='uni', bins=bins), quantize(og_obs, method='eq', bins=bins)]
    titles = [
        'without quantization',
        'uniform quantization',
        'equal quantization',
    ]

    for obs, title in zip(all_obs, titles):
        print(title)
        models = [EmpiricalDistribution(manager.metadata).fit(o, a) for o, a in zip(obs, acs)]
        features = distance_matrix(models, metric='kl')
        heatmap(features, title=title).show()
        distribution_bar(features.flatten(), title='Distances '+title).show()
        distribution_bar(overlap_actions(models), title='Actions overlap '+title).show()
        print()


if __name__ == '__main__':
    quantization_comparison(TrajectoryManager('minetrain').load('ground_truth'))
