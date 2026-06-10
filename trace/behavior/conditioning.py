from typing import Hashable
from collections import Counter, defaultdict
import numpy as np
import copy

from trace.core import all_ints

class EmpiricalDistribution:
    def __init__(self, metadata: dict, alpha: float=1.0, feature_mask:list[bool]|None=None):
        self.metadata, self.alpha = metadata, alpha
        self.env_id = metadata['env_id']
        self.num_actions = len(metadata['actions'])
        self.counts = defaultdict(lambda: np.zeros(self.num_actions))
        if feature_mask is not None:
            assert len(feature_mask) == len(metadata['observations_high']), 'Feature mask shape mismatch'
        self.feature_mask = feature_mask

    def action_values(self):
        return list(self.metadata['actions'].values())

    def get_visited(self):
        return list(self.counts.keys())

    def fit(self, obs:list|np.ndarray, acs:list|np.ndarray):
        if not isinstance(obs[0][0], Hashable): obs = [coords for traj in obs for coords in traj]
        for coords, a in zip(obs, acs):
            if self.feature_mask is not None:
                coords = [feature for feature, mask in zip(coords, self.feature_mask) if mask]
            self.counts[tuple(coords)][a] += 1
        return self

    def action_probs(self, coords:list|np.ndarray):
        coords = tuple(coords) if self.feature_mask is None \
            else (feature for feature, mask in zip(coords, self.feature_mask) if mask)

        if not coords in self.counts.keys():
            return np.array([1 / self.num_actions] * self.num_actions)
        probs = (self.counts[coords] + self.alpha)
        return probs / np.sum(probs)

    def act(self, coords, deterministic:bool=True):
        probs = self.action_probs(coords)
        if deterministic: return int(probs.argmax())
        return int(np.random.choice(self.num_actions, p=probs))



def quantize(obs: list|np.ndarray, method:str, bins:int|list, return_ranges:bool=False):
    assert method in ('eq', 'uni'), f'Method {method} is not supported'
    quant_obs, n_features = copy.deepcopy(obs), len(obs[0][0])
    if isinstance(bins, int): bins = [bins] * n_features
    assert all_ints(bins), f'Bin number should be integer.'

    feature_ranges = []
    for n in range(n_features):
        values = [t[n] for trajectory in obs for t in trajectory]
        ranges, quants = uniform_quantization(min(values), max(values), bins[n]) if method == 'uni' \
            else equal_quantization(values, bins[n])
        feature_ranges.append(ranges)
        for i, trajectory in enumerate(obs):
            for j, t in enumerate(trajectory):
                quant_obs[i][j][n] = apply_mapping(t[n], ranges, quants)

    if return_ranges: return quant_obs, feature_ranges
    return quant_obs


def apply_mapping(value:float|int, ranges:list, quants:list):
    idx = np.searchsorted(ranges, value, side='right') - 1
    idx = np.clip(idx, 0, len(quants) - 1)
    return quants[idx]


def uniform_quantization(vmin:float|int, vmax:float|int, bins: int):
    assert vmin != vmax, f'Redundant value set, {vmin} = {vmax}'
    step = (vmax - vmin) / bins
    nonzero = vmin if vmin != 0 else vmax
    decimals = len(str(nonzero).split('.')[1]) if '.' in str(nonzero) else 0
    steps = [round(vmin + b * step, decimals) for b in range(bins + 1)]
    quants = [round((steps[b] + steps[b + 1]) / 2, decimals) for b in range(bins)]

    return steps, quants



def equal_quantization(values: list | np.ndarray, bins: int):
    assert bins > 0 and (n := len(values)) > 0, "Bins and values do not contain anything"
    counts = Counter(values)

    if len((uniques := sorted(counts))) < bins:
        raise ValueError(f"{len(uniques)} unique values < {bins} bins")

    targets = [round(b * n / bins) for b in range(1, bins)]
    ranges, quants = [uniques[0]], []
    start, cumulative, target_idx, last_boundary = 0, 0, 0, -1

    for last_idx, value in enumerate(uniques):
        cumulative += counts[value]

        if target_idx < len(targets) and cumulative >= targets[target_idx] and last_idx > last_boundary:
            ranges.append(value)

            vals = uniques[start:last_idx + 1]
            quants.append(sum(v * counts[v] for v in vals) / sum(counts[v] for v in vals))

            start = last_idx + 1
            last_boundary = last_idx

            while target_idx < len(targets) and cumulative >= targets[target_idx]:
                target_idx += 1

    ranges.append(uniques[-1])

    if start < len(uniques):
        vals = uniques[start:]
        quants.append(sum(v * counts[v] for v in vals) / sum(counts[v] for v in vals))


    return ranges, quants