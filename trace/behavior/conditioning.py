from collections import defaultdict
from statistics import mean
import numpy as np


class EmpiricalDistribution:
    def __init__(self, metadata: dict, alpha: float=1.0):
        self.metadata, self.alpha = metadata, alpha
        self.env_id, self.actions = metadata['env_id'], metadata['actions']
        self.counts = defaultdict(lambda: np.zeros(len(self.actions)))

    def update(self, obs, action):
        self.counts[obs][action] += 1

    def get_actions(self):
        return list(self.actions.values())

    def get_visited(self):
        return list(self.counts.keys())

    def fit(self, obs:list|np.ndarray, acs:list|np.ndarray, quant:str='uni', bins:int=10):
        assert quant in ['uni', 'eq'], f'Method {quant} is not supported'
        quantization = uniform_quantization if quant == 'uni' else equal_quantization

        obs = np.asarray(obs, dtype=float).copy()
        for dim in range(obs.shape[1]):
            obs[:, dim] = quantization(obs[:, dim], bins=bins)

        for o, a in zip(obs, acs):
            self.counts[tuple(o)][a] += 1
        return self

    def action_probs(self, obs):
        if not obs in self.counts.keys(): return np.array([1 / len(self.actions)] * len(self.actions))
        probs = (self.counts[obs] + self.alpha)
        return probs / np.sum(probs)

    def act(self, obs, deterministic:bool=True):
        probs = self.action_probs(obs)

        if deterministic: return int(probs.argmax())
        return int(np.random.choice(len(self.actions), p=probs))



def quantize(obs: list | np.ndarray, method: str, bins: int):
    assert method in ('eq', 'uni'), f'Method {method} is not supported'
    quant_obs, n_features = obs.copy(), len(obs[0][0])

    for n in range(n_features):
        values = [t[n] for trajectory in obs for t in trajectory]
        ranges, quants = uniform_quantization(min(values), max(values), bins) if method == 'uni' \
            else equal_quantization(values, bins)

        for i, trajectory in enumerate(obs):
            for j, t in enumerate(trajectory):
                quant_obs[i][j][n] = apply_mapping(t[n], ranges, quants)
    return quant_obs



def apply_mapping(value:float|int, steps:list, quants:list):
    idx = np.searchsorted(steps, value, side='left')
    idx = np.clip(idx, 0, len(quants) - 1)
    return quants[idx]


def uniform_quantization(vmin:float|int, vmax:float|int, bins: int):
    step = (vmax - vmin) / bins
    decimals = int(max(0, np.ceil(-np.log10(vmin))) + 2)
    steps = [
        round(vmin + i * step, decimals)
        for i in range(bins + 1)
    ]

    quants = [
        round((steps[i] + steps[i + 1]) / 2, decimals)
        for i in range(bins)
    ]

    return steps, quants


def equal_quantization(values:list|np.ndarray, bins:int):
    assert bins > 0 and (n:=len(values)) > 0, "Bins number and values length must be positives."
    if len(set(values)) < bins: raise ValueError(f"{len(set(values))} unique values < {bins} bins")
    values = sorted(values)

    steps, quants, start = [values[0]], [], 0
    for b in range(1, bins):
        target = round(b * n / bins)
        target = min(max(target, start + 1), n - 1)

        while target < n and values[target] == values[target - 1]:
            target += 1

        steps.append(values[target])
        quants.append(mean(values[start:target+1]))

        start = target

    steps.append(values[-1])
    quants.append(mean(values[start:]))

    return steps, quants

a= [[[0,0], [0,1], [1,1], [2,1], [2,2], [2,3]], [[0,0], [0,1]]]