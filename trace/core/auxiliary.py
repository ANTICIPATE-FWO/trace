from collections import defaultdict
import numpy as np

def homogenize(array:list, pad:int=-1):
    def homogenous_shape(x):
        if not isinstance(x, list): return ()
        sub_array = [homogenous_shape(i) for i in x]
        return (len(x),) + tuple(
            max((s[d] if d < len(s) else 0) for s in sub_array)
            for d in range(max(map(len, sub_array), default=0))
        )

    def fill(x, shp:tuple):
        if not shp: return x
        x = x if isinstance(x, list) else []
        return [
            fill(x[i] if i < len(x) else pad, shp[1:])
            for i in range(shp[0])
        ]

    return fill(array, homogenous_shape(array))


def aggregate_policies(obs:list|np.ndarray, acs:list|np.ndarray, rews:list|np.ndarray, labels:list):
    assert len(acs) == len(obs) == len(rews) == len(labels), f'Shape mismatch {len(acs)} != {len(obs)} != {len(acs[0])}'

    if isinstance(acs, np.ndarray): acs = acs.tolist()
    if isinstance(obs, np.ndarray): obs = obs.tolist()
    if isinstance(rews, np.ndarray): rews = rews.tolist()

    c_ac, c_obs, c_rews = defaultdict(list), defaultdict(list), defaultdict(list)
    for a, o, r, lbl in zip(acs, obs, rews, labels):
        c_ac[lbl].extend(a)
        c_obs[lbl].extend(o)
        c_rews[lbl].extend(r)
    keys = sorted(c_ac)

    return [c_obs[k] for k in keys], [c_ac[k] for k in keys], [c_rews[k] for k in keys]


def tree_features(obs:list|np.ndarray, acs:list|np.ndarray):
    assert len(acs) == len(obs), f'Length mismatch {len(acs)} != {len(obs)}'

    if len(obs[0]) == len(acs[0]) + 1: obs = [episode[:-1] for episode in obs]
    obs = np.array([coords for episode in obs for coords in episode])
    acs = np.array([action for episode in acs for action in episode])

    return obs, acs


