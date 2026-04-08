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


def aggregate_policies(obs:list|np.ndarray, acs:list|np.ndarray, labels:list):
    assert len(acs) == len(obs) == len(labels), f'Shape mismatch {len(acs)} != {len(obs)} != {len(acs[0])}'

    if isinstance(acs, np.ndarray): acs = acs.tolist()
    if isinstance(obs, np.ndarray): obs = obs.tolist()

    c_ac, c_obs = defaultdict(list), defaultdict(list)
    for ac, obs, lbl in zip(acs, obs, labels):
        c_ac[lbl].extend(ac)
        c_obs[lbl].extend(obs)
    keys = sorted(c_ac)

    return [c_obs[k] for k in keys], [c_ac[k] for k in keys]


def tree_features(obs:list|np.ndarray, acs:list|np.ndarray):
    obs = np.array([coords for episode in obs for coords in episode])
    acs = np.array([action for episode in acs for action in episode])
    return obs, acs


def discount(ar: np.ndarray, gamma: float=0.99):
    if not isinstance(ar, np.ndarray): ar = np.array(ar)
    discounts = gamma ** np.arange(ar.shape[0], dtype=np.float32)
    return np.sum(ar * discounts[:, None], axis=0)


def all_ints(lst:list):
    return all(isinstance(x, int) for x in lst)


def discretize(obs, step=0.1):
    return tuple(round(float(x) / step) * step for x in obs)
