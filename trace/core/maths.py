import numpy as np


def dist(a:np.ndarray, b:np.ndarray):
    return np.linalg.norm(np.array(a) - np.array(b))


def seg_angle(a, b):
    a , b = np.array(a), np.array(b)
    assert a.shape == b.shape, f'Shape assertation: {a.shape}, {b.shape}'
    return 180 * np.arctan2(*(b - a)) / np.pi


def seg_proj(point, a, b, eps:float=1e-8):
    point, a, b = np.array(point), np.array(a), np.array(b)
    ab, ap = b - a, point - a

    if (ab_norm := np.linalg.norm(ab)) == 0: return a.copy()

    t = np.dot(ap, ab) / (ab_norm ** 2 + eps)
    t = max(0.0, min(1.0, t))

    return a + t * ab


def trail_comb(nodes:list|np.ndarray, start:tuple|list|np.ndarray):
    trails = [(np.array(start), m) for m in nodes]

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            trails.append((nodes[i], nodes[j]))

    return trails


def same_point(p:np.ndarray, q:np.ndarray, tolerance:float=1e-3):
    return np.linalg.norm(p - q) < tolerance


def is_connected(current_trail:tuple, new_trail:tuple):
    if same_point(current_trail[1], new_trail[0]):
        return True
    return False


def discount(ar: np.ndarray, gamma: float=0.99):
    if not isinstance(ar, np.ndarray): ar = np.array(ar)
    discounts = gamma ** np.arange(ar.shape[0], dtype=np.float32)
    return np.sum(ar * discounts[:, None], axis=0)


def all_ints(lst:list):
    return all(isinstance(x, int) for x in lst)


def discretize(obs:list|np.ndarray, step:float=0.1):
    return tuple(round(float(x) / step) * step for x in obs)
