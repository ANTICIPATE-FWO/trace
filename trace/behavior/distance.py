import numpy as np

from trace.behavior.conditioning import EmpiricalDistribution

def distance_matrix(models: list[EmpiricalDistribution], metric :str='agreement', smoothing: bool=False, norm: bool=True, lam: float=0.5):
    assert metric in ('frobenius', 'wasserstein' ,'agreement', 'kl')
    dist_mat = np.ones((len(models), len(models)))

    for i in range(len(models)):
        #if i % 100 == 0: print(f'{i}/{n}')
        vi = set(models[i].get_visited())

        for j in range(i, len(models)):
            assert models[i].feature_mask == models[j].feature_mask, f'State feature mask mismatch'
            vj = set(models[j].get_visited())
            if not (overlap := vi & vj): continue

            d = 0.0
            if metric == 'frobenius':
                mat_i = [models[i].action_probs(s) for s in vi|vj]
                mat_j = [models[j].action_probs(s) for s in vi|vj]
                d = frobenius(mat_i, mat_j)
            elif metric == 'wasserstein':
                cost = l2_cost(models[i].action_values(), models[j].action_values())
                for s in overlap:
                    d += wasserstein(models[i].action_probs(s), models[j].action_probs(s), cost)
                d /= len(overlap)
            elif metric == 'agreement':
                for s in overlap:
                    d += 1 if models[i].act(s) != models[j].act(s) else 0
                d /= len(overlap)
            elif metric == 'kl':
                d = sum(kl(models[i].action_probs(s), models[j].action_probs(s)) for s in overlap)
                d /= len(overlap)

            dist_mat[i, j] = dist_mat[j, i] = d
    if norm: dist_mat = (dist_mat - dist_mat.min()) / (dist_mat.max() - dist_mat.min())
    return np.exp(-lam * (1-dist_mat)) if smoothing else dist_mat



def wasserstein(dist1:list|np.ndarray, dist2:list|np.ndarray, cost:list|np.ndarray, eps:float=0.5, max_iter:int=1000, tol:float=1e-9):
    dist1, dist2, cost = np.asarray(dist1), np.asarray(dist2), np.asarray(cost)
    kernel = np.exp(-cost / eps)
    u, v = np.ones(len(dist1)), np.ones(len(dist1))

    for _ in range(max_iter):
        u_prev = u.copy()

        u = dist1 / (kernel @ v + 1e-12)
        v = dist2 / (kernel.T @ u + 1e-12)

        if np.linalg.norm(u - u_prev, 1) < tol: break
    return np.sum(np.outer(u, v) * kernel * cost)


def l2_cost(act_coord:list|np.ndarray, act_coord2:list|np.ndarray|None=None):
    if act_coord2 is None: act_coord2 = act_coord
    act_coord, act_coord2 = np.asarray(act_coord), np.asarray(act_coord2)

    if act_coord.ndim == act_coord2.ndim == 1:
        return np.linalg.norm(act_coord - act_coord2)
    return np.array([[np.linalg.norm(i - j) for i in act_coord] for j in act_coord2])


def frobenius(mat1:list|np.ndarray, mat2:list|np.ndarray):
    mat1, mat2 = np.asarray(mat1), np.asarray(mat2)
    assert mat1.shape == mat2.shape, f'{mat1.shape} != {mat2.shape}'

    diff = mat1 - mat2
    return np.sqrt(np.sum(diff * diff))


def kl(dist1:list|np.ndarray, dist2:list|np.ndarray, eps:float=1e-12):
    dist1, dist2 = np.asarray(dist1), np.asarray(dist2)
    assert dist1.shape == dist2.shape, f'{dist1.shape} != {dist2.shape}'

    dist1 = dist1 / (np.sum(dist1) + eps)
    dist2 = dist2 / (np.sum(dist2) + eps)

    return np.sum(dist1 * np.log((dist1 + eps) / (dist2 + eps)))
