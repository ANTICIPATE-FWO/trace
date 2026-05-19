import numpy as np

from trace.behavior.conditioning import EmpiricalDistribution

def distance_matrix(models: list[EmpiricalDistribution], metric :str='agreement', smoothing: bool=False, norm: bool=True, lam: float=0.5):
    assert metric in ('frobenius', 'wasserstein' ,'agreement')

    n = len(models)
    dist_mat = np.ones((n, n))

    for i in range(n):
        if i % 100 == 0: print(f'{i}/{n}')
        vi = set(models[i].get_visited())

        for j in range(i, n):
            vj = set(models[j].get_visited())
            if not (overlap := vi & vj): continue

            d = 0.0
            if metric == 'frobenius':
                mat_i = [models[i].action_probs(s) for s in vi|vj]
                mat_j = [models[j].action_probs(s) for s in vi|vj]
                d = frobenius(mat_i, mat_j)
            elif metric == 'wasserstein':
                cost = l2_cost(models[i].get_actions(), models[j].get_actions())
                for s in overlap:
                    d += wasserstein(models[i].action_probs(s), models[j].action_probs(s), cost)
                d /= len(overlap)
            elif metric == 'agreement':
                for s in overlap:
                    d += 1 if models[i].act(s) != models[j].act(s) else 0
                d /= len(overlap)

            dist_mat[i, j] = dist_mat[j, i] = d
    if norm: dist_mat = (dist_mat - dist_mat.min()) / (dist_mat.max() - dist_mat.min())
    return np.exp(-lam * (1-dist_mat)) if smoothing else dist_mat



def wasserstein(dist1, dist2, cost, epsilon=0.5, max_iter=1000, tol=1e-9):
    dist1, dist2, cost = np.array(dist1), np.array(dist2), np.array(cost)
    kernel = np.exp(-cost / epsilon)
    u, v = np.ones(len(dist1)), np.ones(len(dist1))

    for _ in range(max_iter):
        u_prev = u.copy()

        u = dist1 / (kernel @ v + 1e-12)
        v = dist2 / (kernel.T @ u + 1e-12)

        if np.linalg.norm(u - u_prev, 1) < tol: break
    return np.sum(np.outer(u, v) * kernel * cost)


def l2_cost(act_coord, act_coord2=None):
    if act_coord2 is None: act_coord2 = act_coord
    act_coord, act_coord2 = np.array(act_coord), np.array(act_coord2)

    if act_coord.ndim == act_coord2.ndim == 1:
        return np.linalg.norm(act_coord - act_coord2)
    return np.array([[np.linalg.norm(i - j) for i in act_coord] for j in act_coord2])


def frobenius(mat1, mat2):
    mat1, mat2 = np.asarray(mat1), np.asarray(mat2)
    assert mat1.shape == mat2.shape, f'{mat1.shape} != {mat2.shape}'

    diff = mat1 - mat2
    return np.sqrt(np.sum(diff * diff))