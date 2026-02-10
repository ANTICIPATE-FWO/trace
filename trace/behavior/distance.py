import numpy as np

def wasserstein_dist(p, q, cost, epsilon=0.5, max_iter=1000, tol=1e-9):
    # Sinkhorn-regularized Wasserstein distance.
    p, q, cost = np.array(p), np.array(q), np.array(cost)
    kernel = np.exp(-cost / epsilon)
    u, v = np.ones(len(p)), np.ones(len(p))

    for _ in range(max_iter):
        u_prev = u.copy()

        u = p / (kernel @ v + 1e-12)
        v = q / (kernel.T @ u + 1e-12)

        if np.linalg.norm(u - u_prev, 1) < tol: break
    return np.sum(np.outer(u, v) * kernel * cost)


def l2_cost(a, b=None):
    if b is None: b=a
    a, b = np.array(a), np.array(b)
    if a.ndim == b.ndim == 1: return np.linalg.norm(a-b)
    return np.array([[np.linalg.norm(i-j) for i in a] for j in b])


def frobenius(policies):
    P = np.stack([p.prob_matrix().ravel() for p in policies])
    norms = np.sum(P * P, axis=1)
    dist2 = norms[:, None] + norms[None, :] - 2.0 * P @ P.T
    dist2 = np.maximum(dist2, 0.0)
    return np.sqrt(dist2)