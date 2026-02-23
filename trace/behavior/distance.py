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
    n = len(policies)
    D2 = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            d2 = 0.0
            for s in (set(policies[i].counts.keys()) |set(policies[j].counts.keys())):
                diff = policies[i].action_probs(s) - policies[j].action_probs(s)
                d2 += diff @ diff

            D2[i, j] = D2[j, i] = d2

    return np.sqrt(D2)


def disagreement_rate(p1, p2, min_overlap=3):
    v1, v2 = p1.get_visited(), p2.get_visited()
    overlapping_states = set(v1) & set(v2)
    if (n := len(overlapping_states)) < min_overlap: return 0.0

    disagreement = sum(1 if p1.act(s) != p2.act(s) else 0 for s in overlapping_states)
    return disagreement / n


def agreement_rate(p1, p2, min_overlap=1):
    v1, v2 = p1.get_visited(), p2.get_visited()
    overlapping_states = set(v1) & set(v2)
    if (n := len(overlapping_states)) < min_overlap: return 0.0
    # is normalizing with the number of overlapping states relevant
    agreement = sum(1 if p1.act(s) == p2.act(s) else 0 for s in overlapping_states)
    return agreement / n


def overlap(policies, lam=5.0, smoothing=True):
    n = len(policies)
    #todo make this sparse
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            d = agreement_rate(policies[i], policies[j])
            distance_matrix[i,j] = d
            distance_matrix[j,i] = d

    return np.exp(-lam * (1-distance_matrix)) if smoothing else distance_matrix