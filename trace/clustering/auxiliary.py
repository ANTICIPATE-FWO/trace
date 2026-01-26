import numpy as np

def random_cluster(seed:int=42):
    np.random.seed(seed)
    center = (np.random.randn(2,)*10).astype(int)
    return np.random.randn(50, 2) + center


def synthetic_dataset(clusters:int=3, seed:int=42):
    np.random.seed(seed)
    data = np.vstack([random_cluster(seed + i) for i in range(clusters)])
    np.random.shuffle(data)
    return data


def reindex_clusters(labels: np.ndarray):
    unique = np.unique(labels)
    remap = {old: new for new, old in enumerate(unique)}

    return np.vectorize(remap.get)(labels)


def cluster_connections(labels):
    k, l = len(np.unique(labels[0])), len(np.unique(labels[1]))
    source, target, value = [], [], []

    for i in range(k):
        for j in range(l):
            count = np.sum((labels[0] == i) & (labels[1] == j))
            if count > 0:
                source.append(i)
                target.append(k + j)
                value.append(count)
    return source, target, value
