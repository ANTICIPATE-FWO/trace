import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn_extra.cluster import KMedoids

def random_cluster(seed:int=42):
    np.random.seed(seed)
    center = (np.random.randn(2,)*10).astype(int)
    return np.random.randn(50, 2) + center


def synthetic_dataset(clusters:int=3, seed:int=42):
    np.random.seed(seed)
    data = np.vstack([random_cluster(seed + i) for i in range(clusters)])
    np.random.shuffle(data)
    return data


def k_means(data, k: int = 3):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data)
    return kmeans.labels_, kmeans.cluster_centers_

def k_medoids(data, k: int = 3, metric: str = "euclidean"):
    kmedoids = KMedoids(n_clusters=k, metric=metric, init="k-medoids++")
    kmedoids.fit(data)
    return kmedoids.labels_, kmedoids.cluster_centers_

def gaussian_mixture(data, k_max: int = 10, covariance_type: str = "full"):
    best_gmm, best_bic = None, np.inf

    for k in range(1, k_max + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=covariance_type,
            n_init=10,
            max_iter=500,
            tol=1e-4,
            reg_covar=1e-6,
            random_state=42
        )
        gmm.fit(data)
        bic = gmm.bic(data)

        if bic < best_bic: best_bic, best_gmm = bic, gmm

    return best_gmm.predict(data), best_gmm.means_


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
