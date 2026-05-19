import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn_extra.cluster import KMedoids


def k_means(data, k: int = 3):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data)
    return kmeans.labels_


def k_medoids(data, k: int = 3, metric: str = "cosine"):
    kmedoids = KMedoids(n_clusters=k, metric=metric, init="k-medoids++")
    kmedoids.fit(data)
    return kmedoids.labels_


def gaussian_mixture(data, k: int = 10, covariance: str = "full"):
    best_gmm, best_bic = None, np.inf

    for k in range(1, k + 1):
        gmm = GaussianMixture(n_components=k, covariance_type=covariance, n_init=10, max_iter=500, tol=1e-4,
            reg_covar=1e-6, random_state=42)
        gmm.fit(data)
        bic = gmm.bic(data)

        if bic < best_bic: best_bic, best_gmm = bic, gmm

    return best_gmm.predict(data)


def dirichlet_mixture(data, k: int = 20, covariance: str = "full", weight_concentration_prior: float = 1.0):
    dpgmm = BayesianGaussianMixture(
        n_components=k,
        covariance_type=covariance,
        weight_concentration_prior_type="dirichlet_process",
        weight_concentration_prior=weight_concentration_prior,
        n_init=5,
        max_iter=1000,
        tol=1e-4,
        reg_covar=1e-6,
        random_state=42,
        init_params="kmeans"
    )

    dpgmm.fit(data)
    active = dpgmm.weights_ > 1e-3

    return reindex_clusters(dpgmm.predict(data))


def spectral(data, k=3):
    sc = SpectralClustering(n_clusters=k, affinity='precomputed', assign_labels='kmeans')
    return sc.fit_predict(data)


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
