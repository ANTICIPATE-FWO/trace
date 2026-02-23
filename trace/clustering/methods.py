import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn_extra.cluster import KMedoids
from trace.clustering.auxiliary import reindex_clusters


def k_means(data, k: int = 3):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data)
    return kmeans.labels_


def k_medoids(data, k: int = 3, metric: str = "euclidean"):
    kmedoids = KMedoids(n_clusters=k, metric=metric, init="k-medoids++")
    kmedoids.fit(data)
    return kmedoids.labels_


def gaussian_mixture(data, k: int = 10, covariance_type: str = "full"):
    best_gmm, best_bic = None, np.inf

    for k in range(1, k + 1):
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

    return best_gmm.predict(data)


def dirichlet_mixture(
    data,
    k: int = 20,
    covariance_type: str = "full",
    weight_concentration_prior: float = 1.0
):
    dpgmm = BayesianGaussianMixture(
        n_components=k,
        covariance_type=covariance_type,
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
    #weights = dpgmm.weights_[active]

    return reindex_clusters(dpgmm.predict(data))


def spectral(data, k=3):
    sc = SpectralClustering(
        n_clusters=k,
        affinity='precomputed',
        assign_labels='kmeans'
    )
    return sc.fit_predict(data)