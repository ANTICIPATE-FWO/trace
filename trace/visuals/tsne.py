import numpy as np
from sklearn.manifold import TSNE


def tsne_transform(data, precomputed: bool = False, perplexity: int = 10):
    if precomputed: assert data.shape[0] == data.shape[1], f"Invalid similarity matrix shape: {data.shape}"

    tsne = TSNE(
        n_components = 2,
        perplexity = min(perplexity, len(data)),
        metric = 'precomputed' if precomputed else 'euclidean',
        init = 'random' if precomputed else 'pca',
        learning_rate = "auto",
        random_state = 42
    )

    return tsne.fit_transform(data)