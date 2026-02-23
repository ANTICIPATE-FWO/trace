import numpy as np
from sklearn.manifold import TSNE


def tsne_transform(data, similarity: bool = False, perplexity: int = 15):
    if similarity: assert data.shape[0] == data.shape[1], f"Invalid similarity matrix shape: {data.shape}"

    tsne = TSNE(
        n_components = 2,
        perplexity = min(perplexity, (len(data) - 1) // 3),
        metric = 'precomputed' if similarity else 'euclidean',
        init = 'random' if similarity else 'pca',
        learning_rate = "auto",
        random_state = 42
    )

    return tsne.fit_transform(1.0 - data) if similarity else tsne.fit_transform(data)