import numpy as np
from sklearn.manifold import TSNE


def tsne_transform(data, centers, perplexity: int = 30):
    # Fit t-SNE jointly on data + centers so they share the same embedding space
    combined = np.vstack([data, centers])
    perplexity = min(perplexity, len(data) - 1)

    tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=42)

    embedded = tsne.fit_transform(combined)
    data_2d = embedded[: len(data)]
    centers_2d = embedded[len(data) :]

    return data_2d, centers_2d
