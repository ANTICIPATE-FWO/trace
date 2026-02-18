import numpy as np
from sklearn.manifold import TSNE


def tsne_transform(data, perplexity: int = 30):
    perplexity = min(perplexity, len(data) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=42)
    return tsne.fit_transform(data)
