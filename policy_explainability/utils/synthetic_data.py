import numpy as np


def cluster(seed:int=42):
    np.random.seed(seed)
    center = (np.random.randn(2,)*10).astype(int)
    return np.random.randn(50, 2) + center

def generate_cluster_data(clusters:int=3, seed:int=42):
    np.random.seed(seed)
    data = np.vstack([cluster(seed + i) for i in range(clusters)])
    np.random.shuffle(data)
    return data




