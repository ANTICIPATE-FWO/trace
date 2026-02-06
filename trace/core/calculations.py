import numpy as np

def discount(ar: np.ndarray, gamma: float = 0.99):
    if not isinstance(ar, np.ndarray): ar = np.array(ar)
    discounts = gamma ** np.arange(ar.shape[0], dtype=np.float32)
    return (ar * discounts[:, None]).sum(axis=0)

def wasserstein_distance():
    pass