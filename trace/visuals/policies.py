import numpy as np
import matplotlib.pyplot as plt

from trace.core import env_metadata
dst_actions = env_metadata['deep-sea-treasure-v0']['actions']


def grid_arrows(policy, title: str = "Conditioned Policy (Most Probable Action)", color:str = "black"):
    X, Y = policy.obs_space
    u, v = np.full_like(X, np.nan), np.full_like(Y, np.nan)

    for y in Y:
        for x in X:
            probs = policy.prob_matrix()[y, x]
            if np.max(probs) - np.min(probs) < 1e-6: continue
            u[y, x], v[y, x] = dst_actions[np.argmax(probs)]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.quiver(*policy.obs_space, u, v, angles="xy",
              scale_units="xy",scale=1, width=0.015, color=color,)

    ax.invert_yaxis()
    ax.grid(True)
    ax.set_title(title)
    ax.set_xlabel("y")
    ax.set_ylabel("x")
    return fig

