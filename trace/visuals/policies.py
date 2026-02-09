import numpy as np
import matplotlib.pyplot as plt

from trace.core import env_metadata
ACTIONS = env_metadata['deep-sea-treasure-v0']['actions']


def grid_arrows(policy, title: str = "Conditioned Policy (Most Probable Action)"):
    #todo the for loop can become cleaner, too many declarations
    width, height = policy.obs_shape()
    x, y = policy.obs_space
    prob_matrix = policy.prob_matrix()
    u = np.full_like(x, np.nan, dtype=float)
    v = np.full_like(y, np.nan, dtype=float)

    for y in range(height):
        for x in range(width):
            probs = prob_matrix[y, x]
            if np.max(probs) - np.min(probs) < 1e-6: continue
            best_action = np.argmax(probs)

            dx, dy = ACTIONS[best_action]
            u[y, x] = dx
            v[y, x] = dy

    plt.figure(figsize=(6, 6))
    plt.quiver(
        *policy.obs_space, u, v,
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.015
    )

    plt.gca().invert_yaxis()
    plt.xticks(range(width))
    plt.yticks(range(height))
    plt.grid(True)
    plt.title(title)
    plt.xlabel("y")
    plt.ylabel("x")

    plt.show()
