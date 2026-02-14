import numpy as np
import matplotlib.pyplot as plt

from trace.core import env_metadata
dst_actions = env_metadata['deep-sea-treasure-v0']['actions']


def grid_arrows(policy, title: str = "Conditioned Policy (Most Probable Action)", color:str = "black"):
    X, Y = policy.obs_space
    probs = policy.prob_matrix()
    u, v = np.full_like(X, np.nan, dtype=float), np.full_like(Y, np.nan, dtype=float)

    for x, y in zip(X.ravel(), Y.ravel()):
        if np.max(probs[y, x]) - np.min(probs[y, x]) < 1e-6: continue
        u[y, x], v[y, x] = dst_actions[np.argmax(probs[y, x])]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.quiver(*policy.obs_space, u, v, angles="xy",
              scale_units="xy",scale=1, width=0.015, color=color,)

    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.grid(True)
    ax.set_title(title)
    ax.set_xlabel("y")
    ax.set_ylabel("x")
    plt.tight_layout()
    return fig

def grid_trajectories(observations, space, alpha=0.05, linewidth=1.0):
    width, height = space

    fig, ax = plt.subplots(figsize=(6, 6))

    # draw grid bounds
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)

    # plot each trajectory
    for traj in observations:
        if len(traj) < 2: continue

        xs = [p[0] for p in traj]
        ys = [p[1] for p in traj]

        ax.plot(xs, ys, alpha=alpha, linewidth=linewidth)

    ax.set_aspect('equal')
    ax.set_title("Trajectory Density over Grid")
    ax.invert_yaxis()

    plt.tight_layout()
    ax.grid(True)
    plt.show()

