import numpy as np
import matplotlib.pyplot as plt

from trace.core import env_metadata
dst_actions = env_metadata['deep-sea-treasure-v0']['actions']


def grid_arrows(policy, title: str = "Conditioned Policy (Most Probable Action)", color: str = "black"):
    if policy.env_id != 'deep-sea-treasure-v0': return None
    X, Y = policy.obs_space

    u = np.full((len(Y), len(X)), np.nan, dtype=float)
    v = np.full((len(Y), len(X)), np.nan, dtype=float)


    x0, y0 = X[0], Y[0]
    for state in policy.counts.keys():
        px = policy.action_probs(state)
        if px.max() - px.min() < 1e-6: continue

        y, x = state
        iy, ix = int(y - y0), int(x - x0)
        u[iy, ix], v[iy, ix] = dst_actions[np.argmax(px)]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.quiver(X, Y, u, v, angles="xy", scale_units="xy", scale=1, width=0.015, color=color)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.grid(True)
    ax.set_title(title)
    ax.set_xlabel("y")
    ax.set_ylabel("x")
    plt.tight_layout()

    return fig



def dst_frame():
    import warnings
    import mo_gymnasium as mo_gym

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        render_env = mo_gym.make("deep-sea-treasure-v0", render_mode="rgb_array")
        render_env.reset()
        frame = render_env.render()

    return frame




def grid_map(observations, space=(11,11), title="Trajectory Density over Grid", alpha=0.05, linewidth=1.0, color='red'):
    frame = dst_frame()
    grid_h, grid_w = space
    fig, ax = plt.subplots(figsize=(6, 6))

    # map pixels → grid coords
    ax.imshow(frame, extent=[0, grid_w, grid_h, 0])

    for traj in observations:
        if len(traj) < 2: continue

        xs = [p[1] + 0.5 for p in traj]
        ys = [p[0] + 0.5 for p in traj]

        ax.plot(xs, ys, alpha=alpha, linewidth=linewidth, color=color)

    ax.set_aspect('equal')
    ax.set_title(title)

    ax.axis("off")
    plt.tight_layout()
    return fig

