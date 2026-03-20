import numpy as np
from sklearn.manifold import TSNE


def tsne_transform(data, precomputed: bool = False, perplexity: int = 30):
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


def unique_rewards(rewards, trajectories):
    msg = ''
    msg += "\nExpected return\t\t\t  |\tUnique trajectories\n"
    msg += "-" * 60 + '\n'

    for r_sc, f_traj in zip(rewards, trajectories):
        r_str = np.array2string(r_sc, formatter={'float_kind': lambda x: f"{x:+.3f}"})
        msg += f"{r_str:<25} | {len(f_traj):>3}\n"
    return msg


def dst_frame():
    import warnings
    import mo_gymnasium as mo_gym

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        render_env = mo_gym.make("deep-sea-treasure-v0", render_mode="rgb_array")
        render_env.reset()
        frame = render_env.render()

    return frame
