import matplotlib.pyplot as plt
import numpy as np


def pareto_2d(points: np.ndarray, ground_truth: np.ndarray|None = None, title: str|None = None):
    assert points.shape[1] == 2, f'Visualization not possible for {points.shape[1]} dimensions'
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(points[:, 0], points[:, 1], s=40, label="Coverage set")

    if ground_truth is not None:
        assert ground_truth.shape[1] == 2,\
            f'Visualization not possible for {ground_truth.shape[1]} dimensions (ground truth)'
        ax.scatter(ground_truth[:, 0], ground_truth[:, 1], s=40, label="Ground truth pareto")

    ax.grid()
    ax.legend()
    if title: ax.set_title(title)
    fig.tight_layout()
    return fig


def boxplot(rewards: np.ndarray|list, title: str = "Reward Distribution by Dimension"):
    rewards = np.asarray(rewards)
    n_dims = rewards.shape[1]
    fig, ax = plt.subplots(figsize=(max(6, int(n_dims * 1.2)), 4))

    # tick_labels=[f"Dim {i}" for i in range(n_dims)]
    ax.boxplot([rewards[:, i] for i in range(n_dims)])
    ax.set(title=title, xlabel="Dimension", ylabel="Value")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    return fig