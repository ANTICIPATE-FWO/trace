import matplotlib.pyplot as plt
import numpy as np
from plotly import graph_objs as go

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


def sankey(source: np.ndarray, target: np.ndarray, value: np.ndarray, colors: dict):
    k, l = len(np.unique(source)), len(np.unique(target))
    labels = [f'Cluster {i+1} (b)' for i in range(k)] + [f'Cluster {i+1} (r)' for i in range(l)]
    node_colors = (colors['warm'][:k] + colors['cool'][:l])

    fig = go.Figure(data=[go.Sankey(
        node=dict(label=labels, pad=20, thickness=20, color=node_colors),
        link=dict(source=source, target=target, value=value)
    )])
    fig.update_layout(title_text="Cluster Flow Between Two K-Means Clusterings", font_size=12)
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