import numpy as np
from matplotlib import pyplot as plt

from trace.core import discretize
from trace.visuals.utils import env_frame, tsne_transform, tree_to_graph

import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
from sklearn.tree import DecisionTreeClassifier



def temporal_alignment(sequences: list|np.ndarray, actions : dict, time_range: tuple|None=None, title: str|None=None):
    unique_values = sorted(set(val for seq in sequences for val in seq))
    value_to_int = {val: i for i, val in enumerate(unique_values)}

    if time_range:
        start, end = time_range
        max_len = end - start
    else:
        max_len = max(len(seq) for seq in sequences)
        start, end = 0, max_len - 1


    matrix = np.full((len(sequences), max_len), fill_value=-1)

    for i, seq in enumerate(sequences):
        for j in range(start, min(end, len(seq))):
            matrix[i, j - start] = value_to_int[seq[j]]


    # Colormap with black for padding (-1)
    cmap = plt.cm.get_cmap('viridis', len(unique_values))
    cmap.set_bad(color='black')

    masked_matrix = np.ma.masked_where(matrix == -1, matrix)

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(masked_matrix, aspect='auto', cmap=cmap)

    tick_positions = list(range(len(unique_values)))
    tick_labels = [actions[value] for value in unique_values]

    cbar = fig.colorbar(im, ax=ax, ticks=tick_positions)
    cbar.ax.set_yticklabels(tick_labels)

    if len(sequences) < 10:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i, j] != -1:
                    original_j = j + start if time_range is not None else j
                    ax.text(
                        j, i, sequences[i][original_j],
                        ha='center', va='center', color='black'
                    )

    if title: ax.set_title(title)
    xticks = np.linspace(start, end - 1, 5, dtype=int)
    ax.set(xlabel='Time steps', yticks=[], xticks=xticks, xticklabels=xticks)
    fig.tight_layout()

    return fig


def grid_arrows(policy, action_mapping: dict, title: str = None, color: str = "black"):
    assert policy.env_id == 'deep-sea-treasure-v0', f'Cannot visualize grid for environment {policy.env_id}'
    grid_x, grid_y = policy.obs_space

    u = np.full((len(grid_y), len(grid_x)), np.nan, dtype=float)
    v = np.full((len(grid_y), len(grid_x)), np.nan, dtype=float)

    x0, y0 = grid_x[0], grid_y[0]
    for state in policy.counts.keys():
        px = policy.action_probs(state)
        if px.max() - px.min() < 1e-6: continue

        y, x = state
        iy, ix = int(y - y0), int(x - x0)
        u[iy, ix], v[iy, ix] = action_mapping[np.argmax(px)]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.quiver(grid_x, grid_y, u, v, angles="xy", scale_units="xy", scale=1, width=0.015, color=color)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.grid()
    if title is not None: ax.set(title=title, xlabel='y', ylabel='x')
    fig.tight_layout()

    return fig


def grid_trajectories(obs: list|np.ndarray, acs: list|np.ndarray, space: tuple=(11,11), title: str|None=None, alpha: float=0.1, color: str='red'):
    # todo include minecart with extra argument
    frame = env_frame("deep-sea-treasure-v0")
    grid_h, grid_w = space
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.imshow(frame, extent=(0, grid_w, grid_h, 0))
    for episode in obs:
        if len(episode) < 2: continue

        xs = [coords[1] + 0.5 for coords in episode]
        ys = [coords[0] + 0.5 for coords in episode]

        ax.plot(xs, ys, alpha=alpha, linewidth=1.5, color=color)

    ax.set_aspect('equal')
    ax.axis("off")
    if title: ax.set_title(title)

    fig.tight_layout()
    return fig


def minecart_trajectories(observations: list|np.ndarray, actions: list|np.ndarray, space:tuple=(100, 100), mines:list|None=None,
                          title:str|None=None, alpha=0.05, color='red', mine_color='blue'):
    frame = env_frame("minecart-v0")
    grid_h, grid_w = space
    step = 1/ grid_h
    fig, ax = plt.subplots(figsize=(6, 6))

    # map pixels → grid coords
    ax.imshow(frame, extent=(0, grid_w, grid_h, 0))

    for obs, act in zip(observations, actions):
        if len(obs) < 2: continue

        xs, ys = [], []
        for coords in obs:
            y, x = discretize(coords[:2], step)
            xs.append(x * grid_w)
            ys.append(y * grid_h)

        ax.plot(xs, ys, alpha=alpha, linewidth=1.5, color=color)

        for p, a in zip(obs, act):
            if a == 0:
                y, x = discretize(p[:2], step)
                ax.scatter(x * grid_w, y * grid_h, marker='x', color=mine_color, s=30)

    if mines is not None:
        for m in mines:
            y, x = discretize(m, step)
            ax.scatter(x * grid_w, y * grid_h, marker='o', color='yellow', s=30)

    ax.set_aspect('equal')
    ax.axis("off")
    if title: ax.set_title(title)
    fig.tight_layout()

    return fig


def cluster_scatter(data, labels, colors : list, title: str|None=None, precomputed: bool=True):
    if data.shape[1] > 2: data = tsne_transform(data, precomputed=precomputed)
    fig, ax = plt.subplots(figsize=(8, 6))

    for l in set(labels):
        x, y = data[labels == l, 0], data[labels == l, 1]
        ax.scatter(x, y, label=f"Cluster {l + 1}", color=colors[l % len(colors)], linewidth=8)

    ax.legend()
    ax.grid()
    ax.tick_params(labelbottom=False, labelleft=False)
    if title: ax.set_title(title)
    fig.tight_layout()
    return fig


def decision_tree(obs: list, acs: list, metadata: dict, max_depth: int=8, title: str|None=None):
    assert len(obs) == len(acs), f'Length of observations {len(obs)} does not match length of actions {len(acs)}.'
    observations_features, action_names = metadata['observations_features'], metadata['actions']

    clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=20, ccp_alpha=0.01)
    clf.fit(obs, acs)

    graph, node_labels, edge_labels, _ = tree_to_graph(clf.tree_, clf.classes_, observations_features, action_names)
    pos = graphviz_layout(graph, prog="dot")

    fig, ax = plt.subplots()
    nx.draw(graph, pos, ax=ax, with_labels=False, node_color='white',)
    nx.draw_networkx_labels(graph, pos, labels=node_labels, ax=ax)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, ax=ax)

    ax.axis("off")
    if title is not None: ax.set_title(title)
    fig.tight_layout()

    return fig