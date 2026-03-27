import numpy as np
from matplotlib import pyplot as plt

from trace.core import env_metadata, colors
from trace.visuals.utils import dst_frame, tsne_transform, tree_to_graph

import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
from sklearn.tree import DecisionTreeClassifier


def temporal_alignment(sequences, env_id:str, title: str = "Temporal Alignment"):
    unique_values = sorted(set(val for seq in sequences for val in seq))
    value_to_int = {val: i for i, val in enumerate(unique_values)}

    max_len = max(len(seq) for seq in sequences)
    matrix = np.full((len(sequences), max_len), fill_value=-1)

    for i, seq in enumerate(sequences):
        for j, val in enumerate(seq):
            matrix[i, j] = value_to_int[val]

    cmap = plt.cm.get_cmap('Reds', len(unique_values))

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(matrix, aspect='auto', cmap=cmap)

    tick_positions = list(range(len(unique_values)))
    tick_labels = [env_metadata[env_id]['actions'][value] for value in unique_values]

    cbar = fig.colorbar(im, ax=ax, ticks=tick_positions)
    cbar.ax.set_yticklabels(tick_labels)

    if len(sequences) < 10:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i, j] != -1:
                    ax.text(
                        j, i, sequences[i][j],
                        ha='center', va='center', color='black'
                    )

    ax.set(xlabel='Time steps',yticks=[], title=title)
    fig.tight_layout()

    return fig


def grid_arrows(policy, title: str = None, color: str = "black"):
    assert policy.env_id == 'deep-sea-treasure-v0', f'Cannot visualize grid for environment {policy.env_id}'
    action_mapping = env_metadata[policy.env_id]['action_mapping']
    X, Y = policy.obs_space

    u = np.full((len(Y), len(X)), np.nan, dtype=float)
    v = np.full((len(Y), len(X)), np.nan, dtype=float)

    x0, y0 = X[0], Y[0]
    for state in policy.counts.keys():
        px = policy.action_probs(state)
        if px.max() - px.min() < 1e-6: continue

        y, x = state
        iy, ix = int(y - y0), int(x - x0)
        u[iy, ix], v[iy, ix] = action_mapping[np.argmax(px)]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.quiver(X, Y, u, v, angles="xy", scale_units="xy", scale=1, width=0.015, color=color)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    if title is not None:
        ax.set(title=title, xlabel='y', ylabel='x')
    ax.grid()
    fig.tight_layout()

    return fig


def grid_trajectories(observations, space=(11,11), title="Trajectory Density over Grid", alpha=0.05, linewidth=1.5, color='red'):
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
    fig.tight_layout()
    return fig


def cluster_scatter(data, labels, color_id: int=0, graph_labels: tuple=None, precomputed: bool=True):
    if data.shape[1] > 2: data = tsne_transform(data, precomputed=precomputed)

    fig, ax = plt.subplots(figsize=(8, 6))
    c = colors[color_id]

    for l in set(labels):
        x, y = data[labels == l, 0], data[labels == l, 1]
        ax.scatter(x, y, label=f"Cluster {l + 1}", color=c[l], linewidth=8)

    if graph_labels is not None:
        title, x_label, y_label = graph_labels
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)

    ax.legend()
    ax.grid()
    ax.tick_params(labelbottom=False, labelleft=False)
    fig.tight_layout()
    return fig


def decision_tree(obs: list, acs: list, env_id: str|None=None, max_depth: int=8, title: str|None=None):
    feature_names = env_metadata[env_id]['feature_names'] if env_id else [f"x{i}" for i in range(len(obs[0]))]
    action_names = env_metadata[env_id]["actions"] if env_id else None

    clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=20, ccp_alpha=0.01)
    clf.fit(obs, acs)

    graph, node_labels, edge_labels, _ = tree_to_graph(clf.tree_, clf.classes_, feature_names, action_names)
    pos = graphviz_layout(graph, prog="dot")

    fig, ax = plt.subplots()
    nx.draw(graph, pos, ax=ax, with_labels=False, node_size=3500, node_color='darkorange',)
    nx.draw_networkx_labels(graph, pos, labels=node_labels, ax=ax, font_size=10)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, ax=ax, font_size=9)

    if title is not None: ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()

    return fig