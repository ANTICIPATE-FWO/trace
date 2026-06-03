import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from trace.core import TrajectoryManager
from behavior.condiitoning import uniform_quantization
from trace.visuals.utils import env_frame, tsne_transform, tree_to_graph, tree_features

from sklearn.tree import DecisionTreeClassifier


def temporal_alignment(action_seq:list|np.ndarray, action_names:dict, time_range: tuple|None=None, title: str|None=None):
    if time_range:
        start, end = time_range
        max_len = end - start
    else:
        max_len = max(len(seq) for seq in action_seq)
        start, end = 0, max_len - 1


    matrix = np.full((len(action_seq), max_len), fill_value=-1)

    for i, seq in enumerate(action_seq):
        for j in range(start, min(end, len(seq))):
            matrix[i, j - start] = seq[j]


    cmap = plt.cm.get_cmap('viridis', len(action_names))
    cmap.set_bad(color='black')

    masked_matrix = np.ma.masked_where(matrix == -1, matrix)

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(masked_matrix, aspect='auto', cmap=cmap, interpolation='nearest')

    tick_positions = list(action_names.keys())
    tick_labels = list(action_names.values())

    fig.colorbar(im, ax=ax, ticks=tick_positions).ax.set_yticklabels(tick_labels)

    if len(action_seq) < 10:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i, j] != -1:
                    original_j = j + start if time_range is not None else j
                    ax.text(
                        j, i, action_seq[i][original_j],
                        ha='center', va='center', color='black'
                    )

    if title: ax.set_title(title)
    xticks = np.linspace(start, end - 1, 5, dtype=int)
    ax.set(xlabel='Time steps', yticks=[], xticks=xticks, xticklabels=xticks)
    fig.tight_layout()

    return fig


def grid_trajectories(manager:TrajectoryManager, abstr_frame:bool=False, step_size:float|int=1,
                      title: str|None=None, alpha: float|int=0.1, color: str='red'):

    frame = plt.imread('plots/sketches/dst_frame_abstr.png') if abstr_frame else env_frame(manager.metadata['env_id'])
    h, w = np.array(manager.metadata['observations_high'])[:2]
    fig, ax = plt.subplots()
    ax.imshow(frame, extent=(0, w, 0, h), origin='lower')

    for obs, acs, _ in zip(*manager.conditioning_features(flatten=True)):

        if len(obs) < 2: continue

        x_obs, y_obs = [], []
        x_acs, y_acs = [], []

        for coords, a in zip(obs, acs):
            y, x = coords[:2]
            x_obs.append(x)
            y_obs.append(y)

            if a == 0:
                x_acs.append(x)
                y_acs.append(y)

        ax.plot(x_obs, y_obs, alpha=alpha, linewidth=1.5, color=color)

        if 'minetrain' in manager.metadata['env_id']:
            ax.scatter(x_acs, y_acs, marker='x', color='blue', s=20)

    ax.set(xlim=(0, w), ylim=(h, 0), aspect='equal')
    ax.axis('off')
    if title: ax.set_title(title)
    fig.tight_layout(rect=(0, 0, 1, 1))
    return fig


def cluster_scatter(features:np.ndarray, labels:list, colors:list, title:str|None=None, precomputed:bool=True):
    if features.shape[1] > 2: features = tsne_transform(features, precomputed=precomputed)
    fig, ax = plt.subplots(figsize=(8, 6))

    for l in set(labels):
        x, y = features[labels == l, 0], features[labels == l, 1]
        ax.scatter(x, y, label=f"Cluster {l + 1}", color=colors[l % len(colors)], linewidth=8)

    ax.legend()
    ax.grid()
    ax.tick_params(labelbottom=False, labelleft=False)
    if title: ax.set_title(title)
    fig.tight_layout()
    return fig


def decision_tree(obs: list, acs: list, metadata: dict, max_depth: int=8, title: str|None=None):
    assert len(obs) == len(acs), f'Length mismatch {len(obs)} != {len(acs)}.'
    obs, acs = tree_features(obs, acs)

    clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=20, ccp_alpha=0.01)
    clf.fit(obs, acs)

    graph, nodes, edges, _ = tree_to_graph(
        clf.tree_, clf.classes_, metadata['observations_features'],metadata['actions']
    )
    pos = nx.drawing.nx_pydot.graphviz_layout(graph, prog="dot")

    fig, ax = plt.subplots()
    nx.draw(graph, pos, ax=ax, with_labels=False, node_color='white',)
    nx.draw_networkx_labels(graph, pos, labels=nodes, ax=ax)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edges, ax=ax)

    ax.axis("off")
    if title: ax.set_title(title)
    fig.tight_layout()

    return fig


def heatmap(visited:list, entr:np.ndarray, shape:tuple=(11, 11), v:tuple|None=None, title: str|None=None):
    heat = np.full(shape, np.nan)
    for s, e in zip(visited, entr): heat[*s[:2]] = e

    fig, ax = plt.subplots(figsize=(8, 6))
    square = shape[0] == shape[1]
    vmin, vmax = v if v is not None else None, None
    sns.heatmap(heat, ax=ax, vmin=vmin, vmax=vmax, square=square, cbar=True)

    ax.invert_yaxis()
    ax.tick_params(labelbottom=False, labelleft=False)

    if title: ax.set_title(title)

    fig.tight_layout()
    return fig