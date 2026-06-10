from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from trace.visuals.utils import env_frame, tsne_transform, tree_to_graph, tree_features

from sklearn.tree import DecisionTreeClassifier


def temporal_alignment(action_seq:list|np.ndarray, action_names:dict, time_range:tuple|None=None, title:str|None=None):
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
    masked_matrix = np.ma.masked_where(matrix == -1, matrix)

    cmap = plt.cm.get_cmap('viridis', len(action_names))
    cmap.set_bad(color='black')
    fig, ax = plt.subplots()
    im = ax.imshow(masked_matrix, aspect='auto', cmap=cmap, interpolation='nearest')

    ticks, labels = list(action_names.keys()), list(action_names.values())
    fig.colorbar(im, ax=ax, ticks=ticks).ax.set_yticklabels(labels)

    if title: ax.set_title(title)
    xticks = np.linspace(start, end - 1, 5, dtype=int)
    ax.set(xlabel='Time steps', yticks=[], xticks=xticks, xticklabels=xticks)
    fig.tight_layout()
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


def decision_tree(obs:list, acs:list, metadata:dict, max_depth:int=8, title:str|None=None):
    assert len(obs) == len(acs), f'Length mismatch {len(obs)} != {len(acs)}.'
    obs, acs = tree_features(obs, acs)
    clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=20, ccp_alpha=0.01).fit(obs, acs)

    features, actions = metadata['observations_features'], metadata['actions']
    graph, nodes, edges, _ = tree_to_graph(clf.tree_, clf.classes_, features, actions)
    pos = nx.drawing.nx_pydot.graphviz_layout(graph, prog="dot")

    fig, ax = plt.subplots()
    nx.draw(graph, pos, ax=ax, with_labels=False, node_color='white',)
    nx.draw_networkx_labels(graph, pos, labels=nodes, ax=ax)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edges, ax=ax)

    ax.axis("off")
    if title: ax.set_title(title)
    fig.tight_layout()

    return fig


def heatmap(h:np.ndarray, visited:list|None=None, shape:tuple=(11, 11), bounds:tuple[float|int,float|int]|None=None, title:str|None=None):
    #todo remove sparsity provision (tbd)
    if visited:
        heat = np.full(shape, np.nan)
        for s, e in zip(visited, h): heat[*s[:2]] = e
    else: heat = h

    square = shape[0] == shape[1]

    fig, ax = plt.subplots()
    vmin, vmax = bounds if bounds is not None else (None, None)
    sns.heatmap(heat, ax=ax, vmin=vmin, vmax=vmax, square=square, cbar=True)

    ax.invert_yaxis()
    ax.tick_params(labelbottom=False, labelleft=False)

    if title: ax.set_title(title)

    fig.tight_layout()
    return fig


def distribution_bar(values:list|np.ndarray, title:str|None=None, y_range:tuple|None=None):
    counts = Counter(values)
    keys, vals = list(counts.keys()), list(counts.values())

    if len(set(values)) < 20:
        for k, v in counts.items():
            print(f'{k:.3f}: {v}')

    fig, ax = plt.subplots()
    ax.bar(keys, vals, width=0.1)
    ax.set(xlabel='Values', ylabel='Observed times')
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(10))
    if title: ax.set_title('Distribution: ' + title)
    if y_range is not None: ax.set_ylim(*y_range)
    fig.tight_layout()
    return fig