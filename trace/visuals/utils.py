from itertools import count
import networkx as nx
import numpy as np
from sklearn.manifold import TSNE


def tsne_transform(data, precomputed: bool = False, perplexity: int = 30):
    if precomputed: assert data.shape[0] == data.shape[1], f"Invalid similarity matrix shape: {data.shape}"

    tsne = TSNE(
        n_components = 2,
        perplexity = min(perplexity, len(data)-1),
        metric = 'precomputed' if precomputed else 'euclidean',
        init = 'random' if precomputed else 'pca',
        learning_rate = "auto",
        random_state = 42
    )

    return tsne.fit_transform(data)


def dst_frame():
    import warnings
    import mo_gymnasium as mo_gym

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        render_env = mo_gym.make("deep-sea-treasure-v0", render_mode="rgb_array")
        render_env.reset()
        frame = render_env.render()

    return frame


def tree_to_graph(tree, actions, feature_names, action_names=None, collapse=True, root=0):
    graph = nx.DiGraph()
    node_labels, edge_labels = {}, {}
    node_ids = count()

    def build(node_idx):
        left, right = tree.children_left[node_idx], tree.children_right[node_idx]

        if left == right:
            pred = actions[np.argmax(tree.value[node_idx][0])]
            node_id = next(node_ids)

            graph.add_node(node_id)
            node_labels[node_id] = str(action_names[pred]) if action_names else f"class = {pred}"
            return node_id, pred

        left_id, left_pred = build(left)
        right_id, right_pred = build(right)

        if collapse and left_pred == right_pred:
            return left_id, left_pred

        node_id = next(node_ids)
        graph.add_node(node_id)
        node_labels[node_id] = f"{feature_names[tree.feature[node_idx]]} < {tree.threshold[node_idx]:.2f}"

        graph.add_edge(node_id, left_id)
        graph.add_edge(node_id, right_id)
        edge_labels[(node_id, left_id)] = "yes"
        edge_labels[(node_id, right_id)] = "no"

        return node_id, None

    root_id, _ = build(root)
    return graph, node_labels, edge_labels, root_id