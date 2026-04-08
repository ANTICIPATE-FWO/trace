import numpy as np
import networkx as nx


def similarity_graph(sim_matrix: np.ndarray, threshold: float = 0.5):
    assert sim_matrix.shape[0] == sim_matrix.shape[1], "Matrix must be square"

    n = sim_matrix.shape[0]
    graph = nx.Graph()
    graph.add_nodes_from(range(n))

    for i in range(n):
        for j in range(i+1, n):
            if sim_matrix[i, j] >= threshold and i != j: graph.add_edge(i, j)

    return graph


def connectivity_sweep(sim_matrix: np.ndarray|list, step: float = 0.1):
    num_components, thresholds = [], [step*i for i in range(int(1 / step))]
    assert len(thresholds) > 0, f'Empty threshold list, {step} step invalid'

    for t in thresholds:
        graph = similarity_graph(np.array(sim_matrix), t)
        num_components.append(nx.number_connected_components(graph))

    return num_components


def component_labels(sim_matrix: np.ndarray, threshold: float = 0.5):
    assert sim_matrix.shape[0] == sim_matrix.shape[1], "Matrix must be square"

    graph = similarity_graph(sim_matrix, threshold)
    labels = [None] * sim_matrix.shape[0]

    for component_id, component in enumerate(nx.connected_components(graph)):
        for node in component: labels[node] = component_id

    return labels